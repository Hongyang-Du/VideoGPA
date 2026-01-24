import os
import sys
import logging
import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
import datetime
from typing import Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import wandb
from torch.utils.data import DataLoader, random_split

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXTransformer3DModel,
    CogVideoXDPMScheduler,
)
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# Ensure project custom modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dpo_cogvideox.dataset import DPODataset, collate_fn
from dpo_cogvideox.loss import create_loss_strategy
# ============================================================================
# ⭐ Default Configuration 
# ============================================================================

DATASET_PATH = os.environ.get('DATASET_PATH', '/path/to/your/dataset') # Replace with your actual dataset path

DEFAULT_CONFIG = {
    'devices': [0, 1, 2, 3, 4, 5, 6, 7],
    'metadata_path': f'{DATASET_PATH}/meta_data.json', # Path to metadata JSON file
    'model_path': 'THUDM/CogVideoX-5B-I2V',
    'output_dir':  'your/outputs/root', # Replace with your desired output directory
    'base_path': DATASET_PATH,

    # DPO Dataset Configuration
    'metric_name': 'consistency_score', 
    'metric_mode': 'min',
    'min_gap': 0.05,    
    'metric_threshold': 0.8, 
    'motion_threshold': 0.001,

    
    # Training Configuration
    'learning_rate': 5e-6,
    'beta': 1.0,
    'max_epochs': 100,
    'max_steps': 10000,
    'warmup_steps': 500,
    'batch_size': 2,
    'accumulate_grad_batches': 1,
    'gradient_clip_val': 1.0,

    # LoRA Configuration
    'lora_rank': 64,
    'lora_alpha': 128.0,
    'lora_dropout': 0.0,
    'lora_target_modules': ['to_q', 'to_k', 'to_v', 'to_out.0'],

    # Logging and Checkpointing
    'experiment_name': 'cogvideo_dpo_multigpu',
    'wandb_project': 'cogvideox-i2v-dpo', # Added your project name
    'checkpoint_every_n_steps': 1000,
    'log_every_n_steps': 10,
    'save_top_k': 10,

    # Optimization Switches
    'enable_gradient_checkpointing': True,
    'enable_slicing': True,
    'enable_tiling': True,
}

# ============================================================================
# PyTorch Lightning Training Module
# ============================================================================

class CogVideoXDPOTrainer(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.start_time = None
        self.total_samples = 0
        # 1. Load Models
        self.vae = AutoencoderKLCogVideoX.from_pretrained(config['model_path'], subfolder="vae", torch_dtype=torch.bfloat16)
        self.vae.requires_grad_(False).eval()
        if config.get('enable_slicing'): self.vae.enable_slicing()
        if config.get('enable_tiling'): self.vae.enable_tiling()

        self.transformer = CogVideoXTransformer3DModel.from_pretrained(config['model_path'], subfolder="transformer", torch_dtype=torch.bfloat16)
        lora_config = LoraConfig(
            r=config['lora_rank'], lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'], target_modules=config['lora_target_modules'],
        )
        self.transformer = get_peft_model(self.transformer, lora_config)
        if config.get('enable_gradient_checkpointing'):
            self.transformer.enable_gradient_checkpointing()

        self.ref_transformer = CogVideoXTransformer3DModel.from_pretrained(config['model_path'], subfolder="transformer", torch_dtype=torch.bfloat16)
        self.ref_transformer.requires_grad_(False).eval()

        self.scheduler = CogVideoXDPMScheduler.from_pretrained(config['model_path'], subfolder="scheduler")
        self.loss_fn = create_loss_strategy(strategy='dpo', beta=config['beta'])

    def _shared_step(self, batch):
        """Common Loss Calculation Logic, reused for Training and Validation"""
        x_win = batch['x_win'].permute(0, 2, 1, 3, 4)
        x_lose = batch['x_lose'].permute(0, 2, 1, 3, 4)
        prompt_emb = batch['prompt_emb']
        image_emb = batch.get('image_emb')

        if image_emb is not None:
            with torch.no_grad():
                img_resized = F.interpolate(image_emb, size=(x_win.shape[3]*8, x_win.shape[4]*8))
                img_dist = self.vae.encode(img_resized.unsqueeze(2).to(self.vae.dtype)).latent_dist
                img_lat = img_dist.sample() * self.vae.config.scaling_factor
                img_lat = img_lat.permute(0, 2, 1, 3, 4)
                padding = torch.zeros(x_win.shape[0], x_win.shape[1]-1, *img_lat.shape[2:], device=self.device, dtype=img_lat.dtype)
                img_cond = torch.cat([img_lat, padding], dim=1)
        else:
            img_cond = torch.zeros_like(x_win)

        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x_win.shape[0],), device=self.device)
        noise = torch.randn_like(x_win)

        x_win_noisy = torch.cat([self.scheduler.add_noise(x_win, noise, timesteps), img_cond], dim=2)
        x_lose_noisy = torch.cat([self.scheduler.add_noise(x_lose, noise, timesteps), img_cond], dim=2)

        v_win_pred = self.transformer(x_win_noisy, encoder_hidden_states=prompt_emb, timestep=timesteps, return_dict=True).sample
        v_lose_pred = self.transformer(x_lose_noisy, encoder_hidden_states=prompt_emb, timestep=timesteps, return_dict=True).sample
        
        with torch.no_grad():
            v_win_ref = self.ref_transformer(x_win_noisy, encoder_hidden_states=prompt_emb, timestep=timesteps, return_dict=True).sample
            v_lose_ref = self.ref_transformer(x_lose_noisy, encoder_hidden_states=prompt_emb, timestep=timesteps, return_dict=True).sample

        v_win_target = self.scheduler.get_velocity(x_win, noise, timesteps)
        v_lose_target = self.scheduler.get_velocity(x_lose, noise, timesteps)

        return self.loss_fn(v_win_pred, v_lose_pred, v_win_ref, v_lose_ref, v_win_target, v_lose_target)

    def training_step(self, batch, batch_idx):
        if self.start_time is None:
            self.start_time = datetime.datetime.now()

        loss_out = self._shared_step(batch)
        log_kwargs = {
        "sync_dist": True, 
        "on_step": True,    # Enable real-time monitoring
        "on_epoch": False, 
        "batch_size": self.config['batch_size']
        }
        reward_acc = (loss_out.reward_margin > 0).float().mean()
        self.log('train/loss', loss_out.loss, **log_kwargs)
        self.log('train/reward_margin', loss_out.reward_margin, **log_kwargs)
        self.log('train/reward_accuracy', reward_acc, prog_bar=True, sync_dist=True, on_step=True)
        
        if self.global_step % self.config.get('log_every_n_steps', 10) == 0:
            # Memory usage (GB)
            max_mem = torch.cuda.max_memory_reserved() / (1024**3)
            self.log('stats/max_memory_gb', max_mem, sync_dist=True)

            # Throughput (Samples per second)
            if self.start_time is not None:
                elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
                # Global samples = step * devices * batch_size
                current_total_samples = self.global_step * len(self.config['devices']) * self.config['batch_size']
                throughput = current_total_samples / elapsed_time if elapsed_time > 0 else 0
                self.log('stats/samples_per_sec', throughput, sync_dist=True)

        return loss_out.loss

    def validation_step(self, batch, batch_idx):
        # No gradients during validation
        with torch.no_grad():
            loss_out = self._shared_step(batch)

        val_reward_acc = (loss_out.reward_margin > 0).float().mean()
        self.log('val/loss', loss_out.loss, 
             prog_bar=True, 
             sync_dist=True, 
             on_step=False,
             on_epoch=True,
             batch_size=self.config['batch_size'])
        self.log('val/reward_margin', loss_out.reward_margin, 
             sync_dist=True, on_epoch=True, 
             batch_size=self.config['batch_size'])
        self.log('val/reward_accuracy', val_reward_acc, sync_dist=True, on_epoch=True, batch_size=self.config['batch_size'])
        return loss_out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=self.config['learning_rate'])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.get('warmup_steps', 500), num_training_steps=self.config['max_steps']
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

# ============================================================================
# Main Training Entry Point
# ============================================================================

def main_train(config: Dict[str, Any]):
    # [ANONYMIZED]: Removed hardcoded API key. Use environment variable WANDB_API_KEY.
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    out_p = Path(config['output_dir'])
    ckpt_p = out_p / "checkpoints"
    ckpt_p.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        project=config.get('wandb_project', 'cogvideox-dpo'), 
        name=config.get('experiment_name'), 
        config=config
    )

    # 1. Dataset Preparation and Splitting
    full_dataset = DPODataset(
        base_path=config['base_path'],
        metadata_path=config['metadata_path'],
        metric_name=config.get('metric_name', 'consistency_score'),
        metric_mode=config.get('metric_mode', 'min'),
        min_gap=config.get('min_gap', 0.05),
        motion_threshold=config.get('motion_threshold', 0.001)
    )
    
    # Random Split: 98% Train, 2% Validation
    train_size = int(0.98 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], num_workers=4, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    model = CogVideoXDPOTrainer(config)

    # 2. Configure Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config['devices'],
        strategy=DDPStrategy(timeout=datetime.timedelta(seconds=600)),
        limit_val_batches=50,
        precision="bf16-mixed",
        max_steps=config['max_steps'],
        check_val_every_n_epoch= 1 ,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=DEFAULT_CONFIG.get('gradient_clip_val', 1.0),
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpt_p, 
                filename="{step}-{val/loss:.4f}", 
                monitor="val/loss", 
                mode="min",
                save_top_k=config['save_top_k'], 
                every_n_train_steps=config['checkpoint_every_n_steps']
            ),
            LearningRateMonitor(logging_interval='step')
        ],
        logger=wandb_logger,
        log_every_n_steps=config['log_every_n_steps']
    )

    logging.info(f"🚀 Starting Multi-GPU DPO Training with Validation Set...")
    # Pass both train and validation DataLoaders
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if trainer.is_global_zero:
        model.transformer.save_pretrained(out_p / "final_lora")
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--devices', type=str, default=None)
    parser.add_argument('--base_path', type=str, default="/path/to/dataset")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f).get('training', {})
            config.update(yaml_cfg)
    # Ensure metadata path is constructed relative to potentially overridden base_path
    config['metadata_path'] = f'{config["base_path"]}/meta_data.json'
    if args.devices:
        config['devices'] = [int(d) for d in args.devices.split(',')]

    main_train(config)
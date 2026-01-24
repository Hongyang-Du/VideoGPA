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

DATASET_PATH = os.environ.get('DATASET_PATH', '/path/to/dataset') # Replace with your actual dataset path

DEFAULT_CONFIG = {
    'devices': [0, 1, 2, 3, 4, 5, 6, 7],
    'metadata_path': f'{DATASET_PATH}/your_meta_data_t2v.json', # Replace with your actual metadata path
    'model_path': 'THUDM/CogVideoX-5b',
    'output_dir': 'your/outputs/root',  # Replace with your desired output path
    'base_path': DATASET_PATH,

    # DPO Dataset Configuration
    'metric_name': 'consistency_score', # name of the metric use as signal
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
    'batch_size': 1,
    'accumulate_grad_batches': 2,
    'gradient_clip_val': 1.0,

    # LoRA Configuration
    'lora_rank': 64,
    'lora_alpha': 128.0,
    'lora_dropout': 0.0,
    'lora_target_modules': ['to_q', 'to_k', 'to_v', 'to_out.0'],

    # Logging and Checkpointing
    'experiment_name': 'cogvideo_dpo_t2v',
    'wandb_project': 'cogvideox-t2v-dpo', # Added your project name
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
        """Modified T2V DPO Logic"""
        # 1. Prepare Data (Batch, Frames, Channels, Height, Width)
        # Note: CogVideoX typically expects (B, C, F, H, W) or adjusted via permute
        x_win = batch['x_win'].permute(0, 2, 1, 3, 4)   # Result: [B, C, F, H, W]
        x_lose = batch['x_lose'].permute(0, 2, 1, 3, 4)
        prompt_emb = batch['prompt_emb'] # T5 Text Embeddings

        # 2. Generate Noise and Timesteps
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x_win.shape[0],), device=self.device)
        noise = torch.randn_like(x_win)

        # 3. Add Noise (T2V uses noise directly, no need to concatenate img_cond)
        x_win_noisy = self.scheduler.add_noise(x_win, noise, timesteps)
        x_lose_noisy = self.scheduler.add_noise(x_lose, noise, timesteps)

        # 4. Model Prediction (Transformer receives standard latent dimensions)
        # Note: CogVideoX internally handles conversion from (B, C, F, H, W) to Patches
        v_win_pred = self.transformer(
            hidden_states=x_win_noisy, 
            encoder_hidden_states=prompt_emb, 
            timestep=timesteps, 
            return_dict=True
        ).sample
        
        v_lose_pred = self.transformer(
            hidden_states=x_lose_noisy, 
            encoder_hidden_states=prompt_emb, 
            timestep=timesteps, 
            return_dict=True
        ).sample
        
        # 5. Reference Model Prediction
        with torch.no_grad():
            v_win_ref = self.ref_transformer(x_win_noisy, encoder_hidden_states=prompt_emb, timestep=timesteps, return_dict=True).sample
            v_lose_ref = self.ref_transformer(x_lose_noisy, encoder_hidden_states=prompt_emb, timestep=timesteps, return_dict=True).sample

        # 6. Calculate Target (Flow Matching / Velocity)
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
        # No gradient calculation during validation
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
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])
    
    out_p = Path(config['output_dir'])
    ckpt_p = out_p / "checkpoints"
    ckpt_p.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        project=config.get('wandb_project', 'cogvideox-t2v-dpo'), 
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
    
    # Split: 98% Train, 2% Validation
    train_size = int(0.98 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], num_workers=4, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    model = CogVideoXDPOTrainer(config)

    # 2. Trainer Configuration
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config['devices'],
        strategy=DDPStrategy(timeout=datetime.timedelta(seconds=600)),
        limit_val_batches=50,
        precision="bf16-mixed",
        max_steps=config['max_steps'],
        check_val_every_n_epoch= 1 ,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
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
    
    if args.devices:
        config['devices'] = [int(d) for d in args.devices.split(',')]

    main_train(config)
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
    CogVideoXTransformer3DModel, 
    CogVideoXDPMScheduler,
)
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dpo_cogvideox.dataset15 import DPODataset, collate_fn
from dpo_cogvideox.loss import create_loss_strategy

# ============================================================================
# ⭐ Default Configurations
# ============================================================================
DATASET_PATH = os.environ.get('DATASET_PATH', '/path/to/dataset') 

DEFAULT_CONFIG = {
    'devices': [0,1,2,3,4,5,6,7], # GPU device IDs
    'metadata_path': f'{DATASET_PATH}/your_meta_data_t2v.json',
    'model_path': 'THUDM/CogVideoX1.5-5B',
    'output_dir': 'your/outputs/root',
    'base_path': DATASET_PATH,

    # DPO dataset configs
    'metric_name': 'consistency_score',
    'metric_mode': 'min',
    'min_gap': 0.05,   
    'metric_threshold': 0.8, 
    'motion_threshold': 0.001,

    # Training hyperparameters
    'learning_rate': 5e-6, 
    'beta': 1.0,
    'max_epochs': 100,
    'max_steps': 1500,
    'warmup_steps': 500,
    'batch_size': 1,
    'accumulate_grad_batches': 2, 
    'gradient_clip_val': 1.0,

    # LoRA Configs
    'lora_rank': 64,
    'lora_alpha': 128.0,
    'lora_dropout': 0.0,
    'lora_target_modules': ['to_q', 'to_k', 'to_v', 'to_out.0'],

    'experiment_name': 'cogvideo1.5_dpo_t2v',
    'checkpoint_every_n_steps': 1000,
    'log_every_n_steps': 10,
    'save_top_k': 10,

    # Optimization Switches
    'enable_gradient_checkpointing': True,
}

# ============================================================================
# PyTorch Lightning Module
# ============================================================================

class CogVideoXDPOTrainer(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.start_time = None
        

        # 1. Transformer + LoRA
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            config['model_path'], 
            subfolder="transformer", 
            torch_dtype=torch.bfloat16
        )
        
        # ✅ Dynamic Positional Embedding
        self.transformer.config.use_dynamic_positional_embedding = True 

        lora_config = LoraConfig(
            r=config['lora_rank'], lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'], target_modules=config['lora_target_modules'],
        )
        self.transformer = get_peft_model(self.transformer, lora_config)
        
        if config.get('enable_gradient_checkpointing'):
            self.transformer.enable_gradient_checkpointing()

        # 2. Reference Model (Frozen)
        self.ref_transformer = CogVideoXTransformer3DModel.from_pretrained(
            config['model_path'], 
            subfolder="transformer", 
            torch_dtype=torch.bfloat16
        )
        self.ref_transformer.requires_grad_(False).eval()
        self.ref_transformer.config.use_dynamic_positional_embedding = True

        self.scheduler = CogVideoXDPMScheduler.from_pretrained(config['model_path'], subfolder="scheduler")
        self.loss_fn = create_loss_strategy(strategy='dpo', beta=config['beta'])

    def _shared_step(self, batch):
        """DPO logic adapted for Offline Latents (includes automatic cropping)"""
        # 1. Get Latent Data
        # x_win shape [B, C=16, F=21, H_lat, W_lat]
        x_win = batch['x_win'].to(dtype=torch.bfloat16)
        x_lose = batch['x_lose'].to(dtype=torch.bfloat16)
        prompt_emb = batch['prompt_emb'].to(dtype=torch.bfloat16)

        # [B, C, F, H, W] -> [B, F, C, H, W]
        if x_win.shape[1] == 16: 
            x_win = x_win.permute(0, 2, 1, 3, 4)   
            x_lose = x_lose.permute(0, 2, 1, 3, 4)

        
        B, F, C, H, W = x_win.shape
        
        # Calculate new dimensions that are even
        new_F = F - (F % 2)
        new_H = H - (H % 2)
        new_W = W - (W % 2)
        
        if new_F != F or new_H != H or new_W != W:
            # trim the tensors to the new dimensions
            x_win = x_win[:, :new_F, :, :new_H, :new_W]
            x_lose = x_lose[:, :new_F, :, :new_H, :new_W]
            
        # 2. generate random timesteps & noise
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x_win.shape[0],), device=self.device).long()
        noise = torch.randn_like(x_win)

        # 3. Add noise
        x_win_noisy = self.scheduler.add_noise(x_win, noise, timesteps)
        x_lose_noisy = self.scheduler.add_noise(x_lose, noise, timesteps)

        # 4. Model prediction
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
        
        # 5. Reference model prediction
        with torch.no_grad():
            v_win_ref = self.ref_transformer(
                hidden_states=x_win_noisy, 
                encoder_hidden_states=prompt_emb, 
                timestep=timesteps, 
                return_dict=True
            ).sample
            v_lose_ref = self.ref_transformer(
                hidden_states=x_lose_noisy, 
                encoder_hidden_states=prompt_emb, 
                timestep=timesteps, 
                return_dict=True
            ).sample

        # 6. calculate Loss
        v_win_target = self.scheduler.get_velocity(x_win, noise, timesteps)
        v_lose_target = self.scheduler.get_velocity(x_lose, noise, timesteps)

        return self.loss_fn(v_win_pred, v_lose_pred, v_win_ref, v_lose_ref, v_win_target, v_lose_target)

    def training_step(self, batch, batch_idx):
        if self.start_time is None: self.start_time = datetime.datetime.now()
        loss_out = self._shared_step(batch)
        
        reward_acc = (loss_out.reward_margin > 0).float().mean()
        
        self.log('train/loss', loss_out.loss, sync_dist=True, on_step=True, prog_bar=True)
        self.log('train/reward_margin', loss_out.reward_margin, sync_dist=True, on_step=True)
        self.log('train/reward_accuracy', reward_acc, sync_dist=True, on_step=True)
        
        return loss_out.loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss_out = self._shared_step(batch)
        
        val_reward_acc = (loss_out.reward_margin > 0).float().mean()
        self.log('val/loss', loss_out.loss, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log('val/reward_accuracy', val_reward_acc, sync_dist=True, on_epoch=True)
        return loss_out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=self.config['learning_rate'], weight_decay=1e-3)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.get('warmup_steps', 500), num_training_steps=self.config['max_steps']
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

# ============================================================================
# Main
# ============================================================================

def main_train(config: Dict[str, Any]):

    out_p = Path(config['output_dir'])
    ckpt_p = out_p / "checkpoints"
    ckpt_p.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(project="cogvideox1.5-dpo", name=config.get('experiment_name'), config=config)
    # 1. Dataset & Dataloader
    full_dataset = DPODataset(
        base_path=config['base_path'],
        metadata_path=config['metadata_path'],
        metric_name=config.get('metric_name', 'consistency_score'),
        metric_mode=config.get('metric_mode', 'min'),
        min_gap=config.get('min_gap', 0.05),
        motion_threshold=config.get('motion_threshold', 0.001)
    )
    
    train_size = int(0.98 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], num_workers=4, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    model = CogVideoXDPOTrainer(config)

    # 2. Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config['devices'],
        strategy=DDPStrategy(timeout=datetime.timedelta(seconds=1200)), 
        precision="bf16-mixed",
        max_steps=config['max_steps'],
        check_val_every_n_epoch=1,
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

    logging.info(f"🚀 Starting CogVideoX 1.5 T2V DPO Training (Offline Latents)...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 3. Save Final LoRA 
    if trainer.is_global_zero:
        print("💾 Saving Final LoRA...")
        model.transformer.save_pretrained(out_p / "final_lora")
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--devices', type=str, default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f).get('training', {})
            config.update(yaml_cfg)
    if args.devices:
        config['devices'] = [int(d) for d in args.devices.split(',')]

    main_train(config)
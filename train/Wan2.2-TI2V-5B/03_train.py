"""
Wan2.2 TI2V DPO Training Script

Uses flow matching formulation:
  - z_t = (1 - sigma) * z_0 + sigma * noise
  - Model predicts velocity: v = noise - z_0
  - For TI2V: first temporal frame is clean image latent (sigma=0)

Architecture:
  - WanModel (DiT): 30 layers, dim=3072, in_dim=48, out_dim=48
  - VAE z_dim=48, stride=(4, 16, 16)
  - T5 text embeddings: [L, 4096]
  - LoRA fine-tuning on DiT attention layers
"""

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
from copy import deepcopy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import wandb
from torch.utils.data import DataLoader, random_split
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint as torch_checkpoint

current_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, train_dir)
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../..')))

wan_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'Wan2.2'))
sys.path.insert(0, wan_path)

from wan.modules.model import WanModel
from dataset import DPODataset, collate_fn
from loss import create_loss_strategy

# ============================================================================
# Default Config
# ============================================================================

DEFAULT_CONFIG = {
    'devices': [0, 1, 2, 3, 4, 5, 6, 7],

    # DPO dataset config
    'metric_name': 'consistency_score',
    'metric_mode': 'min',
    'min_gap': 0.05,
    'metric_threshold': 0.8,
    'motion_threshold': 0.001,

    # Training config
    'learning_rate': 5e-6,
    'beta': 1.0,
    'max_epochs': 100,
    'max_steps': 10000,
    'warmup_steps': 500,
    'batch_size': 1,
    'accumulate_grad_batches': 2,
    'gradient_clip_val': 1.0,

    # Flow matching
    'num_train_timesteps': 1000,
    'shift': 5.0,

    # LoRA config
    'lora_rank': 64,
    'lora_alpha': 128.0,
    'lora_dropout': 0.0,
    'lora_target_modules': ['q', 'k', 'v', 'o'],

    # Logging and saving
    'experiment_name': 'wan_ti2v_dpo',
    'wandb_project': 'wan-dpo',
    'checkpoint_every_n_steps': 1000,
    'log_every_n_steps': 10,
    'save_top_k': 10,

    # VAE config (for reference)
    'vae_stride': (4, 16, 16),
    'patch_size': (1, 2, 2),

    # Optimization
    'enable_gradient_checkpointing': True,
}

# ============================================================================
# Flow Matching Utilities
# ============================================================================

def get_sigma_from_timestep(timestep, num_train_timesteps=1000, shift=5.0):
    sigma = timestep.float() / num_train_timesteps
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


def flow_matching_add_noise(z_0, noise, sigma):
    while sigma.dim() < z_0.dim():
        sigma = sigma.unsqueeze(-1)
    return (1.0 - sigma) * z_0 + sigma * noise


def flow_matching_get_velocity(z_0, noise):
    return noise - z_0


def create_ti2v_timestep_tensor(timestep, mask2, seq_len, patch_size):
    temp_ts = (mask2[0][:, ::patch_size[1], ::patch_size[2]] * timestep).flatten()
    temp_ts = torch.cat([
        temp_ts,
        temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
    ])
    return temp_ts.unsqueeze(0)

# ============================================================================
# Training Module
# ============================================================================

class WanDPOTrainer(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.start_time = None

        # Load DiT model with LoRA
        logging.info(f"Loading WanModel from {config['model_path']}")
        self.transformer = WanModel.from_pretrained(config['model_path'])
        self.transformer.to(torch.bfloat16)

        lora_config = LoraConfig(
            r=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=config['lora_target_modules'],
        )
        self.transformer = get_peft_model(self.transformer, lora_config)

        if config.get('enable_gradient_checkpointing'):
            base = self.transformer.base_model.model if hasattr(self.transformer, 'base_model') else self.transformer
            for block in base.blocks:
                orig_forward = block.forward
                def _make_ckpt_forward(fn):
                    def _ckpt_forward(*args, **kwargs):
                        return torch_checkpoint(fn, *args, use_reentrant=False, **kwargs)
                    return _ckpt_forward
                block.forward = _make_ckpt_forward(orig_forward)
            logging.info("Gradient checkpointing enabled on WanAttentionBlocks")

        self.transformer.print_trainable_parameters()

        # Reference model (frozen copy)
        logging.info("Loading reference model...")
        self.ref_transformer = WanModel.from_pretrained(config['model_path'])
        self.ref_transformer.to(torch.bfloat16)
        self.ref_transformer.requires_grad_(False)
        self.ref_transformer.eval()

        self.loss_fn = create_loss_strategy(strategy='dpo', beta=config['beta'])

        self.num_train_timesteps = config.get('num_train_timesteps', 1000)
        self.shift = config.get('shift', 5.0)
        self.vae_stride = config.get('vae_stride', (4, 16, 16))
        self.patch_size = config.get('patch_size', (1, 2, 2))

    def _compute_seq_len(self, z):
        _, _, f, h, w = z.shape
        seq_len = f * (h // self.patch_size[1]) * (w // self.patch_size[2])
        return seq_len

    def _create_mask(self, z_shape, device):
        z_dim, f, h, w = z_shape
        mask = torch.ones(z_dim, f, h, w, device=device)
        mask[:, 0] = 0.0
        return mask

    def _shared_step(self, batch):
        x_win = batch['x_win']
        x_lose = batch['x_lose']
        prompt_emb = batch['prompt_emb']
        image_latent = batch.get('image_latent')

        B = x_win.shape[0]
        seq_len = self._compute_seq_len(x_win)

        timesteps = torch.randint(
            1, self.num_train_timesteps,
            (B,), device=self.device
        )
        sigma = get_sigma_from_timestep(timesteps, self.num_train_timesteps, self.shift)

        noise = torch.randn_like(x_win)

        x_win_noisy = flow_matching_add_noise(x_win, noise, sigma)
        x_lose_noisy = flow_matching_add_noise(x_lose, noise, sigma)

        if image_latent is not None:
            x_win_noisy[:, :, 0:1] = image_latent
            x_lose_noisy[:, :, 0:1] = image_latent

        mask2 = self._create_mask(x_win.shape[1:], self.device)
        t_tensors = []
        for b in range(B):
            t_tensor = create_ti2v_timestep_tensor(
                timesteps[b], mask2, seq_len, self.patch_size
            )
            t_tensors.append(t_tensor)
        t_batch = torch.cat(t_tensors, dim=0)

        context_list = [prompt_emb[b] for b in range(B)]

        x_win_input = [x_win_noisy[b] for b in range(B)]
        x_lose_input = [x_lose_noisy[b] for b in range(B)]

        with torch.no_grad():
            v_win_ref = torch.stack(self.ref_transformer(x_win_input, t=t_batch, context=context_list, seq_len=seq_len))
            v_lose_ref = torch.stack(self.ref_transformer(x_lose_input, t=t_batch, context=context_list, seq_len=seq_len))
        torch.cuda.empty_cache()

        v_win_pred = torch.stack(self.transformer(x_win_input, t=t_batch, context=context_list, seq_len=seq_len))
        v_lose_pred = torch.stack(self.transformer(x_lose_input, t=t_batch, context=context_list, seq_len=seq_len))

        v_win_target = flow_matching_get_velocity(x_win, noise)
        v_lose_target = flow_matching_get_velocity(x_lose, noise)

        return self.loss_fn(
            v_win_pred, v_lose_pred,
            v_win_ref, v_lose_ref,
            v_win_target, v_lose_target
        )

    def training_step(self, batch, batch_idx):
        if self.start_time is None:
            self.start_time = datetime.datetime.now()

        loss_out = self._shared_step(batch)

        log_kwargs = {
            "sync_dist": True,
            "on_step": True,
            "on_epoch": False,
            "batch_size": self.config['batch_size']
        }
        reward_acc = (loss_out.reward_margin > 0).float().mean()
        self.log('train/loss', loss_out.loss, **log_kwargs)
        self.log('train/reward_margin', loss_out.reward_margin, **log_kwargs)
        self.log('train/reward_accuracy', reward_acc, prog_bar=True, sync_dist=True, on_step=True)

        if self.global_step % self.config.get('log_every_n_steps', 10) == 0:
            max_mem = torch.cuda.max_memory_reserved() / (1024**3)
            self.log('stats/max_memory_gb', max_mem, sync_dist=True)

            if self.start_time is not None:
                elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
                total_samples = self.global_step * len(self.config['devices']) * self.config['batch_size']
                throughput = total_samples / elapsed if elapsed > 0 else 0
                self.log('stats/samples_per_sec', throughput, sync_dist=True)

        return loss_out.loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss_out = self._shared_step(batch)

        val_reward_acc = (loss_out.reward_margin > 0).float().mean()
        self.log('val/loss', loss_out.loss,
                 prog_bar=True, sync_dist=True,
                 on_step=False, on_epoch=True,
                 batch_size=self.config['batch_size'])
        self.log('val/reward_margin', loss_out.reward_margin,
                 sync_dist=True, on_epoch=True,
                 batch_size=self.config['batch_size'])
        self.log('val/reward_accuracy', val_reward_acc,
                 sync_dist=True, on_epoch=True,
                 batch_size=self.config['batch_size'])
        return loss_out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=self.config['learning_rate']
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.get('warmup_steps', 500),
            num_training_steps=self.config['max_steps']
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        }

# ============================================================================
# Main
# ============================================================================

def main_train(config: Dict[str, Any]):
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    out_p = Path(config['output_dir'])
    ckpt_p = out_p / "checkpoints"
    ckpt_p.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        project=config.get('wandb_project', 'wan-dpo'),
        name=config.get('experiment_name'),
        config=config
    )

    full_dataset = DPODataset(
        base_path=config['base_path'],
        metadata_path=config['metadata_path'],
        metric_name=config.get('metric_name', 'consistency_score'),
        metric_mode=config.get('metric_mode', 'min'),
        min_gap=config.get('min_gap', 0.05),
        motion_threshold=config.get('motion_threshold', 0.001),
    )

    train_size = int(0.98 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'],
        num_workers=4, shuffle=True, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        num_workers=4, shuffle=False, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
        collate_fn=collate_fn
    )

    model = WanDPOTrainer(config)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config['devices'],
        strategy=DDPStrategy(
            timeout=datetime.timedelta(seconds=1800),
            find_unused_parameters=False,
        ),
        limit_val_batches=50,
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

    logging.info("Starting Wan2.2 TI2V DPO Training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if trainer.is_global_zero:
        model.transformer.save_pretrained(out_p / "final_lora")
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V DPO Training")
    parser.add_argument('--config', type=str, default=None, help="YAML config file")
    parser.add_argument('--devices', type=str, default=None, help="GPU IDs, comma-separated")
    parser.add_argument('--base_path', type=str, required=True, help="Base dataset path")
    parser.add_argument('--model_path', type=str, required=True, help="Wan2.2-TI2V-5B model path")
    parser.add_argument('--metadata_path', type=str, default=None,
                        help="Metadata JSON path (default: <base_path>/meta_wan_data.json)")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Output directory (default: <base_path>/outputs/wan_dpo)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()

    if args.config:
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f).get('training', {})
            config.update(yaml_cfg)

    config['base_path'] = args.base_path
    config['model_path'] = args.model_path
    config['metadata_path'] = args.metadata_path or f'{args.base_path}/meta_wan_data.json'
    config['output_dir'] = args.output_dir or f'{args.base_path}/outputs/wan_dpo'

    if args.devices:
        config['devices'] = [int(d) for d in args.devices.split(',')]

    main_train(config)

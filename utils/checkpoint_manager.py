"""
Checkpoint Manager
Handles saving and loading of model checkpoints with metadata.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints with versioning and metadata tracking.
    """

    def __init__(self, checkpoint_dir: str = 'ckpts'):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.metadata_file = self.checkpoint_dir / 'checkpoint_metadata.json'
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load existing metadata or create new."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'checkpoints': [], 'best': {}}

    def _save_metadata(self) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        epoch: int, metrics: Dict[str, float], module_name: str,
                        is_best: bool = False) -> str:
        """
        Save model checkpoint with metadata.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Validation metrics
            module_name: Name of the module
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{module_name}_epoch{epoch}_{timestamp}.pt'
        filepath = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'module_name': module_name,
            'timestamp': timestamp
        }

        torch.save(checkpoint, filepath)

        # Update metadata
        ckpt_info = {
            'filename': filename,
            'module': module_name,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': timestamp,
            'is_best': is_best
        }
        self.metadata['checkpoints'].append(ckpt_info)

        if is_best:
            self.metadata['best'][module_name] = filename
            # Also save as best checkpoint
            best_path = self.checkpoint_dir / f'{module_name}_best.pt'
            torch.save(checkpoint, best_path)

        self._save_metadata()
        return str(filepath)

    def load_checkpoint(self, filepath: str, model: torch.nn.Module,
                        optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Load checkpoint into model.

        Args:
            filepath: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optional optimizer to load state into

        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(filepath, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'module_name': checkpoint.get('module_name', 'unknown')
        }

    def load_best(self, module_name: str, model: torch.nn.Module,
                  optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Load best checkpoint for a module.

        Args:
            module_name: Name of the module
            model: Model to load weights into
            optimizer: Optional optimizer

        Returns:
            Checkpoint metadata
        """
        if module_name not in self.metadata['best']:
            raise ValueError(f"No best checkpoint found for module: {module_name}")

        filename = self.metadata['best'][module_name]
        filepath = self.checkpoint_dir / filename
        return self.load_checkpoint(str(filepath), model, optimizer)

    def get_latest_checkpoint(self, module_name: str) -> Optional[str]:
        """Get path to latest checkpoint for a module."""
        module_ckpts = [c for c in self.metadata['checkpoints'] if c['module'] == module_name]
        if not module_ckpts:
            return None
        latest = sorted(module_ckpts, key=lambda x: x['timestamp'])[-1]
        return str(self.checkpoint_dir / latest['filename'])

    def list_checkpoints(self, module_name: Optional[str] = None) -> list:
        """List all checkpoints, optionally filtered by module."""
        ckpts = self.metadata['checkpoints']
        if module_name:
            ckpts = [c for c in ckpts if c['module'] == module_name]
        return ckpts

    def cleanup_old_checkpoints(self, module_name: str, keep_n: int = 3) -> None:
        """
        Remove old checkpoints, keeping only the most recent N.

        Args:
            module_name: Module to clean up
            keep_n: Number of checkpoints to keep
        """
        module_ckpts = [c for c in self.metadata['checkpoints'] if c['module'] == module_name]
        sorted_ckpts = sorted(module_ckpts, key=lambda x: x['timestamp'], reverse=True)

        to_remove = sorted_ckpts[keep_n:]
        for ckpt in to_remove:
            filepath = self.checkpoint_dir / ckpt['filename']
            if filepath.exists() and ckpt['filename'] != self.metadata['best'].get(module_name):
                filepath.unlink()
                self.metadata['checkpoints'].remove(ckpt)

        self._save_metadata()


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, metrics: Dict[str, float], path: str) -> None:
    """
    Simple checkpoint saving function.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Validation metrics
        path: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, path)

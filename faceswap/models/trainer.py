"""
Training engine for face swapping model.

This module implements the FaceSwapTrainer class with:
- Training loop with progress tracking
- Checkpoint saving and loading
- GPU/CPU device management
- Loss computation and optimization
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, Any
import json
from datetime import datetime

from .autoencoder import FaceSwapAutoencoder, create_model
from .losses import FaceSwapLoss, DiscriminatorLoss
from ..utils.progress_tracker import ProgressTracker
from ..utils.logger import get_logger
from ..utils.gpu_manager import GPUManager


class FaceSwapTrainer:
    """
    Main training engine for face swapping model.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 device: Optional[str] = None,
                 checkpoint_dir: str = "checkpoints"):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration dictionary
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            checkpoint_dir: Directory to save checkpoints
        """
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        # Device management
        if device is None:
            self.device = GPUManager.detect_gpu()
        else:
            self.device = device
            
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = get_logger("FaceSwapTrainer")
        self.logger.info(f"Initializing trainer on device: {self.device}")
        
        # Model and training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Progress tracking
        self.progress_tracker = None
        
    def build_model(self) -> FaceSwapAutoencoder:
        """
        Build and initialize the face swap model.
        
        Returns:
            Initialized FaceSwapAutoencoder model
        """
        self.logger.info("Building face swap model...")
        
        # Create model
        self.model = create_model(
            input_size=(self.config.get('input_size', 256), self.config.get('input_size', 256)),
            base_filters=self.config.get('base_filters', 64)
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Log model info
        model_info = self.model.get_model_info()
        self.logger.info(f"Model created with {model_info['total_parameters']:,} parameters")
        self.logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
        
        return self.model
    
    def setup_training(self) -> None:
        """Setup optimizer, scheduler, and loss function."""
        if self.model is None:
            raise ValueError("Model must be built before setting up training")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.0001),
            betas=(0.5, 0.999),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.get('scheduler_patience', 10),
            verbose=True
        )
        
        # Loss function
        self.loss_fn = FaceSwapLoss(
            reconstruction_weight=self.config.get('reconstruction_weight', 10.0),
            perceptual_weight=self.config.get('perceptual_weight', 1.0),
            adversarial_weight=self.config.get('adversarial_weight', 0.1),
            identity_weight=self.config.get('identity_weight', 5.0),
            device=self.device
        )
        
        self.logger.info("Training setup completed")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'perceptual': 0.0,
            'identity': 0.0,
            'adversarial': 0.0
        }
        
        num_batches = len(dataloader)
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Move data to device
            source_faces = batch_data['source'].to(self.device)
            target_faces = batch_data['target'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(source_faces, target_faces)
            
            # Calculate losses
            targets = {'source': source_faces, 'target': target_faces}
            losses = self.loss_fn(outputs, targets)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Update progress
            if self.progress_tracker:
                progress = (batch_idx + 1) / num_batches
                self.progress_tracker.update(
                    self.current_epoch + progress,
                    f"Epoch {self.current_epoch + 1}, Batch {batch_idx + 1}/{num_batches}, "
                    f"Loss: {losses['total'].item():.4f}"
                )
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of average validation losses
        """
        self.model.eval()
        val_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'perceptual': 0.0,
            'identity': 0.0,
            'adversarial': 0.0
        }
        
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch_data in dataloader:
                source_faces = batch_data['source'].to(self.device)
                target_faces = batch_data['target'].to(self.device)
                
                # Forward pass
                outputs = self.model(source_faces, target_faces)
                
                # Calculate losses
                targets = {'source': source_faces, 'target': target_faces}
                losses = self.loss_fn(outputs, targets)
                
                # Accumulate losses
                for key in val_losses:
                    if key in losses:
                        val_losses[key] += losses[key].item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
            
        return val_losses
    
    def train(self, 
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              epochs: Optional[int] = None) -> None:
        """
        Main training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            epochs: Number of epochs to train (uses config if None)
        """
        if epochs is None:
            epochs = self.config.get('epochs', 100)
        
        # Setup training if not already done
        if self.optimizer is None:
            self.setup_training()
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(epochs)
        
        self.logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch(train_dataloader)
            
            # Validation
            val_losses = None
            if val_dataloader is not None:
                val_losses = self.validate(val_dataloader)
            
            # Learning rate scheduling
            if val_losses is not None:
                self.scheduler.step(val_losses['total'])
            else:
                self.scheduler.step(train_losses['total'])
            
            # Log progress
            log_msg = f"Epoch {epoch + 1}/{epochs} - "
            log_msg += f"Train Loss: {train_losses['total']:.4f}"
            if val_losses is not None:
                log_msg += f", Val Loss: {val_losses['total']:.4f}"
            
            self.logger.info(log_msg)
            
            # Save training history
            epoch_history = {
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now().isoformat()
            }
            self.training_history.append(epoch_history)
            
            # Save checkpoint
            current_loss = val_losses['total'] if val_losses else train_losses['total']
            if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(epoch + 1, current_loss)
            
            # Save best model
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint(epoch + 1, current_loss, is_best=True)
            
            # Early stopping
            if self._should_early_stop():
                self.logger.info("Early stopping triggered")
                break
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'config': self.config,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        filename = f"checkpoint_epoch_{epoch:04d}.pth"
        if is_best:
            filename = "best_model.pth"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        self.logger.info(f"Checkpoint saved: {filepath}")
        
        # Save config and history as JSON for easy inspection
        if is_best:
            config_path = os.path.join(self.checkpoint_dir, "best_config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    'config': self.config,
                    'training_history': self.training_history[-10:]  # Last 10 epochs
                }, f, indent=2)
        
        return filepath
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training progress
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        
        self.logger.info(f"Checkpoint loaded - Epoch: {self.current_epoch}, Best Loss: {self.best_loss:.4f}")
    
    def _should_early_stop(self) -> bool:
        """Check if early stopping criteria are met."""
        patience = self.config.get('early_stopping_patience', 20)
        
        if len(self.training_history) < patience:
            return False
        
        # Check if validation loss hasn't improved in 'patience' epochs
        recent_losses = [h.get('val_losses', h['train_losses'])['total'] 
                        for h in self.training_history[-patience:]]
        
        return all(loss >= self.best_loss for loss in recent_losses)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.training_history:
            return {"status": "No training history available"}
        
        latest = self.training_history[-1]
        
        return {
            "current_epoch": self.current_epoch,
            "total_epochs_trained": len(self.training_history),
            "best_loss": self.best_loss,
            "latest_train_loss": latest['train_losses']['total'],
            "latest_val_loss": latest.get('val_losses', {}).get('total', 'N/A'),
            "current_lr": latest['learning_rate'],
            "device": self.device,
            "model_parameters": self.model.get_model_info()['total_parameters'] if self.model else 'N/A'
        }
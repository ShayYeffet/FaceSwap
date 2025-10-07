"""
Loss functions for face swapping training.

This module implements various loss functions including:
- Reconstruction loss (L1/L2)
- Perceptual loss using pre-trained VGG features
- Adversarial loss for realistic face generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Tuple, Optional


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features for better identity preservation.
    """
    
    def __init__(self, layers: Optional[list] = None, device: str = 'cuda'):
        super(PerceptualLoss, self).__init__()
        
        if layers is None:
            layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        
        self.layers = layers
        self.device = device
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.vgg = vgg.to(device).eval()
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # VGG normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        # Layer mapping for VGG19
        self.layer_map = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
            'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
            'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35
        }
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for VGG (from [-1,1] to ImageNet normalization)."""
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Apply ImageNet normalization
        return (x - self.mean) / self.std
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from specified VGG layers."""
        x = self.normalize_input(x)
        features = {}
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            layer_name = None
            
            # Check if this layer corresponds to one of our target layers
            for name, idx in self.layer_map.items():
                if i == idx and name in self.layers:
                    layer_name = name
                    break
            
            if layer_name:
                features[layer_name] = x
                
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss between predicted and target images.
        
        Args:
            pred: Predicted images (B, 3, H, W)
            target: Target images (B, 3, H, W)
            
        Returns:
            Perceptual loss value
        """
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0.0
        for layer in self.layers:
            if layer in pred_features and layer in target_features:
                loss += F.mse_loss(pred_features[layer], target_features[layer])
        
        return loss / len(self.layers)


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for realistic face generation.
    """
    
    def __init__(self, loss_type: str = 'lsgan'):
        super(AdversarialLoss, self).__init__()
        self.loss_type = loss_type.lower()
        
        if self.loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported adversarial loss type: {loss_type}")
    
    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        """
        Calculate adversarial loss.
        
        Args:
            pred: Discriminator predictions
            is_real: Whether the input should be considered real
            
        Returns:
            Adversarial loss value
        """
        if self.loss_type == 'lsgan':
            target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
            return self.criterion(pred, target)
        else:  # vanilla
            target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
            return self.criterion(pred, target)


class FaceSwapLoss(nn.Module):
    """
    Combined loss function for face swapping training.
    """
    
    def __init__(self, 
                 reconstruction_weight: float = 10.0,
                 perceptual_weight: float = 1.0,
                 adversarial_weight: float = 0.1,
                 identity_weight: float = 5.0,
                 device: str = 'cuda'):
        super(FaceSwapLoss, self).__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.identity_weight = identity_weight
        
        # Loss components
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(device=device)
        self.adversarial_loss = AdversarialLoss('lsgan')
        
    def forward(self, 
                outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                discriminator_outputs: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss for face swapping.
        
        Args:
            outputs: Model outputs containing reconstructed faces
            targets: Target images (source and target faces)
            discriminator_outputs: Discriminator predictions (optional)
            
        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        
        # Reconstruction losses
        source_recon_loss = self.l1_loss(outputs['source_reconstructed'], targets['source'])
        target_recon_loss = self.l1_loss(outputs['target_reconstructed'], targets['target'])
        reconstruction_loss = (source_recon_loss + target_recon_loss) / 2
        losses['reconstruction'] = reconstruction_loss
        
        # Identity preservation loss (source should reconstruct to source, target to target)
        identity_loss = reconstruction_loss  # Same as reconstruction for identity
        losses['identity'] = identity_loss
        
        # Perceptual losses
        source_perceptual = self.perceptual_loss(outputs['source_reconstructed'], targets['source'])
        target_perceptual = self.perceptual_loss(outputs['target_reconstructed'], targets['target'])
        perceptual_loss = (source_perceptual + target_perceptual) / 2
        losses['perceptual'] = perceptual_loss
        
        # Adversarial loss (if discriminator outputs provided)
        adversarial_loss = torch.tensor(0.0, device=outputs['source_reconstructed'].device)
        if discriminator_outputs is not None:
            # Generator tries to fool discriminator
            source_adv = self.adversarial_loss(discriminator_outputs['source_fake'], True)
            target_adv = self.adversarial_loss(discriminator_outputs['target_fake'], True)
            adversarial_loss = (source_adv + target_adv) / 2
        losses['adversarial'] = adversarial_loss
        
        # Total loss
        total_loss = (self.reconstruction_weight * reconstruction_loss +
                     self.perceptual_weight * perceptual_loss +
                     self.adversarial_weight * adversarial_loss +
                     self.identity_weight * identity_loss)
        
        losses['total'] = total_loss
        
        return losses


class DiscriminatorLoss(nn.Module):
    """
    Discriminator loss for adversarial training.
    """
    
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.adversarial_loss = AdversarialLoss('lsgan')
    
    def forward(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate discriminator loss.
        
        Args:
            real_pred: Discriminator predictions for real images
            fake_pred: Discriminator predictions for fake images
            
        Returns:
            Discriminator loss value
        """
        real_loss = self.adversarial_loss(real_pred, True)
        fake_loss = self.adversarial_loss(fake_pred, False)
        
        return (real_loss + fake_loss) / 2
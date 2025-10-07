"""
Autoencoder model architecture for face swapping.

This module implements the core deep learning model with:
- Shared encoder for both source and target faces
- Separate decoders for source and target reconstruction
- Residual connections and skip connections for better training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class ResidualBlock(nn.Module):
    """Residual block with skip connections for better gradient flow."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_connection(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out


class Encoder(nn.Module):
    """Shared encoder network with convolutional layers and residual connections."""
    
    def __init__(self, input_channels: int = 3, base_filters: int = 64):
        super(Encoder, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks with increasing channels
        self.layer1 = self._make_layer(base_filters, base_filters, 2, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_filters * 2, base_filters * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_filters * 4, base_filters * 8, 2, stride=2)
        
        # Global average pooling and final encoding
        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.final_conv = nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1)
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through encoder with skip connections for decoder.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 256, 256)
            
        Returns:
            Tuple of (encoded_features, skip_connections)
        """
        skip_connections = {}
        
        # Initial processing
        x = self.initial_conv(x)  # (B, 64, 64, 64)
        skip_connections['skip1'] = x
        
        # Residual layers
        x = self.layer1(x)  # (B, 64, 64, 64)
        skip_connections['skip2'] = x
        
        x = self.layer2(x)  # (B, 128, 32, 32)
        skip_connections['skip3'] = x
        
        x = self.layer3(x)  # (B, 256, 16, 16)
        skip_connections['skip4'] = x
        
        x = self.layer4(x)  # (B, 512, 8, 8)
        
        # Final encoding
        x = self.global_pool(x)  # (B, 512, 8, 8)
        encoded = self.final_conv(x)  # (B, 512, 8, 8)
        
        return encoded, skip_connections


class Decoder(nn.Module):
    """Decoder network with skip connections for face reconstruction."""
    
    def __init__(self, base_filters: int = 64, output_channels: int = 3):
        super(Decoder, self).__init__()
        
        # Upsampling layers with skip connections
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 8, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, output_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output values in [-1, 1]
        )
        
    def forward(self, encoded: torch.Tensor, skip_connections: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through decoder with skip connections.
        
        Args:
            encoded: Encoded features from encoder (B, 512, 8, 8)
            skip_connections: Dictionary of skip connection tensors
            
        Returns:
            Reconstructed face tensor (B, 3, 256, 256)
        """
        x = encoded
        
        # Upsample with skip connections
        x = self.upsample1(x)  # (B, 256, 16, 16)
        x = torch.cat([x, skip_connections['skip4']], dim=1)  # (B, 512, 16, 16)
        
        x = self.upsample2(x)  # (B, 128, 32, 32)
        x = torch.cat([x, skip_connections['skip3']], dim=1)  # (B, 256, 32, 32)
        
        x = self.upsample3(x)  # (B, 64, 64, 64)
        x = torch.cat([x, skip_connections['skip2']], dim=1)  # (B, 128, 64, 64)
        
        x = self.upsample4(x)  # (B, 64, 128, 128)
        x = torch.cat([x, skip_connections['skip1']], dim=1)  # (B, 128, 128, 128)
        
        # Final upsampling to original resolution
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Final convolution
        output = self.final_conv(x)  # (B, 3, 256, 256)
        
        return output


class FaceSwapAutoencoder(nn.Module):
    """
    Complete autoencoder model for face swapping.
    
    Architecture:
    - Shared encoder for both source and target faces
    - Separate decoders for source and target reconstruction
    - Skip connections for better detail preservation
    """
    
    def __init__(self, input_channels: int = 3, base_filters: int = 64):
        super(FaceSwapAutoencoder, self).__init__()
        
        # Shared encoder
        self.encoder = Encoder(input_channels, base_filters)
        
        # Separate decoders for source and target
        self.decoder_source = Decoder(base_filters, input_channels)
        self.decoder_target = Decoder(base_filters, input_channels)
        
    def forward(self, source_faces: torch.Tensor, target_faces: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            source_faces: Source face images (B, 3, 256, 256)
            target_faces: Target face images (B, 3, 256, 256)
            
        Returns:
            Dictionary containing reconstructed faces and encoded features
        """
        # Encode both source and target faces
        source_encoded, source_skips = self.encoder(source_faces)
        target_encoded, target_skips = self.encoder(target_faces)
        
        # Reconstruct faces using their respective decoders
        source_reconstructed = self.decoder_source(source_encoded, source_skips)
        target_reconstructed = self.decoder_target(target_encoded, target_skips)
        
        # Cross-reconstruction for face swapping
        source_to_target = self.decoder_target(source_encoded, source_skips)
        target_to_source = self.decoder_source(target_encoded, target_skips)
        
        return {
            'source_reconstructed': source_reconstructed,
            'target_reconstructed': target_reconstructed,
            'source_to_target': source_to_target,
            'target_to_source': target_to_source,
            'source_encoded': source_encoded,
            'target_encoded': target_encoded
        }
    
    def swap_face(self, source_face: torch.Tensor, decoder_type: str = 'target') -> torch.Tensor:
        """
        Perform face swapping inference.
        
        Args:
            source_face: Source face image (1, 3, 256, 256)
            decoder_type: Which decoder to use ('source' or 'target')
            
        Returns:
            Swapped face image (1, 3, 256, 256)
        """
        with torch.no_grad():
            encoded, skip_connections = self.encoder(source_face)
            
            if decoder_type == 'target':
                swapped = self.decoder_target(encoded, skip_connections)
            else:
                swapped = self.decoder_source(encoded, skip_connections)
                
            return swapped
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': 'Encoder-Decoder with Skip Connections'
        }


def create_model(input_size: Tuple[int, int] = (256, 256), 
                base_filters: int = 64) -> FaceSwapAutoencoder:
    """
    Factory function to create the face swap model.
    
    Args:
        input_size: Input image size (height, width)
        base_filters: Base number of filters in the network
        
    Returns:
        Initialized FaceSwapAutoencoder model
    """
    model = FaceSwapAutoencoder(input_channels=3, base_filters=base_filters)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    return model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import yacs.config


def initialize_weights(module: torch.nn.Module) -> None:
    """Custom weight initialization for layers."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class Model(nn.Module):
    def __init__(self, config: yacs.config.CfgNode):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=0)

        # Load pretrained ResNet-18 backbone
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the last fully connected layer to match desired output
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 500)

        # Additional layers (if needed)
        self.extra_fc = nn.Linear(500 + 2, 2)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for custom layers."""
        self.resnet.fc.apply(initialize_weights)
        self.extra_fc.apply(initialize_weights)

    def forward(self, x: torch.Tensor, extra_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input image tensor of shape (batch_size, 3, H, W)
            extra_features: Additional features tensor of shape (batch_size, extra_features)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.resnet(x)  # Forward pass through ResNet-18 backbone
        x = F.relu(x)
        x = torch.cat([x, extra_features], dim=1)  # Concatenate with additional features
        x = self.extra_fc(x)  # Pass through the additional fully connected layer
        return x

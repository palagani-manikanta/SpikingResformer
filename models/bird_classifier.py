# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

class BirdClassifier(nn.Module):
    """
    Bird Classifier mapping predicted concept probabilities (shape: [batch, 112])
    to final bird class logits (shape: [batch, 200]).
    """
    def __init__(self, num_concepts=112, num_classes=200):
        super(BirdClassifier, self).__init__()
        
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        
        # A single linear layer mapping concepts -> classes
        # This keeps the model interpretable by inspecting weight coefficients
        self.linear = nn.Linear(num_concepts, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Concept probabilities of shape (batch_size, num_concepts).
        Returns:
            torch.Tensor: Bird class logits of shape (batch_size, num_classes).
        """
        return self.linear(x)

import torch
import torch.nn as nn

class ConceptPredictor(nn.Module):
    """
    Concept Predictor layer for SpikingResformer representations.
    Maps a high-dimensional feature representation (e.g., Pre-reset Vmem activations of shape (batch, 6144))
    to logits for a set of concepts (e.g., 112 concepts for CUB).
    """
    def __init__(self, input_dim=6144, num_concepts=112, hidden_dim=None):
        super(ConceptPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim
        
        if hidden_dim is not None:
            # Multi-Layer Perceptron (MLP) architecture
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_concepts)
            )
        else:
            # Linear architecture (standard for logistic regression probing / Ante-Hoc CBMs)
            self.net = nn.Linear(input_dim, num_concepts)
            
    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Feature representations of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Unnormalized logits of shape (batch_size, num_concepts).
                          Apply torch.sigmoid to get concept probabilities.
        """
        return self.net(x)

    def get_probabilities(self, x):
        """
        Forward pass returning sigmoid probabilities.
        Args:
            x (torch.Tensor): Feature representations of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Concept probabilities of shape (batch_size, num_concepts).
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

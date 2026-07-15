# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

class CalibratedConceptPredictor(nn.Module):
    """
    Wrapper that intercept-scales concept logits in-place.
    calibrated_logits = A * raw_logits + B
    """
    def __init__(self, original_predictor, A, B):
        super(CalibratedConceptPredictor, self).__init__()
        self.predictor = original_predictor
        # Register A and B as PyTorch parameters so they are handled on CUDA/device correctly
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(B, dtype=torch.float32))
        
    def forward(self, x):
        """
        Intercepts logits and applies Platt Scaling.
        """
        logits = self.predictor(x)
        return self.A * logits + self.B

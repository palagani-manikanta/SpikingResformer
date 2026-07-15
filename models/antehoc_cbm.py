# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
import types

# pyrefly: ignore [missing-import]
from spikingjelly.activation_based import functional, neuron

from models.spikingresformer import spikingresformer_ti
from models.concept_predictor import ConceptPredictor
from models.bird_classifier import BirdClassifier

class AnteHocCBM(nn.Module):
    """
    Ante-Hoc Concept Bottleneck Model.
    Integrates SpikingResformer (backbone), ConceptPredictor (bottleneck),
    and BirdClassifier (downstream classifier) into a single end-to-end model.
    """
    def __init__(self, backbone_checkpoint=None, concept_predictor_checkpoint=None, bird_classifier_checkpoint=None, device='cpu'):
        super(AnteHocCBM, self).__init__()
        
        self.device = device
        
        # 1. Initialize and configure Backbone (SpikingResformer-Ti)
        self.backbone = spikingresformer_ti()
        if backbone_checkpoint:
            self._load_backbone_weights(backbone_checkpoint)
            
        # Configure SpikingJelly nodes to use torch backend and threshold 0.01
        for name, m in self.backbone.named_modules():
            if isinstance(m, neuron.BaseNode):
                m.backend = 'torch'
                m.v_threshold = 1.0
        self.backbone.eval()
        
        # Penultimate spiking module for Pre-reset Vmem extraction
        self.target_module = dict(self.backbone.named_modules()).get("layers.2.6.down.0")
        if self.target_module is None:
            raise ValueError("Target SNN module layers.2.6.down.0 not found in Backbone!")
            
        # Register custom forward method on the target module to extract Pre-reset potentials
        self.target_module.store_v_seq = False
        self.target_module.backend = 'torch'
        self.target_module.multi_step_forward = types.MethodType(self._custom_multi_step_forward, self.target_module)
        
        # 2. Concept Predictor Module
        self.concept_predictor = ConceptPredictor(input_dim=6144, num_concepts=112)
        if concept_predictor_checkpoint:
            self.concept_predictor.load_state_dict(torch.load(concept_predictor_checkpoint, map_location=device))
        self.concept_predictor.eval()
        
        # 3. Downstream Bird Classifier Module
        self.bird_classifier = BirdClassifier(num_concepts=112, num_classes=200)
        if bird_classifier_checkpoint:
            self.bird_classifier.load_state_dict(torch.load(bird_classifier_checkpoint, map_location=device))
        self.bird_classifier.eval()
        
        # StandardScaler variables (fitted parameters saved in concept_scaler.pkl)
        self.scaler_mean = None
        self.scaler_var = None
        
    def set_scaler(self, mean, var):
        """
        Loads standardisation parameters for features scaling inside forward pass.
        Args:
            mean (np.ndarray or torch.Tensor): Mean vector of shape (6144,).
            var (np.ndarray or torch.Tensor): Variance vector of shape (6144,).
        """
        self.scaler_mean = torch.tensor(mean, dtype=torch.float32).to(self.device)
        self.scaler_var = torch.tensor(var, dtype=torch.float32).to(self.device)
        
    def _load_backbone_weights(self, path):
        print(f"Loading backbone weights from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = None
        if isinstance(checkpoint, torch.nn.Module):
            state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict):
            if "model" in checkpoint:
                val = checkpoint["model"]
                state_dict = val.state_dict() if isinstance(val, torch.nn.Module) else val
            elif "state_dict" in checkpoint:
                val = checkpoint["state_dict"]
                state_dict = val.state_dict() if isinstance(val, torch.nn.Module) else val
            else:
                state_dict = checkpoint
        self.backbone.load_state_dict(state_dict, strict=True)
        print("Backbone checkpoint loaded successfully.")

    @staticmethod
    def _custom_multi_step_forward(self_node, x_seq: torch.Tensor):
        """
        Override for target SNN module to capture membrane potential right BEFORE reset.
        """
        self_node.v_float_to_tensor(x_seq[0])
        T = x_seq.shape[0]
        spike_seq = []
        v_pre_seq = []
        for t in range(T):
            self_node.neuronal_charge(x_seq[t])
            v_pre_seq.append(self_node.v.clone())
            spike = self_node.neuronal_fire()
            self_node.neuronal_reset(spike)
            spike_seq.append(spike)
        self_node.v_seq = torch.stack(v_pre_seq)
        return torch.stack(spike_seq)

    def extract_pre_reset_features(self, x):
        """
        Extracts Pre-reset Vmem representation with spatial global average pooling (GAP) and temporal flattening.
        Args:
            x (torch.Tensor): Raw image input tensor of shape (batch, 3, 224, 224).
        Returns:
            torch.Tensor: Pre-reset Vmem features of shape (batch, 6144).
        """
        functional.reset_net(self.backbone)
        _ = self.backbone(x)
        # target_module.v_seq shape: [T, B, C, H, W]
        pre_reset = self.target_module.v_seq
        # Spatial GAP: mean over H and W
        pre_reset_gap = pre_reset.mean(dim=(-2, -1)) # Shape: [T, B, C]
        # Permute and reshape to [B, T * C]
        B = x.size(0)
        pre_reset_flat = pre_reset_gap.permute(1, 0, 2).reshape(B, -1)
        return pre_reset_flat

    def forward(self, x):
        """
        End-to-end inference pass.
        Args:
            x (torch.Tensor): Raw image input tensor of shape (batch, 3, 224, 224).
        Returns:
            dict: Predictions dictionary containing:
                  - 'pre_reset_features': Extracted features of shape (batch, 6144)
                  - 'concept_logits': Raw concept logits of shape (batch, 112)
                  - 'concept_probabilities': Sigmoid concept probabilities of shape (batch, 112)
                  - 'class_logits': Downstream classification logits of shape (batch, 200)
        """
        # 1. Feature Extraction
        features = self.extract_pre_reset_features(x)
        
        # 2. Scaling
        features_scaled = features
        if self.scaler_mean is not None and self.scaler_var is not None:
            std = torch.sqrt(self.scaler_var + 1e-8)
            features_scaled = (features - self.scaler_mean) / std
            
        # 3. Concept Predictor
        concept_logits = self.concept_predictor(features_scaled)
        concept_probs = torch.sigmoid(concept_logits)
        
        # 4. Bird Downstream Classifier
        class_logits = self.bird_classifier(concept_probs)
        
        return {
            "pre_reset_features": features,
            "concept_logits": concept_logits,
            "concept_probabilities": concept_probs,
            "class_logits": class_logits
        }

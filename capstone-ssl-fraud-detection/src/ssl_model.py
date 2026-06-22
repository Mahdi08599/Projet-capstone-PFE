"""
=================================================================
Modèle Self-Supervised Learning — Masked Autoencoder Tabulaire
=================================================================
Architecture simple et explicable :

    Transaction (224 features)
         │
    ┌────▼────┐
    │ Encoder │  couches: 224 → 128 → 64 (représentation latente)
    └────┬────┘
         │
    ┌────▼─────────────┐
    │ Reconstruction   │  64 → 128 → 224 (reconstruire les features masquées)
    │ Head (SSL)       │
    └──────────────────┘
         │
    ┌────▼─────────────┐
    │ Classification   │  64 → 32 → 1 (prédire fraude / non fraude)
    │ Head (fine-tune) │
    └──────────────────┘

Principe pour le jury :
    1. On masque 15% des features d'une transaction
    2. Le modèle apprend à reconstruire les valeurs cachées
    3. Il apprend ainsi la structure normale des transactions
    4. On réutilise cet encodeur pour détecter les fraudes
=================================================================
"""

import torch
import torch.nn as nn


class TransactionEncoder(nn.Module):
    """
    Encodeur MLP : transforme une transaction en représentation compacte.
    
    224 features → 128 neurones → 64 neurones (représentation latente)
    
    C'est le cœur du modèle. Après le pretraining SSL,
    cet encodeur "comprend" la structure des transactions.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.network(x)


class ReconstructionHead(nn.Module):
    """
    Tête de reconstruction : décode la représentation latente
    pour reconstruire les features originales.
    
    64 → 128 → 224 (même taille que l'entrée)
    
    Utilisée UNIQUEMENT pendant le pretraining SSL.
    """
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, output_dim: int = 224):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, z):
        return self.network(z)


class FraudClassificationHead(nn.Module):
    """
    Tête de classification : prédit fraude (1) ou non fraude (0).
    
    64 → 32 → 1 (probabilité de fraude)
    
    Utilisée UNIQUEMENT pendant le fine-tuning.
    """
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, z):
        return self.network(z).squeeze(-1)


class SSLFraudModel(nn.Module):
    """
    Modèle complet avec les deux modes :
    
    Mode "ssl" (pretraining) :
        input → Encoder → ReconstructionHead → features reconstruites
    
    Mode "supervised" (fine-tuning) :
        input → Encoder (gelé) → FraudClassificationHead → probabilité fraude
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        
        self.encoder = TransactionEncoder(input_dim, hidden_dim, latent_dim)
        self.reconstruction_head = ReconstructionHead(latent_dim, hidden_dim, input_dim)
        self.classification_head = FraudClassificationHead(latent_dim)
    
    def forward(self, x, mode: str = "ssl"):
        z = self.encoder(x)
        
        if mode == "ssl":
            return self.reconstruction_head(z)
        elif mode == "supervised":
            return self.classification_head(z)
        else:
            raise ValueError(f"Mode inconnu : {mode}")
    
    def freeze_encoder(self):
        """Gèle l'encodeur pour le fine-tuning (on ne modifie plus ses poids)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder gelé (poids fixés)")
    
    def unfreeze_encoder(self):
        """Dégèle l'encodeur si besoin."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder dégelé")


def build_model(input_dim: int) -> SSLFraudModel:
    """Construit le modèle et affiche un résumé."""
    model = SSLFraudModel(input_dim=input_dim)
    
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    recon_params = sum(p.numel() for p in model.reconstruction_head.parameters())
    classif_params = sum(p.numel() for p in model.classification_head.parameters())
    
    print(f"Modèle créé :")
    print(f"  Input dim     : {input_dim}")
    print(f"  Encoder       : {encoder_params:,} params")
    print(f"  Reconstruction: {recon_params:,} params")
    print(f"  Classification: {classif_params:,} params")
    print(f"  Total         : {total_params:,} params")
    
    return model

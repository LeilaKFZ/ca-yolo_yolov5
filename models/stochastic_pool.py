import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    - F.unfol/F.Fold :  servent à découper/reconstruire les fenêtres sur toute la carte.
    - On calcule localement 'p_i' pour chaque petite fenêtre de taille KxK
    - On effectue un tirage catégoriel avec "torch.distributions.Categorical"
"""

class StochasticPool2d(nn.Module):
    """
    Pooling stochastique 2D – pour chaque fenêtre k×k, on choisit un élément
    Random selon sa probabilité pi = fi / sum(fj).
    """

    def __init__(self, kernel_size, stride=1, padding=0):
        """
        Args:
            kernel_size (int): taille de la fenêtre (k)
            stride (int): pas de glissement de la fenêtre (pixel par pixel)
            padding (int): padding appliqué avant le pooling
        """
        super().__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        x: Tensor de forme [B, C, H, W]
        Retourne:
            output: Tensor de forme [B, C, H, W] après pooling stochastique.
        """

        B, C, H, W = x.shape

        # 1) Extraire tous les patchs k×k :
        #      (B, C * k*k, L) où L = nombre de positions de la fenêtre.
        patches = F.unfold(x, kernel_size=self.k,
                           stride=self.stride,
                           padding=self.padding)
        # Reformatage en (B, C, k*k, L)
        patches = patches.view(B, C, self.k * self.k, -1)  # [B, C, k*k, L]

        # 2) Calculer la somme sur chaque patch (pour normaliser)
        #       [B, C, 1, L]
        sums = patches.sum(dim=2, keepdim=True)

        # 3) Calculer les probabilités pi = fi / sum(fj)
        #       [B, C, k*k, L]
        #    Éviter division par zéro si sum == 0 en ajoutant un epsilon
        eps = 1e-6
        probs = patches / (sums + eps)

        # 4) Tirage aléatoire pour chaque canal et chaque position
        #    On convertit (B, C, k*k, L) en (B*C, k*k, L) pour itérer plus facilement
        bc, kk, L = B * C, self.k * self.k, patches.size(-1)
        p = probs.view(bc, kk, L)  # [B*C, k*k, L]
        f = patches.view(bc, kk, L)  # mêmes valeurs pour indexation

        # Pour chaque colonne (patch), tirer un index selon p[:, :, i]
        # On crée un mask de sortie [B*C, L]
        idx = torch.zeros(bc, L, dtype=torch.long, device=x.device)
        for i in range(L):
            # tirage catégoriel parmi k*k options
            dist = torch.distributions.Categorical(p[:, :, i])
            idx[:, i] = dist.sample()

        # 5) Récupérer la valeur f_r pour chaque patch
        #    on indexe f[b, idx[b,i], i]
        #    on crée un tensor [B*C, L]
        out_vals = f.gather(1, idx.unsqueeze(1)).squeeze(1)  # [B*C, L]

        # 6) Remonter la forme pour pouvoir "fold" : (B, C*k*k, L)
        # CORRECTION: On crée un tenseur avec la valeur sélectionnée à la position idx
        # et des zéros ailleurs pour éviter l'accumulation de valeurs
        out_patches = torch.zeros_like(f)
        for i in range(L):
            for b in range(bc):
                out_patches[b, idx[b, i], i] = out_vals[b, i]

        # Reformater pour fold
        out_patches = out_patches.view(B, C * kk, L)

        # 7) Reconstruire la carte 2D avec F.fold
        output = F.fold(
            out_patches,
            output_size=(H, W),
            kernel_size=self.k,
            stride=self.stride,
            padding=self.padding
        )

        # Puisque les pixels se chevauches 'redandances'
        # 8) CORRECTION: Normaliser par le nombre d'occurrences de chaque position
        # Calculer une carte des compteurs pour chaque position
        ones = torch.ones_like(out_patches)
        counter = F.fold(
            ones,
            output_size=(H, W),
            kernel_size=self.k,
            stride=self.stride,
            padding=self.padding
        )

        # Éviter division par zéro
        counter = torch.clamp(counter, min=1.0)

        # Normaliser l'output par le compteur
        output = output / counter

        return output
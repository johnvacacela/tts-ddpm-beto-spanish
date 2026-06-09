"""
Módulo de Difusión DDPM para predicción de pitch en FastPitch.
Reemplaza el TemporalPredictor lineal por un proceso de difusión estocástico.

Basado en:
- DiffProsody (Oh et al., 2024, IEEE/ACM TASLP)
- DDPM (Ho et al., 2020, NeurIPS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalPosEmb(nn.Module):
    """Embedding posicional sinusoidal para el paso de difusión t."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = np.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiffusionStep(nn.Module):
    """
    Un bloque de la red de denoising.
    Toma: x_noisy [B, T, 1], condicion [B, T, 384], t_emb [B, dim_t]
    Devuelve: noise_pred [B, T, 1]
    """
    def __init__(self, hidden_dim, t_dim):
        super().__init__()
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.t_proj    = nn.Linear(t_dim, hidden_dim)
        self.conv1     = nn.Conv1d(hidden_dim + hidden_dim, hidden_dim, 3, padding=1)
        self.conv2     = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.norm1     = nn.LayerNorm(hidden_dim)
        self.norm2     = nn.LayerNorm(hidden_dim)
        self.act       = nn.GELU()

    def forward(self, x, cond, t_emb):
        # x: [B, T, 1], cond: [B, T, H], t_emb: [B, t_dim]
        cond_out = self.act(self.cond_proj(cond))             # [B, T, H]
        t_out    = self.act(self.t_proj(t_emb)).unsqueeze(1)  # [B, 1, H]
        cond_out = cond_out + t_out                           # broadcast

        # Concatenar x con condición
        x_in = torch.cat([x, cond_out], dim=-1)              # [B, T, 1+H]
        x_in = x_in.transpose(1, 2)                           # [B, 1+H, T]

        out = self.act(self.norm1(self.conv1(x_in).transpose(1, 2)))
        out = self.norm2(self.conv2(out.transpose(1, 2)).transpose(1, 2))
        return out  # [B, T, H]


class PitchDiffusion(nn.Module):
    """
    Predictor de pitch basado en DDPM.
    Opera sobre H_enc', la salida del encoder ya enriquecida con BETO.
    Interfaz compatible con TemporalPredictor:
        forward(enc_out, enc_mask) -> pitch_pred [B, T, 1]
    """
    def __init__(self, input_size, n_steps=100, hidden_dim=256, t_dim=128,
                 n_layers=4, n_predictions=1):
        super().__init__()
        self.n_steps      = n_steps
        self.hidden_dim   = hidden_dim
        self.n_predictions = n_predictions

        # Proyección de H_enc' (encoder con condicionamiento BETO ya sumado).
        self.bert_dropout = nn.Dropout(0.1)
        self.cond_proj = nn.Linear(input_size, hidden_dim)
        # Proyección de x (pitch) al espacio de difusión
        self.x_proj = nn.Linear(n_predictions, hidden_dim)

        # Embedding del paso de difusión t
        self.t_emb = SinusoidalPosEmb(t_dim)

        # Red de denoising (capas)
        self.layers = nn.ModuleList([
            DiffusionStep(hidden_dim, t_dim) for _ in range(n_layers)
        ])

        # Proyección final a pitch
        self.out_proj = nn.Linear(hidden_dim, n_predictions)

        # Schedule de ruido (beta lineal)
        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod',
                             torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: añade ruido a x0 en el paso t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def p_losses(self, x0, cond, mask, bert_emb=None):
        """
        Calcula la loss de difusión para entrenamiento.
        x0: pitch real [B, T, 1]
        cond: H_enc' semánticamente enriquecida [B, T, H]
        mask: [B, T, 1]
        bert_emb: argumento conservado por compatibilidad; no se concatena
                  por frame porque BETO ya fue sumado antes a H_enc'.
        """
        # BETO ya se aplica antes del DDPM; no se concatena por frame.
        B = x0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=x0.device)
        x_noisy, noise = self.q_sample(x0, t)
        noise_pred = self._denoise(x_noisy, cond, t)
        loss = F.mse_loss(noise_pred * mask, noise * mask)
        return loss

    def _denoise(self, x, cond, t):
        """Aplica la red de denoising."""
        cond_h = self.cond_proj(cond)      # [B, T, H]
        t_emb  = self.t_emb(t)             # [B, t_dim]
        h = self.x_proj(x)                # [B, T, H]
        for layer in self.layers:
            h = layer(h, cond_h, t_emb)   # [B, T, H]
        return self.out_proj(h)            # [B, T, 1]

    @torch.no_grad()
    def forward(self, enc_out, enc_out_mask, bert_emb=None):
        """
        Inferencia: proceso reverse de difusión.
        enc_out corresponde a H_enc', ya condicionada semánticamente.
        bert_emb se conserva solo por compatibilidad de interfaz.
        Compatible con TemporalPredictor.forward()
        """
        B, T, _ = enc_out.shape
        cond = enc_out * enc_out_mask

        # Empezar desde ruido puro
        x = torch.randn(B, T, self.n_predictions, device=enc_out.device)

        # Proceso reverse: T → 0
        for t_val in reversed(range(self.n_steps)):
            t = torch.full((B,), t_val, device=enc_out.device, dtype=torch.long)
            noise_pred = self._denoise(x, cond, t)

            beta      = self.betas[t_val]
            alpha     = self.alphas[t_val]
            alpha_bar = self.alphas_cumprod[t_val]

            # Ecuación de reverse DDPM
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred
            )
            if t_val > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)

        return x * enc_out_mask

"""
BERT Semantic Conditioner para FastPitch — Español
Modelo: dccuchile/bert-base-spanish-wwm-cased (BETO)

Flujo:
    Párrafo completo (str)
        → BETO (bidireccional, ve todo el párrafo)
        → Vector CLS (768d) — resumen semántico global
        → Proyección Linear(768 → 384) + LayerNorm + Tanh
        → Vector semántico (384d)
        → Suma aditiva al encoder output de FastPitch
        → Predictor de pitch DDPM condicionado semánticamente
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class BERTSemanticConditioner(nn.Module):
    """
    Extrae el vector CLS de BETO para un párrafo completo
    y lo proyecta a la dimensión del encoder de FastPitch (384d).

    Args:
        bert_hidden_size  : dimensión de salida de BETO (768)
        encoder_dim       : dimensión del encoder de FastPitch (384)
        freeze_bert       : si True, BETO no se entrena (solo la proyección)
    """

    def __init__(self,
                 bert_hidden_size: int = 768,
                 encoder_dim: int = 384,
                 freeze_bert: bool = True):
        super().__init__()

        # Cargar BETO
        self.tokenizer = AutoTokenizer.from_pretrained(
            'dccuchile/bert-base-spanish-wwm-cased')
        self.bert = AutoModel.from_pretrained(
            'dccuchile/bert-base-spanish-wwm-cased')

        # Congelar BETO — solo entrenamos la proyección
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Proyección entrenable: BERT (768d) → encoder dim (384d)
        # LayerNorm normaliza la escala, Tanh acota el rango [-1,1]
        # Esto balancea la contribución BERT vs encoder en el cond_proj del DDPM
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden_size, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.Tanh()
        )

    def forward(self, parrafo: str, device: str = 'cuda') -> torch.Tensor:
        """
        Args:
            parrafo : texto completo del párrafo en español
            device  : dispositivo de cómputo

        Returns:
            tensor de shape (1, 384) listo para inyección aditiva
        """
        # Tokenizar párrafo completo (hasta 512 tokens)
        tokens = self.tokenizer(
            parrafo,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)

        # Extraer vector CLS — resumen semántico del párrafo completo
        with torch.no_grad():
            out = self.bert(**tokens)

        cls_vector = out.last_hidden_state[:, 0, :]  # (1, 768)

        # Proyectar al espacio del encoder de FastPitch
        return self.projection(cls_vector)           # (1, 384)

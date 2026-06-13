"""Language encoder for task conditioning."""

import torch
import torch.nn as nn
from typing import Optional


class LanguageEncoder(nn.Module):
    """Encode language instructions into embeddings for cross-attention.

    Supports:
    - Flan-T5 (default)
    - Simple embedding fallback for testing
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-xl",
        hidden_dim: int = 1024,
        max_length: int = 128,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._freeze = freeze_encoder

        # Projection from T5 dim to model hidden_dim
        self.projection = None  # initialized lazily or in _init_model

    def _init_model(self):
        """Lazy-load T5 model."""
        try:
            from transformers import T5EncoderModel, T5Tokenizer

            self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self._model = T5EncoderModel.from_pretrained(self.model_name)

            t5_dim = self._model.config.d_model
            self.projection = nn.Linear(t5_dim, self.hidden_dim)

            if self._freeze:
                self._model.eval()
                for p in self._model.parameters():
                    p.requires_grad = False
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text: Optional[list] = None,
    ) -> torch.Tensor:
        """Encode text to embeddings.

        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) mask
            text: list of strings (alternative to input_ids)

        Returns:
            embeddings: (B, L, hidden_dim)
        """
        if self._model is None:
            self._init_model()

        if text is not None:
            encoded = self._tokenizer(
                text, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            input_ids = encoded.input_ids.to(self._model.device)
            attention_mask = encoded.attention_mask.to(self._model.device)

        with torch.no_grad() if self._freeze else torch.enable_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # (B, L, t5_dim)

        return self.projection(hidden)  # (B, L, hidden_dim)


class SimpleLanguageEncoder(nn.Module):
    """Lightweight language encoder for testing (no pretrained model needed)."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 1024, max_length: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Embedding(max_length, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.max_length = max_length

    def forward(self, input_ids: torch.Tensor = None, text: list = None, **kwargs) -> torch.Tensor:
        """Encode token ids to embeddings.

        Args:
            input_ids: (B, L) token ids
            text: list of strings (alternative, auto-encoded)
        """
        if input_ids is None and text is not None:
            input_ids = encode_text_simple(text, max_length=self.max_length)
            input_ids = input_ids.to(self.embedding.weight.device)
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_encoding(positions)
        return self.layer_norm(x)


def encode_text_simple(texts: list, vocab: dict = None, max_length: int = 32) -> torch.Tensor:
    """Simple character-level encoding for testing without T5."""
    if vocab is None:
        # Build simple char vocab
        chars = set()
        for t in texts:
            chars.update(t.lower())
        vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}
        vocab["<pad>"] = 0

    batch_ids = []
    for text in texts:
        ids = [vocab.get(c, 0) for c in text.lower()[:max_length]]
        ids += [0] * (max_length - len(ids))
        batch_ids.append(ids)

    return torch.tensor(batch_ids, dtype=torch.long)

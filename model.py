"""
Diacritic Restorer Model - Encoder-Decoder architecture for Yoruba diacritic restoration.
"""

import torch
import torch.nn as nn


class DiacriticRestorer(nn.Module):
    """
    Encoder-Decoder model for Yoruba diacritic restoration.

    Architecture:
    - Embedding layer for character-level input
    - Bidirectional LSTM encoder to capture context
    - LSTM decoder to predict characters with diacritics
    - Linear output layer to vocabulary
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Bidirectional encoder
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Decoder (takes concatenated bidirectional output)
        self.decoder = nn.LSTM(
            hidden_dim * 2,  # bidirectional encoder output
            hidden_dim * 2,  # 512 hidden units
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.output = nn.Linear(hidden_dim * 2, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, target_ids=None):
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] - input character indices
            target_ids: [batch, seq_len] - target character indices (for teacher forcing)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Embed input
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        embedded = self.dropout(embedded)

        # Encode
        encoder_out, (hidden, cell) = self.encoder(embedded)
        # encoder_out: [batch, seq_len, hidden*2]

        # Prepare decoder initial states
        # Reshape hidden states from bidirectional encoder
        # hidden: [num_layers*2, batch, hidden] -> [num_layers, batch, hidden*2]
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)

        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

        # Decode (using encoder output directly - character-level alignment)
        decoder_out, _ = self.decoder(encoder_out, (hidden, cell))
        # decoder_out: [batch, seq_len, hidden*2]

        decoder_out = self.dropout(decoder_out)

        # Project to vocabulary
        logits = self.output(decoder_out)  # [batch, seq_len, vocab_size]

        return logits

    def predict(self, input_ids):
        """Predict output characters (no teacher forcing)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            predictions = torch.argmax(logits, dim=-1)
        return predictions

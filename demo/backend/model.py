"""
DiacriticRestorer model for Yoruba text.
"""
import torch
import torch.nn as nn


class DiacriticRestorer(nn.Module):
    """
    Bidirectional LSTM model for restoring diacritics in Yoruba text.
    """
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2):
        """
        Initialize the model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
        """
        super(DiacriticRestorer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional encoder LSTM
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Decoder LSTM (unidirectional)
        self.decoder = nn.LSTM(
            embedding_dim + hidden_dim * 2,  # + encoder hidden states
            hidden_dim,
            num_layers,
            batch_first=True
        )
        
        # Output projection
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        # Embedding
        embedded = self.embedding(x)
        
        # Encoder
        encoder_outputs, (hidden, cell) = self.encoder(embedded)
        
        # Prepare decoder hidden state from encoder
        # Concatenate forward and backward hidden states
        batch_size = x.size(0)
        num_layers = self.decoder.num_layers
        
        # Reshape hidden states: (num_layers * num_directions, batch, hidden) -> (num_layers, batch, hidden * num_directions)
        hidden_forward = hidden[:num_layers]  # Forward direction
        hidden_backward = hidden[num_layers:]  # Backward direction
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=2)
        
        cell_forward = cell[:num_layers]
        cell_backward = cell[num_layers:]
        cell_concat = torch.cat([cell_forward, cell_backward], dim=2)
        
        # Project to decoder hidden size (if needed)
        if hidden_concat.size(2) != self.decoder.hidden_size:
            # If encoder hidden is 2x decoder hidden, we need to project
            # For now, just take the first half (this might need adjustment)
            hidden_concat = hidden_concat[:, :, :self.decoder.hidden_size]
            cell_concat = cell_concat[:, :, :self.decoder.hidden_size]
        
        # Decoder with attention-like mechanism
        # Use encoder outputs as context
        decoder_input = embedded
        decoder_outputs = []
        
        for t in range(embedded.size(1)):
            # Concatenate current decoder input with encoder context
            context = encoder_outputs[:, t, :].unsqueeze(1)
            decoder_input_t = decoder_input[:, t:t+1, :]
            decoder_input_concat = torch.cat([decoder_input_t, context], dim=2)
            
            # Decoder step
            decoder_output, (hidden_concat, cell_concat) = self.decoder(
                decoder_input_concat,
                (hidden_concat, cell_concat)
            )
            decoder_outputs.append(decoder_output)
        
        # Concatenate all decoder outputs
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        
        # Output projection
        logits = self.output(decoder_outputs)
        
        return logits


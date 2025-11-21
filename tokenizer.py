"""
Yoruba Tokenizer - Character-level tokenizer for Yoruba text with diacritics.
"""

import unicodedata
import torch


class YorubaTokenizer:
    """Character-level tokenizer for Yoruba text with diacritics."""

    def __init__(self, vocab=None):
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.pad_idx = 0
        self.unk_idx = 1

        if vocab is not None:
            # Use provided vocabulary
            self.vocab = vocab
        else:
            # Build default vocabulary
            self.vocab = self._build_default_vocab()

        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.vocab)

    def _build_default_vocab(self):
        """Build default vocabulary with common characters."""
        vocab = [self.pad_token, self.unk_token]
        vocab.extend(list('abcdefghijklmnopqrstuvwxyz'))
        vocab.extend([' ', '.', ',', '!', '?', "'", '"', '-', ':', ';', '(', ')'])
        vocab.extend(list('0123456789'))
        return vocab

    @classmethod
    def from_texts(cls, texts):
        """
        Build tokenizer with vocabulary from actual texts.

        Args:
            texts: List of strings to extract characters from

        Returns:
            YorubaTokenizer with vocabulary built from texts
        """
        # Start with special tokens
        vocab = ['<PAD>', '<UNK>']

        # Collect all unique characters from texts
        char_set = set()
        for text in texts:
            # Normalize unicode
            text = unicodedata.normalize('NFC', text.lower())
            char_set.update(text)

        # Sort for reproducibility
        sorted_chars = sorted(char_set)
        vocab.extend(sorted_chars)

        print(f"Built vocabulary with {len(vocab)} characters from {len(texts)} texts")

        return cls(vocab=vocab)

    def encode(self, text, max_length=150):
        """Convert text to list of indices."""
        text = text.lower()
        indices = []

        # Normalize unicode to handle combining characters
        text = unicodedata.normalize('NFC', text)

        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.unk_idx)

        # Truncate if needed
        if len(indices) > max_length:
            indices = indices[:max_length]

        return indices

    def decode(self, indices):
        """Convert list of indices back to text."""
        chars = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx == self.pad_idx:
                continue
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
            else:
                chars.append(self.unk_token)
        return ''.join(chars)

    def __len__(self):
        return self.vocab_size

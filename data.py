"""
Data utilities for Yoruba diacritic restoration.
"""

import unicodedata
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def strip_diacritics(text):
    """
    Remove diacritics from Yoruba text while preserving base characters.
    This creates the 'input' for our model (text without tones).
    """
    # Mapping of accented characters to base characters
    diacritic_map = {
        # Tonal vowels -> base vowels
        'á': 'a', 'à': 'a', 'â': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i',
        'ó': 'o', 'ò': 'o', 'ô': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u',
        # Special Yoruba characters
        'ẹ': 'e', 'ẹ́': 'e', 'ẹ̀': 'e',
        'ọ': 'o', 'ọ́': 'o', 'ọ̀': 'o',
        'ṣ': 's',
        'ń': 'n', 'ǹ': 'n', 'n̄': 'n',
    }

    # Normalize to NFC form
    text = unicodedata.normalize('NFC', text)

    result = []
    for char in text:
        if char.lower() in diacritic_map:
            # Preserve case
            base = diacritic_map[char.lower()]
            result.append(base.upper() if char.isupper() else base)
        else:
            result.append(char)

    return ''.join(result)


def has_yoruba_diacritics(text):
    """Check if text contains Yoruba diacritics."""
    yoruba_chars = set('áàâéèêíìîóòôúùûẹọṣńǹ')
    text_lower = text.lower()
    return any(char in yoruba_chars for char in text_lower)


def filter_quality_sentences(sentences, min_length=10, max_length=150):
    """Filter sentences for quality."""
    filtered = []
    for sent in sentences:
        # Length check
        if len(sent) < min_length or len(sent) > max_length:
            continue
        # Must contain diacritics
        if not has_yoruba_diacritics(sent):
            continue
        # Basic quality: mostly alphabetic
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in sent) / len(sent)
        if alpha_ratio < 0.8:
            continue
        filtered.append(sent)
    return filtered


def load_yoruba_data_from_file(file_path, max_samples=50000):
    """
    Load Yoruba text from local file.
    Returns list of (stripped_text, original_text) pairs and raw sentences for tokenizer.
    """
    print(f"Loading Yoruba data from {file_path}...")

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Clean and filter lines
    yoruba_sentences = []
    for line in lines:
        line = line.strip()
        if len(line) >= 10 and len(line) <= 150:
            # Check it has some Yoruba diacritics
            if has_yoruba_diacritics(line):
                yoruba_sentences.append(line)
        if len(yoruba_sentences) >= max_samples:
            break

    print(f"Loaded {len(yoruba_sentences)} quality sentences")

    # Create training pairs (stripped -> original)
    pairs = []
    for sent in yoruba_sentences:
        stripped = strip_diacritics(sent)
        # Only include if stripping actually changed something
        if stripped.lower() != sent.lower():
            pairs.append((stripped.lower(), sent.lower()))

    print(f"Created {len(pairs)} training pairs")

    # Return both pairs and raw sentences for building tokenizer
    return pairs, yoruba_sentences


def split_data(pairs, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split data into train, validation, and test sets."""
    random.seed(seed)
    shuffled_pairs = pairs.copy()
    random.shuffle(shuffled_pairs)

    n = len(shuffled_pairs)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_pairs = shuffled_pairs[:train_end]
    val_pairs = shuffled_pairs[train_end:val_end]
    test_pairs = shuffled_pairs[val_end:]

    return train_pairs, val_pairs, test_pairs


class YorubaDiacriticDataset(Dataset):
    """Dataset for Yoruba diacritic restoration."""

    def __init__(self, pairs, tokenizer, max_length=150):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_text, target_text = self.pairs[idx]

        input_ids = self.tokenizer.encode(input_text, self.max_length)
        target_ids = self.tokenizer.encode(target_text, self.max_length)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'input_text': input_text,
            'target_text': target_text
        }


def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]

    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)

    # Ensure same length
    max_len = max(input_ids_padded.size(1), target_ids_padded.size(1))

    if input_ids_padded.size(1) < max_len:
        padding = torch.zeros(input_ids_padded.size(0), max_len - input_ids_padded.size(1), dtype=torch.long)
        input_ids_padded = torch.cat([input_ids_padded, padding], dim=1)

    if target_ids_padded.size(1) < max_len:
        padding = torch.zeros(target_ids_padded.size(0), max_len - target_ids_padded.size(1), dtype=torch.long)
        target_ids_padded = torch.cat([target_ids_padded, padding], dim=1)

    return {
        'input_ids': input_ids_padded,
        'target_ids': target_ids_padded
    }

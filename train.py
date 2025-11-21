"""
Training script for Yoruba diacritic restoration model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

from tokenizer import YorubaTokenizer
from model import DiacriticRestorer
from data import (
    load_yoruba_data_from_file,
    split_data,
    YorubaDiacriticDataset,
    collate_fn
)


def train_one_epoch(model, loader, optimizer, criterion, device, clip_grad=1.0):
    """
    Train for one epoch.

    Returns:
        avg_loss: Average loss over the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids, target_ids)

        # Compute loss (flatten for CrossEntropy)
        logits_flat = logits.view(-1, logits.size(-1))  # [batch*seq_len, vocab]
        targets_flat = target_ids.view(-1)              # [batch*seq_len]

        loss = criterion(logits_flat, targets_flat)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a dataset.

    Returns:
        avg_loss: Average loss
        accuracy: Character-level accuracy
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_chars = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            # Forward pass
            logits = model(input_ids)

            # Compute loss
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_ids.view(-1)
            loss = criterion(logits_flat, targets_flat)

            total_loss += loss.item()
            num_batches += 1

            # Compute accuracy (ignoring padding)
            predictions = torch.argmax(logits, dim=-1)
            mask = target_ids != 0  # Non-padding positions

            correct = (predictions == target_ids) & mask
            total_correct += correct.sum().item()
            total_chars += mask.sum().item()

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_chars if total_chars > 0 else 0

    return avg_loss, accuracy


def calculate_detailed_metrics(model, loader, tokenizer, device):
    """
    Calculate detailed evaluation metrics.

    Returns:
        dict: Various metrics including character accuracy, diacritic accuracy, exact match
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_inputs = []

    # Characters that have diacritics
    diacritic_chars = set('áàâéèêíìîóòôúùûẹọṣńǹ')

    total_chars = 0
    correct_chars = 0
    total_diacritic_positions = 0
    correct_diacritics = 0
    exact_matches = 0
    total_sentences = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing metrics"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            # Get predictions
            logits = model(input_ids)
            predictions = torch.argmax(logits, dim=-1)

            # Process each example in batch
            for i in range(input_ids.size(0)):
                input_seq = input_ids[i].cpu().tolist()
                target_seq = target_ids[i].cpu().tolist()
                pred_seq = predictions[i].cpu().tolist()

                # Find length by looking for where padding starts
                try:
                    input_len = input_seq.index(0) if 0 in input_seq else len(input_seq)
                except ValueError:
                    input_len = len(input_seq)

                try:
                    target_len = target_seq.index(0) if 0 in target_seq else len(target_seq)
                except ValueError:
                    target_len = len(target_seq)

                actual_len = target_len

                # Trim sequences to actual length
                pred_text = tokenizer.decode(pred_seq[:actual_len])
                target_text = tokenizer.decode(target_seq[:actual_len])
                input_text = tokenizer.decode(input_seq[:input_len])

                all_predictions.append(pred_text)
                all_targets.append(target_text)
                all_inputs.append(input_text)

                # Character accuracy
                min_len = min(len(pred_text), len(target_text))
                for j in range(min_len):
                    total_chars += 1
                    if pred_text[j] == target_text[j]:
                        correct_chars += 1

                    # Diacritic-specific accuracy
                    if target_text[j] in diacritic_chars:
                        total_diacritic_positions += 1
                        if pred_text[j] == target_text[j]:
                            correct_diacritics += 1

                # Exact match
                total_sentences += 1
                if pred_text.strip() == target_text.strip():
                    exact_matches += 1

    metrics = {
        'character_accuracy': correct_chars / total_chars if total_chars > 0 else 0,
        'diacritic_accuracy': correct_diacritics / total_diacritic_positions if total_diacritic_positions > 0 else 0,
        'exact_match_rate': exact_matches / total_sentences if total_sentences > 0 else 0,
        'total_sentences': total_sentences,
        'total_chars': total_chars,
        'total_diacritic_positions': total_diacritic_positions
    }

    return metrics, all_predictions, all_targets, all_inputs


def analyze_errors(predictions, targets, inputs, num_examples=20):
    """Analyze common error patterns."""
    errors = []
    confusion_pairs = Counter()

    for pred, tgt, inp in zip(predictions, targets, inputs):
        if pred.strip() != tgt.strip():
            errors.append((inp, tgt, pred))

            # Track character-level confusions
            min_len = min(len(pred), len(tgt))
            for i in range(min_len):
                if pred[i] != tgt[i]:
                    confusion_pairs[(tgt[i], pred[i])] += 1

    print("\n" + "=" * 50)
    print("ERROR ANALYSIS")
    print("=" * 50)
    print(f"\nTotal errors: {len(errors)} / {len(predictions)} ({len(errors)/len(predictions)*100:.1f}%)")

    print("\nMost common character confusions:")
    for (tgt_char, pred_char), count in confusion_pairs.most_common(15):
        print(f"  '{tgt_char}' -> '{pred_char}': {count}")

    print(f"\nSample errors (showing {min(num_examples, len(errors))}):")
    for i, (inp, tgt, pred) in enumerate(errors[:num_examples]):
        print(f"\n  Error {i+1}:")
        print(f"    Input:      {inp}")
        print(f"    Expected:   {tgt}")
        print(f"    Predicted:  {pred}")


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot([acc * 100 for acc in history['val_accuracy']], label='Val Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Training history saved to {save_path}")


def main():
    # Configuration
    DATA_FILE = "Yoruba Text C3 Clean Plus Noisy.txt"
    BATCH_SIZE = 32
    MAX_LENGTH = 150
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    CLIP_GRAD = 1.0

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_pairs, raw_sentences = load_yoruba_data_from_file(DATA_FILE, max_samples=50000)

    # Build tokenizer from raw sentences
    tokenizer = YorubaTokenizer.from_texts(raw_sentences)

    # Split data
    train_pairs, val_pairs, test_pairs = split_data(data_pairs)
    print(f"Train: {len(train_pairs)} pairs")
    print(f"Validation: {len(val_pairs)} pairs")
    print(f"Test: {len(test_pairs)} pairs")

    # Create datasets and dataloaders
    train_dataset = YorubaDiacriticDataset(train_pairs, tokenizer, MAX_LENGTH)
    val_dataset = YorubaDiacriticDataset(val_pairs, tokenizer, MAX_LENGTH)
    test_dataset = YorubaDiacriticDataset(test_pairs, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = DiacriticRestorer(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    ).to(device)

    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    # Training loop
    best_val_loss = float('inf')

    print("\nStarting training...")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 50)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 30)

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, CLIP_GRAD)

        # Evaluate
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy*100:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'tokenizer_vocab': tokenizer.vocab
            }, 'best_model.pt')
            print("Saved best model!")

    print("\nTraining complete!")

    # Plot training history
    plot_training_history(history)

    # Load best model and evaluate on test set
    checkpoint = torch.load('best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")

    print("\nEvaluating on test set...")
    metrics, predictions, targets, inputs = calculate_detailed_metrics(model, test_loader, tokenizer, device)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"Character Accuracy: {metrics['character_accuracy']*100:.2f}%")
    print(f"Diacritic Accuracy: {metrics['diacritic_accuracy']*100:.2f}%")
    print(f"Exact Match Rate: {metrics['exact_match_rate']*100:.2f}%")

    # Error analysis
    analyze_errors(predictions, targets, inputs)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': tokenizer.vocab_size,
            'embedding_dim': model.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers
        },
        'tokenizer_vocab': tokenizer.vocab,
        'metrics': metrics
    }, 'yoruba_diacritic_model.pt')
    print("\nModel saved to yoruba_diacritic_model.pt")


if __name__ == '__main__':
    main()

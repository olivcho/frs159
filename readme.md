Yoruba Diacritic Restoration Project
Project Overview
A context-aware neural network that restores missing tonal diacritics in Yoruba text using surrounding context. This addresses a real NLP problem: 90%+ of digital Yoruba text omits diacritics due to keyboard accessibility, creating semantic ambiguity.
Example:
    •    Input: o ri owo lana
    •    Output: ó rí owó lána (he saw money yesterday)
Problem Statement
Yoruba is a tonal language with 3 tones (High á, Mid a, Low à) where tone changes meaning:
    •    ọwọ́ (hand) vs owó (money) vs ọ̀wọ̀ (respect)
    •    igba (calabash) vs ìgbà (time/season)
Most digital text omits these critical markers, causing ambiguity for both humans and NLP systems.
Technical Architecture
Data Pipeline
    1    Source: JW300 Yoruba corpus (~250K sentences with proper diacritics)
    2    Preprocessing: Filter for quality (length, diacritic presence)
    3    Synthetic Training: Strip diacritics from clean text to create input-output pairs
    4    Split: 80% train, 10% validation, 10% test
Model Architecture
Input (no diacritics) → Embedding → Bi-LSTM Encoder → LSTM Decoder → Output (with diacritics)
Components:
    •    Embedding Layer: Converts character indices to dense 128-dim vectors
    •    Encoder: 2-layer bidirectional LSTM (256 hidden units)
    ◦    Reads context from both directions
    ◦    Captures dependencies between words
    •    Decoder: 2-layer LSTM (512 hidden units)
    ◦    Predicts characters with correct diacritics
    ◦    Uses encoded context representation
    •    Output Layer: Linear projection to vocabulary size
Key Design Choices:
    •    Character-level (not word-level) for better handling of morphology
    •    Bidirectional encoding to use full sentence context
    •    Teacher forcing during training for faster convergence
Tokenizer
Custom character-level tokenizer:
    •    Vocabulary: ~70 characters (base letters + tonal vowels + punctuation)
    •    Special tokens: <PAD> (0), <UNK> (1)
    •    Max sequence length: 150 characters
    •    Case-insensitive
Training Configuration
    •    Loss: CrossEntropyLoss (ignore padding)
    •    Optimizer: Adam (lr=0.001)
    •    Batch Size: 32
    •    Epochs: 5
    •    Regularization: Dropout (0.3), gradient clipping (1.0)
    •    LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)
    •    Hardware: Google Colab T4 GPU (free)
Project Structure
yoruba-diacritics/
├── data/
│   ├── raw/              # Downloaded JW300 corpus
│   ├── processed/        # Filtered and split data
│   └── stats.json        # Dataset statistics
├── models/
│   ├── tokenizer.py      # YorubaTokenizer class
│   ├── model.py          # DiacriticRestorer architecture
│   └── best_model.pt     # Saved checkpoint
├── train.py              # Training script
├── evaluate.py           # Evaluation and metrics
├── inference.py          # Restore diacritics on new text
├── notebooks/
│   └── main.ipynb        # Complete Colab notebook
├── requirements.txt
└── README.md
Key Classes and Functions
YorubaTokenizer
class YorubaTokenizer:
    def __init__(self)
    def encode(text, max_length=150) -> List[int]
    def decode(indices) -> str
Converts between text and numerical indices for neural network processing.
DiacriticRestorer
class DiacriticRestorer(nn.Module):
    def __init__(vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
    def forward(input_ids, target_ids=None) -> logits
Main neural network model. Returns logits of shape [batch, seq_len, vocab_size].
Training Functions
def train_one_epoch(model, loader, optimizer, criterion, device) -> avg_loss
def evaluate(model, loader, criterion, device) -> (avg_loss, accuracy)
Inference
def restore_diacritics(text, model, tokenizer, device) -> str
Main function to restore diacritics on new text.
Expected Performance
    •    Character Accuracy: 75-85%
    •    Diacritic-Specific Accuracy: 70-80%
    •    Exact Match Rate: 40-60%
The model performs best when:
    •    Sentence has clear context
    •    Words are in training vocabulary
    •    Standard Yoruba grammar patterns
Common failure cases:
    •    Rare words not in training set
    •    Genuinely ambiguous contexts
    •    Code-switching (Yoruba-English)
    •    Non-standard orthography
Development Guidelines
Adding New Features
    1    New tokenizer: Modify YorubaTokenizer class, ensure vocab compatibility
    2    Architecture changes: Update DiacriticRestorer.__init__(), maintain input/output shapes
    3    New metrics: Add to calculate_detailed_metrics() function
Debugging Tips
    •    Check tokenizer encode/decode round-trip
    •    Verify tensor shapes at each layer
    •    Use small batch (1-2 examples) for debugging
    •    Print intermediate activations if loss doesn't decrease
Performance Optimization
    •    Increase batch size if GPU memory allows
    •    Add more LSTM layers for complex patterns
    •    Try transformer architecture (requires more data)
    •    Data augmentation: add noise to training data
Dependencies
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
Usage Examples
Training
# Load data
dataset = load_dataset("Helsinki-NLP/opus-100", "en-yo")
train_pairs = create_training_pairs(dataset)

# Create model
model = DiacriticRestorer(vocab_size=tokenizer.vocab_size).to(device)

# Train
for epoch in range(5):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
Inference
# Load model
model, tokenizer = load_model('yoruba_diacritic_model.pt', device)

# Restore diacritics
text = "omo naa lo si ile"
restored = restore_diacritics(text, model, tokenizer, device)
print(restored)  # "ọmọ náà lọ sí ilé"
Research Context
This project addresses low-resource NLP challenges for African languages:
    •    Limited labeled datasets
    •    Orthographic inconsistency in digital text
    •    Cultural and linguistic preservation through technology
Related Work:
    •    Neural diacritic restoration for Arabic (Darwish 2014)
    •    Low-resource sequence labeling (Garrette & Baldridge 2013)
    •    Character-level models for morphologically rich languages
Novel Contribution: Quantifies how much tonal information loss affects downstream NLP tasks (sentiment analysis, NER, etc.).
Future Extensions
    1    Attention mechanism: Visualize which context words influence predictions
    2    Multi-task learning: Joint training with POS tagging or NER
    3    Transfer learning: Extend to other tonal languages (Igbo, Twi, Ga)
    4    Confidence scores: Output probability distributions for ambiguous cases
    5    Interactive correction: Allow users to fix errors and retrain
Academic Context
For Princeton FRS159 (African Languages & NLP) final project:
    •    Language Profile: Yoruba linguistic features, speaker statistics, geographic distribution
    •    Technical Component: Context-aware diacritic restoration system
    •    Analysis: Error patterns, ambiguity resolution, impact on NLP tasks
Deliverables:
    •    15-min presentation + 5-min Q&A
    •    2-page technical report
    •    Live demonstration
    •    Code repository
Common Issues & Solutions
Issue: Model predicts same character repeatedly
Solution: Check if targets are being used in decoder (teacher forcing). Verify gradient flow.
Issue: Loss not decreasing
Solution:
    •    Check tokenizer encoding matches model vocabulary
    •    Verify loss function ignores padding (index 0)
    •    Reduce learning rate
    •    Check for NaN gradients
Issue: High training accuracy but low validation accuracy
Solution: Overfitting. Add dropout, reduce model size, or increase training data.
Issue: Out of memory error
Solution: Reduce batch size, decrease hidden_dim, or use gradient accumulation.
Contact & Resources
    •    Dataset: OPUS-100 (https://huggingface.co/datasets/Helsinki-NLP/opus-100)
    •    Yoruba Grammar: Modern Yoruba Grammar by Awobuluyi (https://example.com/)
    •    NLP Resources: MasakhaNE Project (https://www.masakhane.io/)
License
MIT License - Academic use encouraged

Last Updated: 2024 Author: Oliver (Princeton ECE '28) Course: FRS159 - African Languages and NLP
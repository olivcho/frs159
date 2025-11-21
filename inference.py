"""
Inference script for Yoruba diacritic restoration.
"""

import torch
from tokenizer import YorubaTokenizer
from model import DiacriticRestorer


def load_model(path, device=None):
    """
    Load model for inference.

    Args:
        path: Path to saved model checkpoint
        device: torch device (defaults to cuda if available)

    Returns:
        model: Loaded DiacriticRestorer model
        tokenizer: YorubaTokenizer with vocabulary from training
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path, map_location=device)

    # Recreate tokenizer with saved vocabulary
    tokenizer = YorubaTokenizer(vocab=checkpoint['tokenizer_vocab'])

    # Recreate model
    config = checkpoint['model_config']
    model = DiacriticRestorer(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, tokenizer, device


def restore_diacritics(text, model, tokenizer, device):
    """
    Restore diacritics in Yoruba text.

    Args:
        text: Input text without diacritics
        model: Trained DiacriticRestorer model
        tokenizer: YorubaTokenizer instance
        device: torch device

    Returns:
        str: Text with restored diacritics
    """
    model.eval()

    # Encode input
    input_ids = tokenizer.encode(text.lower())
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Get prediction
    with torch.no_grad():
        logits = model(input_tensor)
        predictions = torch.argmax(logits, dim=-1)

    # Decode
    restored = tokenizer.decode(predictions[0].cpu().tolist())

    return restored


def demo_restoration(texts, model, tokenizer, device):
    """
    Demo function to show diacritic restoration.
    """
    print("\n" + "=" * 60)
    print("DIACRITIC RESTORATION DEMO")
    print("=" * 60)

    for text in texts:
        restored = restore_diacritics(text, model, tokenizer, device)
        print(f"\nInput:    {text}")
        print(f"Restored: {restored}")


def interactive_demo(model, tokenizer, device):
    """
    Interactive demo for diacritic restoration.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE YORUBA DIACRITIC RESTORATION")
    print("=" * 60)
    print("Enter Yoruba text without diacritics to restore them.")
    print("Type 'quit' to exit.")
    print("")

    while True:
        try:
            text = input("Input: ").strip()
        except EOFError:
            break

        if text.lower() == 'quit':
            print("Goodbye!")
            break

        if not text:
            continue

        restored = restore_diacritics(text, model, tokenizer, device)
        print(f"Restored: {restored}")
        print("")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Yoruba Diacritic Restoration')
    parser.add_argument('--model', type=str, default='yoruba_diacritic_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to restore diacritics for')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive demo')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with example sentences')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model, tokenizer, device = load_model(args.model)
    print(f"Model loaded. Using device: {device}")

    if args.text:
        # Single text restoration
        restored = restore_diacritics(args.text, model, tokenizer, device)
        print(f"\nInput:    {args.text}")
        print(f"Restored: {restored}")

    elif args.demo:
        # Demo with example sentences
        demo_texts = [
            "o ri owo lana",
            "omo naa lo si ile",
            "e ku aaro",
            "mo fe lo si oja",
            "baba mi wa nile",
            "olorun a bukun e",
            "ise agbe ni ise ile wa",
            "ojo ti de",
        ]
        demo_restoration(demo_texts, model, tokenizer, device)

    elif args.interactive:
        # Interactive mode
        interactive_demo(model, tokenizer, device)

    else:
        # Default: show demo
        demo_texts = [
            "o ri owo lana",
            "omo naa lo si ile",
            "e ku aaro",
        ]
        demo_restoration(demo_texts, model, tokenizer, device)
        print("\nUse --interactive for interactive mode or --text 'your text' for single restoration")


if __name__ == '__main__':
    main()

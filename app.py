"""
Flask API server for Yoruba diacritic restoration.
"""

from flask import Flask, request, jsonify
import torch
from tokenizer import YorubaTokenizer
from model import DiacriticRestorer

app = Flask(__name__)

# Global model variables
model = None
tokenizer = None
device = None


def load_model(path='FINALMODEL.pt'):
    """Load the model on startup."""
    global model, tokenizer, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint = torch.load(path, map_location=device)

    # Recreate tokenizer with saved vocabulary
    tokenizer = YorubaTokenizer(vocab=checkpoint['tokenizer_vocab'])

    # Recreate model - handle both checkpoint formats
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        model = DiacriticRestorer(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        ).to(device)
    else:
        model = DiacriticRestorer(
            vocab_size=len(checkpoint['tokenizer_vocab']),
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from {path}")


def restore_diacritics(text):
    """Restore diacritics in Yoruba text."""
    input_ids = tokenizer.encode(text.lower())
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        predictions = torch.argmax(logits, dim=-1)

    restored = tokenizer.decode(predictions[0].cpu().tolist())
    return restored


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.route('/restore', methods=['POST'])
def restore():
    """
    Restore diacritics in Yoruba text.

    Request JSON:
        {"text": "omo naa lo si ile"}

    Response JSON:
        {"input": "omo naa lo si ile", "restored": "ọmọ náà lọ sí ilé"}
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request'}), 400

    text = data['text']

    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'Invalid text input'}), 400

    restored = restore_diacritics(text)

    return jsonify({
        'input': text,
        'restored': restored
    })


@app.route('/restore/batch', methods=['POST'])
def restore_batch():
    """
    Restore diacritics for multiple texts.

    Request JSON:
        {"texts": ["omo naa lo si ile", "e ku aaro"]}

    Response JSON:
        {"results": [{"input": "...", "restored": "..."}, ...]}
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()

    if not data or 'texts' not in data:
        return jsonify({'error': 'Missing "texts" field in request'}), 400

    texts = data['texts']

    if not isinstance(texts, list):
        return jsonify({'error': '"texts" must be a list'}), 400

    results = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            restored = restore_diacritics(text)
            results.append({'input': text, 'restored': restored})
        else:
            results.append({'input': text, 'error': 'Invalid text'})

    return jsonify({'results': results})


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Yoruba Diacritic Restoration API')
    parser.add_argument('--model', type=str, default='FINALMODEL.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run server on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run server on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')

    args = parser.parse_args()

    # Load model
    load_model(args.model)

    # Run server
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("\nEndpoints:")
    print("  GET  /health         - Health check")
    print("  POST /restore        - Restore diacritics for single text")
    print("  POST /restore/batch  - Restore diacritics for multiple texts")
    print("\nExample usage:")
    print('  curl -X POST http://localhost:5000/restore -H "Content-Type: application/json" -d \'{"text": "omo naa lo si ile"}\'')

    app.run(host=args.host, port=args.port, debug=args.debug)

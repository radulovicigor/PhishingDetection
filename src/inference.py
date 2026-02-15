"""
Inference module.
Provides CLI for single email prediction with explanation.
"""
import argparse
import json
import sys
from typing import Optional

from src.config import MODEL_PATH
from src.train import load_model
from src.explain import explain_email
from src.utils import setup_logger

logger = setup_logger(__name__)


def predict_email(
    text: str,
    urls: str = "",
    model_path: Optional[str] = None
) -> dict:
    """
    Predict whether an email is phishing.
    
    Args:
        text: Email text (subject + body combined)
        urls: URLs string (optional)
        model_path: Path to model file (optional)
        
    Returns:
        Dictionary with prediction results and explanation
    """
    # Load model
    path = model_path or MODEL_PATH
    pipeline = load_model(path)
    
    # Get explanation (includes prediction)
    result = explain_email(pipeline, text, urls)
    
    return result


def format_output(result: dict, pretty: bool = True) -> str:
    """
    Format prediction result as JSON string.
    
    Args:
        result: Prediction result dictionary
        pretty: Whether to pretty-print
        
    Returns:
        JSON string
    """
    # Create clean output
    output = {
        'label': result['label'],
        'score': result['score'],
        'reasons': result['reasons'],
        'heuristics': result['heuristics'],
    }
    
    if pretty:
        return json.dumps(output, indent=2, ensure_ascii=False)
    return json.dumps(output, ensure_ascii=False)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phishing Email Detector - Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.inference --text "Your account has been suspended. Click here to verify."
  python -m src.inference --text "Meeting tomorrow at 3pm" --urls ""
  python -m src.inference --subject "URGENT" --body "Click to verify" --urls "http://bit.ly/xyz"
        """
    )
    
    # Input options
    parser.add_argument(
        '--text',
        type=str,
        help='Combined email text (subject + body)'
    )
    parser.add_argument(
        '--subject',
        type=str,
        default='',
        help='Email subject (combined with body if text not provided)'
    )
    parser.add_argument(
        '--body',
        type=str,
        default='',
        help='Email body (combined with subject if text not provided)'
    )
    parser.add_argument(
        '--urls',
        type=str,
        default='',
        help='URLs in the email (comma-separated)'
    )
    
    # Output options
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output raw JSON (for scripting)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model file (default: models/model.joblib)'
    )
    
    args = parser.parse_args()
    
    # Determine text input
    if args.text:
        text = args.text
    elif args.subject or args.body:
        text = f"{args.subject}\n{args.body}"
    else:
        parser.error("Please provide --text or --subject/--body")
        return 1
    
    try:
        # Run prediction
        result = predict_email(text, args.urls, args.model)
        
        if args.json:
            # Raw JSON output
            print(format_output(result, pretty=False))
        else:
            # Human-readable output
            print("\n" + "=" * 50)
            print("PHISHING DETECTION RESULT")
            print("=" * 50)
            
            # Prediction
            label = result['label']
            score = result['score']
            
            if label == "PHISHING":
                print(f"\n[!] PREDICTION: {label}")
            else:
                print(f"\n[OK] PREDICTION: {label}")
            
            print(f"   Confidence Score: {score:.2%}")
            
            # Reasons
            print("\nREASONS:")
            if result['reasons']:
                for reason in result['reasons']:
                    print(f"   - {reason}")
            else:
                print("   No specific reasons identified.")
            
            # Key heuristics
            print("\nKEY INDICATORS:")
            heuristics = result['heuristics']
            print(f"   URLs found: {heuristics['url_count']}")
            print(f"   Keyword hits: {heuristics['keyword_hits']}")
            print(f"   Suspicious TLD: {'Yes' if heuristics['suspicious_tld'] else 'No'}")
            print(f"   URL shortener: {'Yes' if heuristics['uses_shortener'] else 'No'}")
            print(f"   CAPS ratio: {heuristics['caps_ratio']:.1%}")
            
            print("\n" + "=" * 50)
        
        return 0
        
    except FileNotFoundError as e:
        if args.json:
            print(json.dumps({'error': str(e)}))
        else:
            print(f"Error: {e}")
            print("Please run 'python -m src.train' first to train the model.")
        return 1
    except Exception as e:
        if args.json:
            print(json.dumps({'error': str(e)}))
        else:
            print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

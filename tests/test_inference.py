"""
Tests for inference module.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config import MODEL_PATH


class TestInferenceHelpers:
    """Tests for inference helper functions."""
    
    def test_model_path_exists(self):
        """Verify model path is configured correctly."""
        assert MODEL_PATH.name == "model.joblib"
        assert "models" in str(MODEL_PATH)


class TestExplanationParsing:
    """Tests for explanation parsing logic."""
    
    def test_generate_reasons_format(self):
        """Test that reasons are formatted correctly."""
        from src.explain import generate_reasons
        
        contributions = [
            {'feature': 'url_count', 'value': 3, 'contribution': 0.5},
            {'feature': 'keyword_hits', 'value': 5, 'contribution': 0.3},
        ]
        
        reasons = generate_reasons(contributions, "Test text", "")
        
        assert isinstance(reasons, list)
        for reason in reasons:
            assert isinstance(reason, str)
    
    def test_generate_reasons_url_count(self):
        """Test URL count reason generation."""
        from src.explain import generate_reasons
        
        contributions = [
            {'feature': 'url_count', 'value': 3, 'contribution': 0.5},
        ]
        
        reasons = generate_reasons(contributions, "http://a.com", "")
        
        # Should mention URLs
        if reasons:
            assert any("URL" in r for r in reasons)
    
    def test_generate_reasons_keywords(self):
        """Test keyword reason generation."""
        from src.explain import generate_reasons
        
        contributions = [
            {'feature': 'keyword_hits', 'value': 5, 'contribution': 0.4},
        ]
        
        reasons = generate_reasons(contributions, "verify password account", "")
        
        if reasons:
            assert any("keyword" in r.lower() for r in reasons)


class TestPredictEmailFunction:
    """Tests for predict_email function (requires trained model)."""
    
    def test_predict_email_mock(self):
        """Test prediction with mocked explain_email."""
        mock_result = {
            'label': 'PHISHING',
            'score': 0.8,
            'prediction': 1,
            'reasons': ['Contains suspicious content'],
            'heuristics': {'url_count': 1},
            'top_contributing_features': [],
        }
        
        with patch('src.inference.load_model') as mock_load:
            with patch('src.inference.explain_email', return_value=mock_result):
                mock_load.return_value = MagicMock()
                
                from src.inference import predict_email
                result = predict_email("Test suspicious email", "http://bad.xyz")
                
                assert result['label'] == 'PHISHING'
                assert result['score'] == 0.8


class TestCLIArguments:
    """Tests for CLI argument parsing."""
    
    def test_cli_help(self):
        """Test that CLI can show help."""
        import subprocess
        import sys
        
        # This should not raise an error
        result = subprocess.run(
            [sys.executable, "-m", "src.inference", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        
        assert "Phishing Email Detector" in result.stdout or result.returncode == 0


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not trained - run 'python -m src.train' first"
)
class TestInferenceWithModel:
    """Integration tests that require a trained model."""
    
    def test_predict_phishing_email(self):
        """Test prediction on obvious phishing email."""
        from src.inference import predict_email
        
        phishing_text = """
        URGENT: Your account has been suspended!
        
        Click here immediately to verify your identity:
        http://secure-verify.xyz/login
        
        If you don't verify within 24 hours, your account will be permanently deleted.
        """
        
        result = predict_email(phishing_text, "http://secure-verify.xyz/login")
        
        assert 'label' in result
        assert 'score' in result
        assert 'reasons' in result
        assert result['label'] in ['PHISHING', 'LEGIT']
        assert 0 <= result['score'] <= 1
    
    def test_predict_legit_email(self):
        """Test prediction on legitimate email."""
        from src.inference import predict_email
        
        legit_text = """
        Hi team,
        
        Just a reminder that our weekly meeting is tomorrow at 3pm.
        Please bring your status updates.
        
        Thanks,
        John
        """
        
        result = predict_email(legit_text, "")
        
        assert 'label' in result
        assert 'score' in result
        assert 'reasons' in result
    
    def test_predict_empty_urls(self):
        """Test prediction with empty URLs."""
        from src.inference import predict_email
        
        result = predict_email("Simple test message", "")
        
        assert 'label' in result
        assert result['heuristics']['url_count'] == 0


class TestFormatOutput:
    """Tests for output formatting."""
    
    def test_format_output_json(self):
        """Test JSON output formatting."""
        from src.inference import format_output
        
        result = {
            'label': 'PHISHING',
            'score': 0.85,
            'reasons': ['Test reason'],
            'heuristics': {'url_count': 1},
        }
        
        output = format_output(result, pretty=True)
        parsed = json.loads(output)
        
        assert parsed['label'] == 'PHISHING'
        assert parsed['score'] == 0.85
    
    def test_format_output_compact(self):
        """Test compact JSON output."""
        from src.inference import format_output
        
        result = {
            'label': 'LEGIT',
            'score': 0.1,
            'reasons': [],
            'heuristics': {},
        }
        
        output = format_output(result, pretty=False)
        
        # Should be single line
        assert '\n' not in output
        assert json.loads(output)['label'] == 'LEGIT'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

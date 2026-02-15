"""
Tests for feature extraction module.
"""
import pytest
import numpy as np

from src.features import (
    count_urls,
    has_url,
    count_exclamations,
    count_questions,
    get_text_length,
    get_word_count,
    get_caps_ratio,
    count_keyword_hits,
    has_suspicious_tld,
    uses_url_shortener,
    extract_urls_from_text,
    parse_urls_column,
    extract_domain,
    extract_tld,
    extract_all_heuristics,
    HeuristicFeaturesTransformer,
)


class TestURLExtraction:
    """Tests for URL extraction functions."""
    
    def test_extract_urls_from_text_http(self):
        text = "Click here: http://example.com/page"
        urls = extract_urls_from_text(text)
        assert len(urls) == 1
        assert "http://example.com/page" in urls[0]
    
    def test_extract_urls_from_text_https(self):
        text = "Visit https://secure.example.com"
        urls = extract_urls_from_text(text)
        assert len(urls) == 1
        assert "https://secure.example.com" in urls[0]
    
    def test_extract_urls_from_text_multiple(self):
        text = "Links: http://a.com and https://b.com here"
        urls = extract_urls_from_text(text)
        assert len(urls) == 2
    
    def test_extract_urls_from_text_empty(self):
        assert extract_urls_from_text("") == []
        assert extract_urls_from_text("No links here") == []
    
    def test_parse_urls_column_comma(self):
        urls = parse_urls_column("http://a.com, http://b.com")
        assert len(urls) == 2
    
    def test_parse_urls_column_space(self):
        urls = parse_urls_column("http://a.com http://b.com")
        assert len(urls) == 2
    
    def test_parse_urls_column_empty(self):
        assert parse_urls_column("") == []
        assert parse_urls_column(None) == []


class TestDomainExtraction:
    """Tests for domain extraction functions."""
    
    def test_extract_domain_http(self):
        assert extract_domain("http://example.com/path") == "example.com"
    
    def test_extract_domain_https(self):
        assert extract_domain("https://sub.example.com") == "sub.example.com"
    
    def test_extract_domain_no_protocol(self):
        assert extract_domain("example.com/path") == "example.com"
    
    def test_extract_tld(self):
        assert extract_tld("example.com") == "com"
        assert extract_tld("sub.example.co.uk") == "uk"
        assert extract_tld("localhost") is None


class TestURLHeuristics:
    """Tests for URL-related heuristic features."""
    
    def test_count_urls_from_text(self):
        text = "Visit http://a.com and http://b.com"
        assert count_urls(text, "") == 2
    
    def test_count_urls_from_column(self):
        text = "No urls here"
        urls_col = "http://a.com, http://b.com"
        assert count_urls(text, urls_col) == 2
    
    def test_count_urls_combined(self):
        text = "Visit http://a.com"
        urls_col = "http://b.com"
        # Combined unique URLs
        count = count_urls(text, urls_col)
        assert count >= 1
    
    def test_has_url(self):
        assert has_url("http://example.com", "") == 1
        assert has_url("no url", "") == 0
    
    def test_uses_url_shortener_bitly(self):
        text = "Click: http://bit.ly/abc123"
        assert uses_url_shortener(text, "") == 1
    
    def test_uses_url_shortener_tinyurl(self):
        text = "Click: https://tinyurl.com/xyz"
        assert uses_url_shortener(text, "") == 1
    
    def test_uses_url_shortener_none(self):
        text = "Visit https://legitimate-site.com"
        assert uses_url_shortener(text, "") == 0
    
    def test_suspicious_tld_xyz(self):
        text = "Visit http://malicious.xyz"
        assert has_suspicious_tld(text, "") == 1
    
    def test_suspicious_tld_tk(self):
        text = "Link: http://free.tk"
        assert has_suspicious_tld(text, "") == 1
    
    def test_suspicious_tld_com(self):
        text = "Visit http://safe.com"
        assert has_suspicious_tld(text, "") == 0


class TestTextHeuristics:
    """Tests for text-based heuristic features."""
    
    def test_count_exclamations(self):
        assert count_exclamations("Hello!") == 1
        assert count_exclamations("WOW!!! Amazing!!") == 5
        assert count_exclamations("No exclamation") == 0
    
    def test_count_questions(self):
        assert count_questions("What?") == 1
        assert count_questions("Really?? Why??") == 4
        assert count_questions("No question") == 0
    
    def test_get_text_length(self):
        assert get_text_length("Hello") == 5
        assert get_text_length("") == 0
    
    def test_get_word_count(self):
        assert get_word_count("Hello world") == 2
        assert get_word_count("One") == 1
        assert get_word_count("") == 0
    
    def test_caps_ratio_all_caps(self):
        ratio = get_caps_ratio("HELLO WORLD TEST")
        assert ratio == 1.0
    
    def test_caps_ratio_none(self):
        ratio = get_caps_ratio("hello world test")
        assert ratio == 0.0
    
    def test_caps_ratio_mixed(self):
        ratio = get_caps_ratio("HELLO world TEST normal")
        assert 0.0 < ratio < 1.0
    
    def test_keyword_hits_multiple(self):
        text = "Please verify your account password immediately"
        hits = count_keyword_hits(text)
        assert hits >= 3  # verify, account, password
    
    def test_keyword_hits_none(self):
        text = "Meeting scheduled for tomorrow afternoon"
        hits = count_keyword_hits(text)
        assert hits == 0


class TestExtractAllHeuristics:
    """Tests for combined heuristic extraction."""
    
    def test_extract_all_heuristics_phishing(self):
        text = "URGENT! Verify your account! http://malicious.xyz"
        urls = ""
        features = extract_all_heuristics(text, urls)
        
        assert features['url_count'] >= 1
        assert features['has_url'] == 1
        assert features['exclamation_count'] >= 2
        assert features['keyword_hits'] >= 2
        assert features['suspicious_tld'] == 1
    
    def test_extract_all_heuristics_legit(self):
        text = "Meeting tomorrow at 3pm. Please confirm."
        urls = ""
        features = extract_all_heuristics(text, urls)
        
        assert features['url_count'] == 0
        assert features['has_url'] == 0
        assert features['suspicious_tld'] == 0


class TestHeuristicFeaturesTransformer:
    """Tests for the scikit-learn compatible transformer."""
    
    def test_transformer_fit(self):
        transformer = HeuristicFeaturesTransformer()
        # fit should be a no-op
        result = transformer.fit(None)
        assert result is transformer
    
    def test_transformer_transform_array(self):
        transformer = HeuristicFeaturesTransformer()
        X = np.array([
            ["Hello http://example.com", ""],
            ["No urls here", "http://test.com"],
        ])
        result = transformer.transform(X)
        
        assert result.shape == (2, 10)  # 2 samples, 10 features
        assert result.dtype == np.float64
    
    def test_transformer_transform_dataframe(self):
        import pandas as pd
        
        transformer = HeuristicFeaturesTransformer()
        df = pd.DataFrame({
            'text': ["Hello http://example.com", "No urls"],
            'urls': ["", "http://test.com"],
        })
        result = transformer.transform(df)
        
        assert result.shape == (2, 10)
    
    def test_transformer_feature_names(self):
        transformer = HeuristicFeaturesTransformer()
        names = transformer.get_feature_names_out()
        
        assert len(names) == 10
        assert 'url_count' in names
        assert 'keyword_hits' in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

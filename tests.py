# tests.py
from main.sentiment_model import analyze_sentiment
from unittest.mock import patch
from main.analyzer import final_conversation_sentiment

def test_positive_sentiment():
    sentiment, score = analyze_sentiment("Thank you, the issue was resolved")
    assert sentiment == "Positive"
    assert score > 0.05

def test_custom_negative_word():
    sentiment, score = analyze_sentiment("I want a refund, the product was broken")
    assert sentiment == "Negative"
    assert score < -0.05

def test_strong_negative_keyword():
    sentiment, score = analyze_sentiment("This is the worst experience ever")
    assert sentiment == "Negative"
    assert score < -0.5  # worst = -4.0 override


def test_mixed_sentence():
    sentiment, score = analyze_sentiment("Good service but delay in delivery")
    # Could be neutral depending on balance
    assert sentiment in ["Positive", "Neutral", "Negative"]
    assert -1.0 <= score <= 1.0

def test_overall_positive_sentiment():
    mock_data = [
        {"score": 0.7},
        {"score": 0.5},
        {"score": 0.4},
    ]

    with patch("main.analyzer.load_conversation", return_value=mock_data):
        result = final_conversation_sentiment()
        assert result.startswith("Positive")

def test_overall_negative_sentiment():
    mock_data = [
        {"score": -0.6},
        {"score": -0.4},
        {"score": -0.5},
    ]

    with patch("main.analyzer.load_conversation", return_value=mock_data):
        result = final_conversation_sentiment()
        assert result.startswith("Negative")

def test_overall_neutral_sentiment():
    mock_data = [
        {"score": 0.1},
        {"score": -0.1},
        {"score": 0.0},
    ]

    with patch("main.analyzer.load_conversation", return_value=mock_data):
        result = final_conversation_sentiment()
        assert result.startswith("Neutral")

def test_no_conversation_found():
    with patch("main.analyzer.load_conversation", return_value=[]):
        result = final_conversation_sentiment()
        assert result == "No conversation found."
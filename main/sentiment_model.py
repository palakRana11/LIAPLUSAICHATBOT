from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create analyzer
analyzer = SentimentIntensityAnalyzer()


custom_words = {
    # Strong negative (customer frustration)
    "refund": -3.0,
    "worst": -4.0,
    "angry": -3.5,
    "complaint": -3.0,
    "issue": -2.0,
    "problem": -2.2,
    "broken": -2.5,
    "delay": -2.0,
    "charged": -1.8,
    "scam": -4.0,
    "frustated": -4.0,

    # Soft negative
    "confused": -1.5,
    "unclear": -1.2,
    "not working": -2.3,

    # Positive customer-care context
    "thank you": 2.5,
    "good": 2.1,
    "great": 2.9,
    "resolved": 2.8,
    "appreciate": 2.3,
    "excellent": 3.0,
    "helpful": 2.0,
    "working": 1.8,
    "worked": 1.8,
    "fixed": 2.2,

    # Customer-intent awareness
    "support": 1.2,
    "agent": 0.8,
    "service": 0.5,
}

# Inject into VADER
for word, score in custom_words.items():
    analyzer.lexicon[word] = score


# SENTIMENT FUNCTION 

def analyze_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    compound = sentiment_scores["compound"]  # -1 → 1

    # Human-readable sentiment label
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Save full numeric compound value (not 1 or -1)
    return sentiment, float(compound)

from main.storage import load_conversation

def final_conversation_sentiment():
    convo = load_conversation()
    if not convo:
        return "No conversation found."

    scores = [m["score"] for m in convo]

    avg = sum(scores) / len(scores)

    if avg > 0.2:
        result = "Positive – overall satisfaction"
    elif avg < -0.2:
        result = "Negative – general dissatisfaction"
    else:
        result = "Neutral – mixed sentiment"

    return result

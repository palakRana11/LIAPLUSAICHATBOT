import json
import os

FILE_PATH = "data/conversation.json"

def load_conversation():
    if not os.path.exists(FILE_PATH):
        return []
    with open(FILE_PATH, "r") as f:
        try:
            return json.load(f)
        except:
            return []

def save_user_message(message, sentiment, score, role="user"):
    convo = load_conversation()
    convo.append({
        "role": role,
        "message": message,
        "sentiment": sentiment,
        "score": score
    })
    with open(FILE_PATH, "w") as f:
        json.dump(convo, f, indent=4)

def clear_conversation():
    # Do not delete file
    with open(FILE_PATH, "w") as f:
        json.dump([], f)

from sentence_transformers import SentenceTransformer, util

# Load a small embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

knowledge_base = {
    # General Service Questions
    "Hello": "Hi! How are you doing? How can I help you today.",
    "What is your return policy?": "Our return policy allows returns within 30 days.",
    "What are your working hours?": "We operate from 9 AM to 6 PM, Monday to Saturday.",
    "How can I reset my password?": "You can reset your password using the 'Forgot Password' option on the login page.",
    "Do you offer customer support?": "Yes, we provide 24/7 customer support through chat and email.",
    "Help me with this": "Ofcourse! Let me guide you through the process",
    
    # Customer Satisfaction
    "Your service is really good": "Thank you for your kind feedback! It motivates us to keep improving.",
    "I am happy with the support team": "We are glad to hear that! Your satisfaction is our priority.",
    "I am happy": "We are glad to hear that! Your satisfaction is our priority.",
    "Thank you for the quick help": "You're most welcome! We're always here to assist you.",
    "I really appreciate your service": "We appreciate your feedback! Thank you for trusting us.",
    "Everything is working perfectly": "Great! We're happy everything is functioning smoothly for you.",

    # Customer Dissatisfaction / Complaints
    "I am not happy with your service": "I'm sorry to hear that. I will help you resolve the issue immediately.",
    "My order is delayed": "I apologize for the delay. Let me check your order status right away.",
    "I want a refund": "I understand your concern. I will help you with the refund process.",
    "My product is broken": "I'm really sorry to hear that. I will assist you with a replacement.",
    "Your support is very slow": "I apologize for the inconvenience. I will prioritize your request now.",
    "I can't login to my account": "I’m sorry about the trouble. Let me guide you through resolving the login issue.",
    "I was charged incorrectly": "I apologize for the confusion. I will check the billing details for you.",
    "This is the worst experience ever": "I'm very sorry you feel this way. I will personally help fix the issue.",
    "I am frustated": "I'm very sorry you feel this way. I will personally help fix the issue.",
    "This is fraud": "I'm very sorry you feel this way. I will personally help fix the issue.",
    "This is wrong": "I'm very sorry you feel this way. I will personally help fix the issue.",

    # Neutral / Clarification Questions
    "How do I contact customer care?": "You can contact us through live chat, email, or our toll-free number.",
    "Where can I track my order?": "You can track your order from your account dashboard under 'My Orders'.",
    "How long does delivery take?": "Delivery usually takes 3–5 business days depending on your location.",
    "How do I update my address?": "You can update your address from the account settings section.",
    "Do you offer discounts?": "Yes, we offer seasonal and promotional discounts throughout the year.",
}


# Precompute embeddings
kb_questions = list(knowledge_base.keys())
kb_embeddings = model.encode(kb_questions, convert_to_tensor=True)

def get_bot_reply(user_text):

    # Convert user query to embedding
    user_embedding = model.encode(user_text, convert_to_tensor=True)

    # Retrieve closest match
    scores = util.cos_sim(user_embedding, kb_embeddings)[0]

    best_match_id = scores.argmax().item()
    best_score = scores[best_match_id].item()

    # Threshold (to avoid wrong matches)
    if best_score < 0.4:
        return "I'm not completely sure about that, but I will help you! Could you rephrase your question?"

    # Return retrieved answer
    matched_question = kb_questions[best_match_id]
    return knowledge_base[matched_question]

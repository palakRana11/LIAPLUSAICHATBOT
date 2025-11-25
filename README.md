# Customer Support Chatbot with Sentiment Analysis

## Overview
This project implements a **customer support chatbot** using **semantic search** with `SentenceTransformer` and **sentiment analysis** using a custom-enhanced `VADER` model. The chatbot can respond to general customer queries, track sentiment trends, and provide an overall summary of conversation sentiment.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/palakRana11/LIAPLUSAICHATBOT
    ```
2. Install :
   ```bash
   pip install -r requirements.txt
    ```
3. Run the Streamlit app :
   ```bash
   streamlit run main_app.py
    ```
4. Interact with the chatbot in the browser.
---
## Chosen Technologies

1. Python 
2. Streamlit UI - for User Interface
3. Sentence Transformers - for semantic embeddings for KB-Search
4. VADER Sentiment Analysis- customized sentiment scoring and provides a compund score which is helpful to plot the mood trend during the whole conversation
5. Pandas and MatplotLib
 
---

## Explanation of the Sentiment Logic

The project uses **VADER Sentiment Analyzer** as the core model for detecting emotional tone in user messages.  
To improve accuracy in customer-service scenarios, a **custom lexicon** is injected into VADER — assigning stronger positive or negative weights to domain-specific terms.

###  Custom Lexicon Enhancements
- **Strong negative indicators:**  
  Words like *“refund”*, *“worst”*, *“broken”*, and *“frustated”* are assigned highly negative scores.
- **Positive indicators:**  
  Terms such as *“thank you”*, *“great”*, and *“resolved”* are weighted positively.

These custom additions help the model understand customer emotions more precisely in support-related conversations.

---

###  How Sentiment Is Classified
Each message receives a **compound sentiment score** ranging from **-1 to +1**.

| Compound Score Range | Sentiment Label |
|----------------------|-----------------|
| ≥ **0.05**           | **Positive**    |
| ≤ **-0.05**          | **Negative**    |
| Otherwise            | **Neutral**     |

This ensures quick and intuitive classification of user emotions.

---

### 🔹 Overall Conversation Sentiment
To summarize the entire interaction, the system computes the **average compound score** across all user messages:

- High average → overall positive conversation  
- Low average → overall negative experience  
- Middle range → neutral/mixed interaction

This provides a concise emotional overview of the user’s full session.
---

### Status of Tier 2 Implementation
Tier 2 (Statement-Level Sentiment Analysis) has been fully implemented.
Each user message is evaluated individually using the enhanced VADER-based logic, and its sentiment label is displayed alongside the message. The overall trend is plotted based on the compound scores returned by the VADER model which can be used to analyze the patterns in mood and behaviour of the customer. The system also computes the final conversation sentiment by averaging all individual sentiment scores.

---
## Tests
The project includes a small automated test suite using pytest to verify whether the tier 1 and tier 2 functions work correctly.

### Tests Summary:
- Message sentiment (Positive / Negative / Neutral)
- Detection of custom lexicon words (e.g., refund, worst, broken)
- Handling of mixed or neutral sentences
- Conversation-level sentiment summary using mocked data
- Behavior when no conversation history exists

# *Test File*
   ```bash
   tests.py
   ```
# *To run Tests*
   ```bash
   pytest tests.py
   ```


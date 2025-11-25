
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

from main.chatbot_model import get_bot_reply
from main.sentiment_model import analyze_sentiment
from main.storage import save_user_message, load_conversation, clear_conversation
from main.analyzer import final_conversation_sentiment

# --- PRE-PROCESS INPUT BEFORE WIDGETS RENDER ---
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

if "clear_box" in st.session_state and st.session_state["clear_box"]:
    st.session_state["user_input"] = ""
    st.session_state["clear_box"] = False
# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Customer Support Chatbot", layout="centered")

st.title("Customer Support Chatbot")

# -----------------------------------------------------------
# RESET BUTTON
# -----------------------------------------------------------
if st.button("Reset Conversation"):
    clear_conversation()
    st.rerun()

# -----------------------------------------------------------
# LOAD HISTORY
# -----------------------------------------------------------
history = load_conversation()

# -----------------------------------------------------------
# CUSTOM STYLING
# -----------------------------------------------------------
st.markdown("""
<style>
/* Remove top padding globally */
.main .block-container {
    padding-top: 0rem !important;
}



/* User bubble */
.user-bubble {
    background: #d6ebff;
    padding: 10px 14px;
    border-radius: 10px;
    max-width: 70%;
    margin-left: auto;
    margin-bottom: 15px;
    color: #000;
    text-align: right;
}

/* Bot bubble */
.bot-bubble {
    background: #e8e8e8;
    padding: 10px 14px;
    border-radius: 10px;
    max-width: 70%;
    margin-right: auto;
    margin-bottom: 15px;
    color: #000;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Remove margins from headings */
h1, h2, h3, h4, h5, h6 {
    margin-top: 0rem;
    margin-bottom: 0rem;
}

/* Reduce spacing below chat window */
.chat-window {
    margin-bottom: 0rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# CHAT WINDOW
# -----------------------------------------------------------
st.markdown("### Conversation")
st.markdown('<div class="chat-window">', unsafe_allow_html=True)

for msg in history:
    if msg["role"] == "user":
        st.markdown(
            f'''
            <div class="user-bubble">
                <strong>You:</strong><br>{msg["message"]}
                <br><small>Sentiment: <b>{msg["sentiment"]}</b> ({msg["score"]})</small>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'''
            <div class="bot-bubble">
                <strong>Bot:</strong><br>{msg["message"]}
            </div>
            ''',
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# -----------------------------------------------------------
# INPUT BOX — ENTER TO SEND
# -----------------------------------------------------------
if "enter_pressed" not in st.session_state:
    st.session_state["enter_pressed"] = False

def enter_callback():
    st.session_state["enter_pressed"] = True

user_input = st.text_input(
    "Type your message…",
    key="user_input",
    on_change=enter_callback
)

send_pressed = st.button("Send")

# -----------------------------------------------------------
# PROCESS MESSAGE (NO DOUBLE SEND)
# -----------------------------------------------------------

# Create a single unified trigger lock
if "trigger_lock" not in st.session_state:
    st.session_state.trigger_lock = False

# Check if user pressed enter or button
trigger = False

# ENTER key
if st.session_state["enter_pressed"] and not st.session_state.trigger_lock:
    trigger = True
    st.session_state["enter_pressed"] = False

# SEND button
if send_pressed and not st.session_state.trigger_lock:
    trigger = True

if trigger:
    st.session_state.trigger_lock = True  # prevent double send in this rerun

    text = st.session_state["user_input"].strip()
    if text:
        sentiment, score = analyze_sentiment(text)

        save_user_message(
            message=text,
            sentiment=sentiment,
            score=score,
            role="user"
        )

        bot_reply = get_bot_reply(
            "You are a polite, helpful customer-care assistant. "
            "Respond professionally. User said: " + text
        )

        save_user_message(
            message=bot_reply,
            sentiment="neutral",
            score=0,
            role="bot"
        )

        st.session_state["clear_box"] = True
        st.session_state.trigger_lock = False  # unlock for next message
        st.rerun()


# -----------------------------------------------------------
# SENTIMENT TREND GRAPH
# -----------------------------------------------------------
st.subheader("Sentiment Trend Across User Messages")

data = load_conversation()
if data:
    user_data = [d for d in data if d["role"] == "user"]
    df = pd.DataFrame(user_data)

    plt.figure(figsize=(8, 4))
    plt.plot(df["score"], linewidth=2, marker="o")
    plt.grid(alpha=0.3)
    plt.title("User Sentiment Trend (Compound Score)")
    plt.xlabel("Message Index")
    plt.ylabel("Sentiment Score")
    plt.ylim(-1, 1)

    st.pyplot(plt)

# -----------------------------------------------------------
# SUMMARY SENTIMENT
# -----------------------------------------------------------
if st.button("Show Summary Sentiment"):
    result = final_conversation_sentiment()
    st.write("### Final Conversation Analysis:")
    st.success(result)

import os
import requests
import streamlit as st

backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# ======================
# = Connect to Backend =
# ======================
def backend_predict(text, true_label):
    url = backend_url.rstrip("/") + "/predict"
    payload = {
        "text": text,
        "bought": true_label
    }
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()["predicted_bought"]


# ==============
# = App Layout =
# ==============
# 1. Title
st.title('Book Purchase Intention Analyzer')
# 2. Short description of what the app does.
st.text("This app accepts any book review the amazon \
         user inputs and intelligently returns the \
         prediction of the purchase record, positive or negative.")

st.text("The target is to show it is hard to predict users' \
         purchase behavior by their words.")

# 3. Create the User Input Interface
user_text = st.text_area("Enter a book review to analyze:",
                         "Enter text here...", height=200)
true_label = st.text_area("Enter a bought record:",
                          "Enter text here...", height=200)
# 4. Add analyze button & Write an if block that checks
#    if the "Analyze" button has been pressed
if st.button("Analyze"):
    # Make sure the user has entered some text
    # before trying to make a prediction.
    if not user_text:
        st.write("Please write review before analyzing.")
    else:
        # Load the model and class names
        pred = backend_predict(user_text, true_label)
        # pred = await backend_predict(user_text, true_label)
        st.subheader(f"Prediction: {pred}")
        st.subheader('Prediction Result:')
        if pred == true_label:
            st.success(f"Predicted Purchase Record: {pred} \U0001F44D")
        else:
            st.error(f"Predicted Purchase Record: {pred} \U0001F44E")
        st.subheader('Welcome to try again.')


# API_BASE = "http://localhost:8000"  
# PREDICT_URL = f"{API_BASE.rstrip('/')}/predict"

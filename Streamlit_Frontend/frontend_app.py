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
        "true_sentiment": true_label
    }
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()["sentiment"]


# ==============
# = App Layout =
# ==============
# 1. Title
st.title('Objective Movie Review Sentiment Analyzer')
# 2. Short description of what the app does.
st.text("This app accepts any movie review the \
         user inputs and intelligently returns the \
         sentiment of the words, positive or negative.")

# 3. Create the User Input Interface
user_text = st.text_area("Enter a movie review to analyze:",
                         "Enter text here...", height=200)
true_label = st.text_area("Enter a true sentiment:",
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
            st.success(f"Predicted Sentiment: {pred} \U0001F44D")
        else:
            st.error(f"Predicted Sentiment: {pred} \U0001F44E")
        st.subheader('Welcome to try again.')


# API_BASE = "http://localhost:8000"   # 或从环境变量读取
# PREDICT_URL = f"{API_BASE.rstrip('/')}/predict"

# session = requests.Session()

# def backend_predict(text, true_label, timeout=10):
#     payload = {"text": text, "true_sentiment": true_label}
#     try:
#         resp = session.post(PREDICT_URL, json=payload, timeout=timeout)
#         resp.raise_for_status()
#         data = resp.json()
#         # 返回完整数据，caller 决定如何使用
#         return data  # e.g. {"sentiment": "...", "cached": True/False}
#     except requests.exceptions.Timeout:
#         raise RuntimeError("Request to backend timed out")
#     except requests.exceptions.ConnectionError:
#         raise RuntimeError("Cannot connect to backend; is it running?")
#     except requests.exceptions.HTTPError as e:
#         # 服务器返回 4xx/5xx，尝试提取 detail
#         try:
#             err = resp.json().get("detail", resp.text)
#         except Exception:
#             err = resp.text
#         raise RuntimeError(f"Backend returned HTTP \
#                              {resp.status_code}: {err}") from e

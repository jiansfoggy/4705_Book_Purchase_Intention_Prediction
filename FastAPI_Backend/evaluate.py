import json
import os
import requests
from sklearn.metrics import accuracy_score

backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

def load_test_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_prediction(text, true_label, url="http://127.0.0.1:8000/predict"):
    payload = {
        "text": text,
        "bought": true_label
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()["predicted_bought"]


def main():
    test_data = load_test_data("./test_data.json")
    y_true = []
    y_pred = []
    url = backend_url.rstrip("/") + "/predict"
    cnt = 0
    for entry in test_data:
        text = entry["text"]
        true_label = entry["bought"]
        try:
            pred = get_prediction(text, true_label, url)
        except Exception as e:
            print(f"Error predicting for text '{text[:30]}...': {e}")
            continue

        y_true.append(true_label)
        y_pred.append(pred)
        cnt += 1
        if cnt%10000==0:
            print(f"Text: {text[:10]}... | True: {true_label} | Pred: {pred}")

    # compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy is {accuracy:.2%}.")


if __name__ == "__main__":
    main()

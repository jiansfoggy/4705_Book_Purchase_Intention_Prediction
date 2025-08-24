import json
import requests
from sklearn.metrics import accuracy_score


def load_test_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_prediction(text, true_label, url="http://localhost:8000/predict"):
    payload = {
        "text": text,
        "true_sentiment": true_label
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()["sentiment"]


def main():
    test_data = load_test_data("./test.json")
    y_true = []
    y_pred = []

    for entry in test_data:
        text = entry["text"]
        true_label = entry["true_label"].capitalize()
        try:
            pred = get_prediction(text, true_label)
        except Exception as e:
            print(f"Error predicting for text '{text[:30]}...': {e}")
            continue

        y_true.append(true_label)
        y_pred.append(pred)
        print(f"Text: {text[:50]}... | True: {true_label} | Pred: {pred}")

    # compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy is {accuracy:.2%}.")


if __name__ == "__main__":
    main()

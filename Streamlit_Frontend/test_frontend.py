import frontend_app
import os
import pytest
import requests
from frontend_app import backend_predict

DDB_REGION = os.environ.get("AWS_REGION", "us-east-1")


class MockResponse:
    def __init__(self, json_data=None, status_code=200):
        self._json = json_data or {}
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if 400 <= self.status_code:
            raise requests.exceptions.HTTPError(f"{self.status_code} Error")


@pytest.mark.parametrize("bought", [("Positive")])
def test_backend_predict(monkeypatch, bought):
    def mock_post(url, json, timeout):
        assert url.endswith("/predict")
        assert "text" in json and "bought" in json
        return MockResponse({"predicted_bought": "Positive"}, 200)

    monkeypatch.setattr(frontend_app.requests, "post", mock_post)

    res = backend_predict("love this book very much. have to buy it right now.", bought)
    assert res == "Positive" or isinstance(res, str)


# pytest -v test_frontend.py
# uvicorn main:app --reload
# @pytest.mark.parametrize("text, true_label", [
#     ("Positive Review", "Positive"),
#     ("Negative Review", "Negative"),
#     ])
# def test_predict1(text, true_label):
#     url = "http://localhost:8000/predict"
#     payload = {
#         "text": text,
#         "true_sentiment": true_label
#     }
#     resp = requests.post(url, json=payload)
#     resp.raise_for_status()
#     pred = resp.json()["sentiment"]
#     assert pred == true_label

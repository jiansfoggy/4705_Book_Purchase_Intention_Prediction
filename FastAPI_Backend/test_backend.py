import main
import pytest
import types
from botocore.exceptions import ClientError
from fastapi import HTTPException
from main import ensure_table, predict, TextInput


class FakeTable:
    def __init__(self, name, exists=True):
        self.table_name = name
        self._exists = exists
        self.table_status = "ACTIVE" if exists else None
        self._storage = {}

    def load(self):
        if not self._exists:
            raise ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
                "DescribeTable"
            )

    def get_item(self, Key):
        text_hash = Key["text_hash"]
        if text_hash in self._storage:
            return {"Item": self._storage[text_hash]}
        return {}

    def put_item(self, Item):
        self._storage[Item["text_hash"]] = Item
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}
        

class FakeResource:
    def __init__(self, existing_tables=None):
        self._existing = set(existing_tables or [])

    def Table(self, name):
        return FakeTable(name, exists=(name in self._existing))

    def create_table(self, **kwargs):
        # Simulate creation and return an object with meta.client.get_waiter('table_exists').wait
        name = kwargs.get("TableName")
        # add to existing set
        self._existing.add(name)
        class NewTable:
            def __init__(self, name):
                self.table_name = name
                self.table_status = "CREATING"
                class MetaClient:
                    def get_waiter(self, _):
                        class Waiter:
                            def wait(self, **kwargs):
                                # simulate wait -> table exists
                                return
                        return Waiter()
                self.meta = types.SimpleNamespace(client=MetaClient())
        return NewTable(name)


def test_ensure_table_existing(monkeypatch):
    table_name = "Backend_Log_Cache"
    fake_resource = FakeResource(existing_tables=[table_name])
    monkeypatch.setattr(main, "connect_dynamodb", lambda: fake_resource)

    tbl = main.ensure_table(table_name=table_name, create_if_missing=False)
    assert tbl is not None
    assert getattr(tbl, "table_name") == table_name
    assert tbl.table_status == "ACTIVE" or hasattr(tbl, "table_status")


def test_ensure_table_creates_if_missing(monkeypatch):
    table_name = "Backend_Log_Cache"
    fake_resource = FakeResource(existing_tables=[])
    monkeypatch.setattr(main, "connect_dynamodb", lambda: fake_resource)

    tbl = main.ensure_table(table_name=table_name, create_if_missing=True, wait_timeout=5)
    assert tbl is not None
    assert tbl.table_name == table_name


@pytest.mark.asyncio
@pytest.mark.parametrize("text, true_label", [
    ("Love this move so much. This is great.", "Positive"),
    ("So bad. So boring.", "Negative"),
    ])
async def test_predict2(monkeypatch, text, true_label):
    monkeypatch.setattr(main, "ensure_table", lambda *args, **kwargs: FakeTable("Backend_Log_Cache"))
    # fake_resource = FakeResource(existing_tables={"Backend_Log_Cache"})
    # monkeypatch.setattr(main.boto3, "resource", lambda *args, **kwargs: fake_resource)
    payload = TextInput(text=text, true_sentiment=true_label)
    pred = await predict(payload)
    assert pred["sentiment"] == true_label.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("text, true_label, stat_code, expected_detail", [
    ("!!!!", "_", 400, "True_label can only be either negative or positive."),
    (124234, " ", 422, "Text must be string."),
    ])
async def test_predict3(monkeypatch, text, true_label, stat_code, expected_detail):
    monkeypatch.setattr(main, "ensure_table", lambda *args, 
                        **kwargs: FakeTable("Backend_Log_Cache"))
    payload = TextInput.model_construct(
        text=text,
        true_sentiment=true_label,
        _fields_set={"text", "true_sentiment"}
    )
    with pytest.raises(main.HTTPException) as excinfo:
        await predict(payload)

    assert excinfo.value.status_code == stat_code
    assert excinfo.value.detail == expected_detail


# pytest -v test_backend.py
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

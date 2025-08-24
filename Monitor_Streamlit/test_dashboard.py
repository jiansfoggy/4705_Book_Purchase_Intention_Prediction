# tests/test_streamlit_launch.py
import monitor_app
import pytest
import types
from botocore.exceptions import ClientError
from monitor_app import main, ensure_table


class FakeTable:
    def __init__(self, name, exists=True):
        self.table_name = name
        self._exists = exists
        self.table_status = "ACTIVE" if exists else None
        self._storage = {}

    def load(self):
        if not self._exists:
            d1 = {"Code": "ResourceNotFoundException", "Message": "Not found"}
            raise ClientError({"Error": d1}, "DescribeTable")

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
        # Simulate creation and return an object with 
        # meta.client.get_waiter('table_exists').wait
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
    monkeypatch.setattr(monitor_app, "connect_dynamodb", lambda: fake_resource)

    tbl = ensure_table(table_name=table_name, create_if_missing=False)
    assert tbl is not None
    assert getattr(tbl, "table_name") == table_name
    assert tbl.table_status == "ACTIVE" or hasattr(tbl, "table_status")


def test_ensure_table_creates_if_missing(monkeypatch):
    table_name = "Backend_Log_Cache"
    fake_resource = FakeResource(existing_tables=[])
    monkeypatch.setattr(monitor_app, "connect_dynamodb", lambda: fake_resource)

    tbl = ensure_table(
        table_name=table_name, 
        create_if_missing=True, 
        wait_timeout=5)
    assert tbl is not None
    assert tbl.table_name == table_name


def test_main(monkeypatch):
    fake_resource = FakeResource(existing_tables=["Backend_Log_Cache"])
    monkeypatch.setattr(monitor_app, "connect_dynamodb", lambda: fake_resource)

    try:
        main()
    except Exception as exc:
        pytest.fail(f"Calling main() raised an exception: {exc}")

# pytest -v test_dashboard.py

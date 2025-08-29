import pandas as pd
# import pytest
import train_model


# Dummy classes to capture calls inside log_artifact
class DummyArtifact:
    def __init__(self, name, type, metadata=None):
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.files = []

    def add_file(self, path):
        self.files.append(path)

    def wait(self):
        # no-op for tests
        pass


class DummyRun:
    def __init__(self):
        self.logged_artifacts = []
        self.linked_models = []
        self.summary = {}
        self.config = {}

    def log_artifact(self, artifact):
        self.logged_artifacts.append(artifact)

    def link_model(self, path, registered_model_name, aliases):
        self.linked_models.append((path, registered_model_name, aliases))

    def finish(self):
        # no-op for tests
        pass

    def log(self, data):
        # if code under test calls wandb.log
        self.last_log = data


def test_log_artifact(tmp_path, monkeypatch):
    # Prepare dummy run + monkeypatch Artifact factory
    run = DummyRun()
    monkeypatch.setattr(train_model.wandb, "Artifact", DummyArtifact)

    # Create dummy files
    data_file = tmp_path / "dummy_data.csv"
    model_file = tmp_path / "dummy_model.pkl"
    data_file.write_text("a,b,c\n1,2,3")
    model_file.write_text("fake")

    # Call log_artifact
    dataset_name = "my_ds"
    model_name = "my_model"
    alias = "v123"
    data_art, model_art = train_model.log_artifact(
        run=run,
        data_path=str(data_file),
        model_path=str(model_file),
        dataset_name=dataset_name,
        model_name=model_name,
        alias=alias,
        metadata={"foo": "bar"},
    )

    # Two artifacts created
    assert isinstance(data_art, DummyArtifact)
    assert isinstance(model_art, DummyArtifact)

    # They each have exactly one file
    assert data_art.files == [str(data_file)]
    assert model_art.files == [str(model_file)]

    # run.log_artifact called for both
    assert run.logged_artifacts == [data_art, model_art]

    # run.link_model called once, with model artifact details
    assert run.linked_models == [
        (
            str(model_file),
            f"{model_name}-artifact",
            [alias],
        )
    ]

    # Check metadata propagated
    assert data_art.metadata == {"foo": "bar"}
    assert model_art.metadata == {"foo": "bar"}


def test_main(tmp_path, monkeypatch):
    # chdir into tmp_path so that relative paths resolve there
    monkeypatch.chdir(tmp_path)

    # 1) Stub wandb.login
    monkeypatch.setattr(train_model.wandb, "login", lambda: None)

    # 2) Stub git hash
    monkeypatch.setattr(train_model, "get_git_commit_hash", lambda: "deadbeef")

    # 3) Stub data_load and split_XY
    sample_df = pd.DataFrame({
        "text": ["good book, must buy", "boring noval. drop it."],
        "bought": ["Positive", "Negative"]
    })
    monkeypatch.setattr(train_model, "data_load", lambda path: sample_df)
    monkeypatch.setattr(
        train_model,
        "split_XY",
        lambda df: (
            df["text"],
            df["bought"].map({"Positive": 1, "Negative": 0})
        )
    )

    # 4) Stub create_pipeline to write a dummy checkpoint
    ckpt_path = tmp_path / "purchase_model.pkl"

    def fake_create_pipeline(X, y, ckpt_path_arg):
        # simulate saving model
        with open(ckpt_path_arg, "w") as f:
            f.write("fake-model-bytes")
    monkeypatch.setattr(train_model, "create_pipeline", fake_create_pipeline)

    # 5) Stub log_artifact to capture its inputs and return two dummy artifacts
    called = {}

    class Art:
        def __init__(self, name):
            self.name = name
            self.aliases = []

        def wait(self):
            pass

    def fake_log_artifact(run, data_path, model_path, dataset_name,
                          model_name, alias, metadata=None):
        called["data_path"] = data_path
        called["model_path"] = model_path
        called["dataset_name"] = dataset_name
        called["model_name"] = model_name
        called["alias"] = alias
        # return two dummy artifacts
        return Art("ds-art"), Art("model-art")

    monkeypatch.setattr(train_model, "log_artifact", fake_log_artifact)

    # 6) Stub init_wandb to return our DummyRun
    run = DummyRun()
    monkeypatch.setattr(train_model, "init_wandb", lambda **kwargs: run)

    # 7) Stub wandb.finish (so main can call it)
    monkeypatch.setattr(train_model.wandb, "finish", lambda: None)

    # Execute main()
    train_model.main()

    # Validate that create_pipeline wrote the file
    assert ckpt_path.exists()

    # Validate that log_artifact was called with correct relative paths
    assert called["data_path"].endswith("review_data.csv")
    assert called["model_path"].endswith("purchase_model.pkl")
    assert called["dataset_name"] == "Amazon_Review_2023"
    assert called["model_name"] == "MultinomialNB"
    assert called["alias"] == "staging"

    # Validate that run.summary was populated
    assert run.summary["model_registered_name"] == "model-art"
    assert run.summary["registered_aliases"] == "staging"
    assert run.summary["git_commit"] == "deadbeef"
    assert run.summary["data_artifact"] == "ds-art"

# pytest -v test_manage.py

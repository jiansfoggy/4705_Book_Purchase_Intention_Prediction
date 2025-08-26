import joblib
import os
import subprocess
import wandb
import warnings
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
warnings.filterwarnings('ignore')


# ==============
# = Init WandB =
# ==============
def get_git_commit_hash():
    # Capture Code Version from Git Commit
    # Return short git commit hash
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        commit_hash = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        commit_hash = commit_hash.decode("utf-8").strip()
        return commit_hash
    except Exception:
        return "unknown"


def init_wandb(project_name="Book_Purchase_Intention_Prediction",
               experiment_name=None, entity=None, config=None,
               save_code=True):

    if entity is None:
        entity = os.environ.get("WANDB_ENTITY", None)

    # Initialize a new W&B run
    run = wandb.init(
        project=project_name, name=experiment_name,
        entity=entity, config=config, reinit=True)

    if save_code:
        try:
            # log_code will capture python files in repo as an artifact
            run.log_code(".")
        except Exception as e:
            # best-effort: continue even if code snapshot fails
            run.log({"_code_logging_error": str(e)})
    return run


# =================
# = Data Load and =
# = Preprocess    =
# =================
def log_artifact(run, data_path, model_path, dataset_name="Amazon Review 2023",
                 model_name="NB", alias="staging", metadata=None):
    # Create data artifact
    artifact_data = wandb.Artifact(
        name=f"{dataset_name}-artifact",
        type="dataset", metadata=metadata or {})
    artifact_data.add_file(data_path)  # add the csv file into artifact
    run.log_artifact(artifact_data)

    # Create model artifact
    artifact_model = wandb.Artifact(
        name=f"{model_name}-artifact",
        type="model", metadata=metadata or {})
    artifact_model.add_file(model_path)
    run.log_artifact(artifact_model)
    run.link_model(path=model_path,
                   registered_model_name=f"{model_name}-artifact",
                   aliases=[alias])

    # artifact_model = run.link_model(path=model_path,
    #                                 registered_model_name=f"{model_name}-artifact",
    #                                 aliases=[alias])

    return artifact_data, artifact_model


def data_load(file_path):
    df = pd.read_csv(file_path)
    print(df.info())
    print(df.head(3))
    return df


def split_XY(dataset):
    X = dataset.text
    y = dataset.bought.map({'Positive': 1, 'Negative': 0})
    return X, y


# ==============
# = Model and  =
# = Train Func =
# ==============
def create_pipeline(X, y, ckpt_path):
    # Log dataset info to W&B
    wandb.log({
        "n_samples": len(X)
    })
    model = Pipeline([
           ('tfidf', TfidfVectorizer()),
           ('clf', MultinomialNB())
           ])
    model.fit(X, y)
    joblib.dump(model, ckpt_path)
    print("Pretrained weight is saved.")


# =================
# = Main Workflow =
# =================
def main():
    # Prepare info for WandB
    entity = "jsfoggy"
    git_hash = get_git_commit_hash()
    dataset_version = "v1"
    config = {
        "git_commit": git_hash,  # code version
        "dataset": "Amazon Review 2023 -- Book Subset",
        "data version": dataset_version,
        "model_name": "MultinomialNB",
        "test_size": 0.2,
        "random_state": 42,
        "alpha": 1.0,
        "max_iter": 1000,
        "X_train": None,
        "y_train": None}
    # promote_to_production_threshold = 0.9

    # Load data and run model
    file_path = '../data/review_data.csv'
    ckpt_path = './purchase_model.pkl'
    movie_reviews = data_load(file_path)
    X, y = split_XY(movie_reviews)
    models = {
        "MultinomialNB": lambda config: "create_pipeline(X,y)"
    }
    # remember to log performance metrics (e.g., accuracy, F1-score)
    for model_name, model_func in models.items():
        # Initialize W&B for this specific model
        run = init_wandb(project_name="Book_Purchase_Intention_Prediction",
                         experiment_name=f"{model_name}-Exp",
                         entity=entity, config=config, save_code=True)

        run.config.update({"model_name": model_name})

        create_pipeline(X, y, ckpt_path)
        # Promote to Staging or Production
        # aliases = ["latest", "staging"]
        # if metrics["accuracy"] >= promote_to_production_threshold:
        #     aliases = ["latest", "production"]
        artifact_data, artifact_model = log_artifact(
            run, data_path=file_path, model_path=ckpt_path,
            dataset_name="Amazon_Review_2023", model_name=run.config["model_name"],
            alias="staging", metadata=None)

        # Promote to Staging or Production
        artifact_data.wait()
        artifact_data.aliases.append("staging")
        artifact_model.wait()
        # use "production" or aliases when we officially run model
        artifact_model.aliases.append("staging")
        print("Model registered and promoted to 'staging'.")

        run.summary["model_registered_name"] = f"{artifact_model.name}"
        run.summary["registered_aliases"] = "staging"  # aliases
        run.summary["git_commit"] = git_hash
        run.summary["data_artifact"] = f"{artifact_data.name}"

        wandb.finish()
        print(f"Experiment for {model_name} completed!\n")


# =================
# = Start Program =
# =================
if __name__ == '__main__':
    wandb.login()
    main()

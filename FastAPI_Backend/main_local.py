import boto3
import hashlib
import joblib
import json
import os
import requests
import time
import wandb
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from decimal import Decimal
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
# from typing import List

DDB_TABLE_NAME = os.environ.get("DDB_TABLE", "Backend_Log_Cache")
DDB_REGION = os.environ.get("AWS_REGION", "us-east-1")
os.makedirs("./logs", exist_ok=True)


# ================================
# = Check Running Environment:   =
# = Local Machine or AWS Lab EC2 =
# ================================
def is_ec2_env():
    try:
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/",
            timeout=0.2)
        return response.status_code == 200
    except requests.RequestException:
        return False


# ==============
# = Set up AWS =
# = DynamoDB   =
# ==============
# get temporary permit
def connect_dynamodb():
    # figure out the current environment: local or AWS learner lab EC2
    # connect to DynamoDB based on different environment
    if is_ec2_env():
        # boto3 will pick up credentials from env or ~/.aws/credentials automatically
        print("Detected EC2 (Learner Lab). Using IAM Role credentials...")
        dynamodb = boto3.resource("dynamodb", region_name=DDB_REGION)
    else:
        print("Detected local environment. Using AWS credentials from env/config...")
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_token = os.getenv("AWS_SESSION_TOKEN")
        session = boto3.Session(
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
                aws_session_token=aws_token,
                region_name=DDB_REGION)
        # dynamodb = session.resource("dynamodb")
        # dynamodb = boto3.resource("dynamodb", region_name="us-east-1",
        #                           aws_access_key_id=aws_key,
        #                           aws_secret_access_key=aws_secret,
        #                           aws_session_token=aws_token)
    return session.resource("dynamodb")


def ensure_table(table_name=DDB_TABLE_NAME, create_if_missing=True,
                 wait_timeout=60):

    dynamodb = connect_dynamodb()
    table = dynamodb.Table(table_name)

    try:
        table.load()
        print(f"[DDB] Table '{table_name}' found.")
        return table
    except ClientError as e:
        err_code = e.response.get("Error", {}).get("Code", "")
        if err_code not in ("ResourceNotFoundException", "ValidationException"):
            raise

        # load the existing DynamoDB table, otherwise, create new DynamoDB table
        print(f"[DDB] Table '{table_name}' not found - creating...") 
        try:
            new_table = dynamodb.create_table(
                    TableName=table_name,
                    AttributeDefinitions=[{"AttributeName": "text_hash", "AttributeType": "S"}],
                    KeySchema=[{"AttributeName": "text_hash", "KeyType": "HASH"}],
                    BillingMode="PAY_PER_REQUEST",
                    Tags=[{"Key":"final_project","Value":"API_logs"}])
        except (ClientError, NoCredentialsError, EndpointConnectionError) as create_err:
                print(f"[DDB] Failed to create table: {create_err}")
                raise

        new_table.meta.client.get_waiter('table_exists').wait(TableName=table_name,
            WaiterConfig={'Delay': 3, 'MaxAttempts': max(1, wait_timeout // 3)})
        table = dynamodb.Table(table_name)
        print(f"[DDB] Created table {table_name}")
        return table


def query_dynamodb_cache(text: str, table=None):
    """Return stored item or None. Item contains predicted_sentiment and true_sentiment etc."""
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    resp = table.get_item(Key={"text_hash": text_hash})
    if resp:
        return resp.get("Item")
    else:
        print(f"[DDB] get_item error: {e}")
        return None
    

def log_cache(text,pred,true_label,table):
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    ts = time.time()
    data = {
            "timestamp": ts,
            "request_text": text,
            "text_hash": text_hash,
            "predicted_sentiment": pred,
            "true_sentiment": true_label,
            "model_name": "MultinomialNB-artifact",
            "model_alias": "production"}
    with open("./logs/prediction_logs.json", "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")
    print(f"Create local log file at ./logs/prediction_logs.json")
    data["timestamp"]=Decimal(str(ts))
    try:
        table.put_item(Item=data)
        print(f"[DDB] put succeed: Cache data to DynamoDB")
    except ClientError as e:
        print(f"[DDB] put failed for: {text_hash} error: {e}")


# =======================
# = Set up FastAPI and  =
# = Load Model Artifact =
# =======================
def load_model_from_wandb(model_name="MultinomialNB-artifact", alias="latest"):
    """
    Load Weights & Biases Model Registry.
    Args:
        model_name: Registered model name in W&B
        alias: Alias or version, e.g. "production", "staging", "v1"
    """
    # method 1
    api = wandb.Api()
    # method 2
    # run = wandb.init(project="Personalized Book Recommender", entity="jsfoggy", job_type="inference")
    try:
        # Pull certain version from Model Registry
        # and Download to local path
        # method 1
        art = api.artifact(f"jsfoggy/Personalized Book Recommender/{model_name}:{alias}")
        artifact=art.get_path("sentiment_model.pkl").download()
        # method 2
        # artifact = run.use_model(name=f"jsfoggy/Personalized Book Recommender/{model_name}:{alias}") 
        model = joblib.load(artifact)
        print(f"Model '{model_name}:{alias}' loaded successfully from W&B.")
        return model

    except Exception as e:
        print(f"Could not load model from W&B: {e}")
        local_paths = [
            "../Model_Management/sentiment_model.pkl",
            "./Model_Management/sentiment_model.pkl",
            "./sentiment_model.pkl"
        ]
        for path in local_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                print(f"Model loaded locally from {path}")
                return model
        raise FileNotFoundError("No model found locally or in W&B Registry.")


class TextInput(BaseModel):
    text: str = Field(..., example="I loved this movie!")
    true_sentiment: str = Field(..., example="Positive")
    # text: str
    # true_sentiment: str

# ====================
# = Predict Endpoint =
# ====================
app = FastAPI(
    title="Personalized Book Recommender",
)

@app.get("/health")
def health():
    """
    Health Check Endpoint
    This endpoint is used to verify that the
    API server is running and responsive.
    It's a common practice for monitoring services.
    """
    return {"status": "ok"}


@app.post("/predict")
#async 
def predict(input_data: TextInput):
    """
    Prediction Endpoint
    Takes a feature vector and returns a binary prediction (0 or 1).
    """

    text_val = input_data.text
    if text_val is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be missing."
            )

    if not isinstance(text_val, str):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text must be string."
            )

    text = text_val.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty."
            )

    true_label_val = input_data.true_sentiment
    if true_label_val is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="True_label cannot be missing."
            )

    if not isinstance(true_label_val, str):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="True_label must be string."
            )

    true_label = true_label_val.strip().lower()
    if true_label not in ["negative", "positive"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="True_label can only be either negative or positive."
            )

    model = load_model_from_wandb(model_name="MultinomialNB-artifact", alias="latest")
    
    # 1) After getting text, check if it is already cached in the DynamoDB.
    table = ensure_table(create_if_missing=True)  # set True if you want code to auto-create table
    print("Table status:", table.table_status)
    item = query_dynamodb_cache(text, table=table)

    if item:
        # Cache hit: return the stored predicted sentiment
        # item may store predicted_sentiment as string
        pred = item.get("predicted_sentiment")
        return {"sentiment": pred, "cached": True}
    
    # 2) Not found in DB => do prediction
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Cannot make predictions."
        )

    category = ["negative", "positive"]
    prediction = model.predict([text])[0]
    pred = category[int(prediction)]
    log_cache(text, pred, true_label, table)

    return {"sentiment": pred}


if __name__ == "__main__":
    input_data=TextInput(text="my family love it!! I am willing to watch again.", true_sentiment="Positive")
    # input_data.text = "What a lovely story!!"
    # input_data.true_sentiment = "Positive"
    # pred = predict(input_data)
    try:
        pred = predict(input_data)
        print("Sample result:", pred)
    except Exception as e:
        print("Error during sample prediction:", e)
        # helpful debugging info: print environment and credential presence
        print("is_ec2_env:", is_ec2_env())
        print("AWS env vars:", {
            "AWS_ACCESS_KEY_ID": bool(os.getenv("AWS_ACCESS_KEY_ID")),
            "AWS_SECRET_ACCESS_KEY": bool(os.getenv("AWS_SECRET_ACCESS_KEY")),
            "AWS_SESSION_TOKEN": bool(os.getenv("AWS_SESSION_TOKEN"))
        })


# uvicorn main:app --reload
# curl 'http://127.0.0.1:8000/health'
# POST http://localhost:8000/predict?text=This%20movie%20was
# %20a%20masterpiece!&true_sentiment=Positive

# Book Purchase Intention Prediction

## Phase 2: Backend API and Database Integration

If you test the program locally, please use `main_local.py`.

If you run it on EC2, please use `main.py`.

First, let's deug by running the program locally.

## 2.1 Initial and Config Amazon DynamoDB

If we run program locally, please do the following steps to get temporary permit for couple hours.

If we run program on **AWS EC2** build from **AWS Academy Learner Lab**, please skip, since AWS will automatically tackle permit issue.

1. Launch **AWS Academy Learner Lab** and click **AWS Details** on the top right.
2. Under **Cloud Access**, click **Show**, and record `aws_access_key_id`
, `aws_secret_access_key`, `aws_session_token`, `AWSAccountId`, and `Region`.
3. Open **terminal** and install `boto3`, `awscli`, `hashlib`.
   
    ```bash
    python3 -m pip install --upgrade pip
    python3 -m pip install boto3 awscli hashlib
    ```

4. In the terminal, run `aws configure` and enter copied `aws_access_key_id`, `aws_secret_access_key`, `Region`, and `json`.

    If you need to delete the old or wrong configuration info, run these lines:

    ```bash
    # 1. Run aws configure to set up permit
    aws configure
    # 2. Clear or Rename Existing Config, when input wrong info
    mv ~/.aws ~/.aws.backup
    rm ~/.aws/credentials ~/.aws/config
    # 2. Rerun aws configure
    aws configure
    # 3. Must add KEY_ID, ACCESS_KEY, and SESSION_TOKEN again
    #    to make sure everything is updated correctly.
    export AWS_ACCESS_KEY_ID="<KEY_ID from AWS Details>"
    export AWS_SECRET_ACCESS_KEY="<ACCESS_KEY from AWS Details>"
    export AWS_SESSION_TOKEN="<SESSION_TOKEN from AWS Details>"
    # 4. Verify new configuration works
    aws configure list
    aws sts get-caller-identity
    ```

    If you see the following feedback, you successfully set up a valid permit.
    ```bash 
    (venv) jiansun@Mac FastAPI_Backend % aws configure list
          Name                    Value             Type    Location
          ----                    -----             ----    --------
       profile                <not set>             None    None
    access_key     ****************4VZT              env    
    secret_key     ****************O8Lb              env    
        region                us-east-1      config-file    ~/.aws/config
    (venv) jiansun@Mac FastAPI_Backend % aws sts get-caller-identity
    {
        "UserId": "AROAZY5ZX5V7X2JNOIQWR:user4226724=jian",
        "Account": "672015052159",
        "Arn": "arn:aws:sts::672015052159:assumed-role/voclabs/user4226724=jian"
    }
    ```

5. Now, Amazon DynamoDB is ready. It is good for code to create new tables.

## 2.2 Build Backend FastAPI

To build FastAPI, there are 3 essential files: `main.py`, `Dockerfile`, `Makefile`.

`main.py` is for couple tasks:

| Task List                                                | 
| :------------------------------------------------------- |
| Check if queried text is in good format                  | 
| Connect to DynamoDB                                      | 
| Create a new table or Load the existing one              | 
| Return the cached prediction if the text is in the table | 
| Extract model artifact from WandB Model Registry         |
| Apply model to predict                                   | 
| Push the prediction into the DynamoDB as new record      |
| Save the prediction into the local json file             | 

`Dockerfile` is to config docker environment to run API.

`Makefile` integrates docker commands to create, launch, and delete docker image and container.

## 2.3 Query Endpoints Locally

### 2.3.1 Quickly Debug

- In the `./FastAPI_Backend`, run `python3 main_local.py` to debug. If no error pops out and it returns the prediction result, it means the code is no bug.

    ```bash 
    (venv) jiansun@Mac FastAPI_Backend % python3 main_local.py
    wandb: Currently logged in as: jsfoggy to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
    wandb: WARNING Artifact.get_path(name) is deprecated, use Artifact.get_entry(name) instead.
    Model 'MultinomialNB-artifact:latest' loaded successfully from W&B.
    Detected local environment. Using AWS credentials from env/config...
    [DDB] Table 'Backend_Log_Cache' found.
    Table status: ACTIVE
    Model 'MultinomialNB-artifact:latest' loaded successfully from W&B.
    Detected local environment. Using AWS credentials from env/config...
    [DDB] Table 'Backend_Log_Cache' found.
    Table status: ACTIVE
    Create local log file at ./logs/prediction_logs.json
    [DDB] put succeed: Cache data to DynamoDB
    Sample result: {'sentiment': 'positive'}
    ```

### 2.3.2 Deploy to Docker Environment

- Given that the program is deployed in the Docker environment, it can't read `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc from local machine. We need to take these information to Docker environment while building it.
   
    1. Enter `./FastAPI_Backend` and create a `.env` file

        ```
        WANDB_API_KEY="<paste from wandb>"
        AWS_ACCESS_KEY_ID="<paste from AWS Details>"
        AWS_SECRET_ACCESS_KEY="<paste from AWS Details>"
        AWS_SESSION_TOKEN="<paste from AWS Details>"
        AWS_REGION="<paste from AWS Details>"
        ```
    2. Add `.env` in the `docker run` command.
        
        ```bash
        docker run --env-file .env -d --name $(CONTAINER_NAME) -p 8000:8000 $(IMAGE_NAME)
        ```
- Enter `./FastAPI_Backend` and set up Docker container by `make` command.

    ```bash 
    cd ./FastAPI_Backend
    make build
    make run
    # enter ./FastAPI_Backend at a new terminal window and
    # run uvicorn, this can activate FastAPI too.
    python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

    Once running, FastAPI automatically generates its documentation. Explore and test all endpoints via Swagger UI at:

    ```
    http://127.0.0.1:8000/docs
    http://0.0.0.0:8000/docs
    ```

    This shows if you build API successfully.
- To check error, if you need to debug the Dockerfile and Makefile

    ```bash 
    # 1. check container status
    docker ps --filter "name=<container name>"
    # 2. check startup logs, this can show you the IP of local host
    #    that you can visit.
    docker logs <container name>
    ```
- To delete image and container

    ```bash 
    make clean
    ```

### 2.3.3 Test the Endpoint

To check the status of endpoint, please run

```bash
# Check health
curl http://127.0.0.1:8000/health

# Predict sentiment
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"text":"Four Stars. compelling read","bought":"Negative"}' \
     http://127.0.0.1:8000/predict

curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"text":"Five Stars. Gift for my mom that has cancer and she loves it!","bought":"Positive"}' \
     http://127.0.0.1:8000/predict
```

Features & Endpoints

Test the follow command in the Postman

### **1. `GET /health`**
- **Purpose**: Health check to ensure the API is running.
- **Response**: `{ "status": "ok" }`

### **2. `POST /predict`**
- **Purpose**: Classify input text as Positive or Negative.

* **Running Example**: http://127.0.0.1:8000/predict?text=This%20movie%20was%20a%20masterpiece!&true_sentiment=Positive.

- **Request Body**:
  ```json
  {
    "text": "Four Stars. compelling read",
    "bought":"Negative"
  }
  ```
* **Successful Response**:

  ```json
  {
    "predicted_bought": "Negative"
  }
  ```
* **Error Cases**:

  * `400 Bad Request` if the text is empty.
  * `503 Service Unavailable` if the model fails to load.

## 2.4 Check Cache in Amazon DynamoDB

1. Launch **AWS Academy Learner Lab**, click **Start Lab**, and click **AWS** on the left corner when light turns green.
2. Type in DynamoDB in the Search bar and choose it.
3. Then, the DynamoDB Dashboard pops up.
4. In the sidebar, click **Tables** to check all the table list, click **Explore items** to view table content.
5. If the `FastAPI_Backend` gets built successfully, you can find caches here.

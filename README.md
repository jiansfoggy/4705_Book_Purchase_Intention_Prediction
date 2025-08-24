# Personalized Book Recommender

## Introduction

A pipeline tests the program and application and deliveries and deploys the model to Github, which contains two services run independently but communicate with each other and their Docker build file. The entire process will be deployed on AWS EC2:

A lightweight **FastAPI**-based service that classifies text as **Positive** or **Negative**, and provides sentiment probabilities.

A **Streamlit App** that reads the logs from the shared volume to visualize model performance.

A **Docker** Volume that persist log data and share it between the two containers.

A **.github/workflows** that tests the code quality and program's bug while uploading to Github and pulling repository from `dev` branch to `master` one.

A **AWS EC2** server running Linux environment to deploy services like FastAPI backend and Streamlit monitor dashboard.

### **Project Architecture**

- `data`

- `Model_Management`: contains python file to train machine learning for book recommandation, and model weight file.

- `Monitor_Streamlit`: contains files to build streamlit

- `Prediction_FastAPI`: contains files to build FastAPI

- `Makefile`: builds multi-containers for this application

- ` CI/CD Pipeline`: `.github/workflows/ci.yml` lists workflows and points to check for automate code quality checking while pulling request to `master` branch.

- `README.md`: introduces the entire project and displays how to run the project.

---

## Prerequisites

To run this app, please make sure `Docker`, `Git`, `WandB`, `FastAPI`, `Postman`, `pytest`, `AWS Sandbox`, other essential softwares, and other essential python packages mentioned in the `requirements.txt` are installed. 

Turn on **AWS Learner Club** at step 1.

---

## Phase 1: Experimentation and Model Management

1. run `train_model.py` to start training.

    ```bash
    python3 train_model.py
    ```

2. The model weight is saved in the current directory.

3. **Weights & Biases** (WandB) helps log all essential information, like Git Commit, hyperparameters, performance metrics, and data version.

   Please click here to check the previous records.

4. WandB also helps save 3 artifacts such as **dataset** artifact, **model** one, and **code** one via the following functions.

    ```python
    # 1.Create code artifact
    if save_code:
        try:
            # log_code will capture python files in repo as an artifact
            run.log_code(".")
        except Exception as e:
            # best-effort: continue even if code snapshot fails
            run.log({"_code_logging_error": str(e)})

    # 2.Create data artifact
    artifact_data = wandb.Artifact(name=f"{dataset_name}-artifact", 
                                   type="dataset", metadata=metadata or {})
    artifact_data.add_file(data_path)  # add the csv file into artifact
    run.log_artifact(artifact_data)
    
    # 3.Create model artifact
    artifact_model = wandb.Artifact(name=f"{model_name}-artifact", type="model", metadata=metadata or {})
    artifact_model.add_file(model_path)
    run.log_artifact(artifact_model)
    ```

5. Then, the following code finishes **Model Registry** and promote best-performing model to a "Staging".

    ```python
    # Link to Model Registry
    run.link_model(path=model_path, 
                   registered_model_name=f"{model_name}-artifact", 
                   aliases=[alias])

    # Promote to Staging or Production
    aliases = ["latest", "staging"]
    if metrics["accuracy"] >= 0.95:
        aliases = ["latest", "production"]
    artifact_data.wait()
    artifact_data.aliases.append(aliases) 
    artifact_model.wait()
    artifact_model.aliases.append(aliases)
    ```

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
     -d '{"text":"What a lovely story!!","true_sentiment":"Positive"}' \
     http://0.0.0.0:8000/predict

curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"text":"Bad weather, bad movie, bad day!!Feel bad","true_sentiment":"Negative"}' \
     http://0.0.0.0:8000/predict
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
    "text": "I love this!",
    "true_sentiment":"Positive"
  }
  ```
* **Successful Response**:

  ```json
  {
    "sentiment": "Positive"
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

## Phase 3: Frontend and Live Monitoring

Streamlit helps build two services: Frontend User Interface and Model Monitoring Dashboard.

## 3.1: Streamlit Frontend -- User Interface

This section allows the user to enter query information and ground truth and send it to FastAPI backend and fetch the prediction and display it.

Here are the steps to test the program in the local machine.

1. Get the url of backend FastAPI by the following code

    ```python
    backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    ```

2. Apply `.post()` to send signal to backend and fetch result.

    ```python
    resp = requests.post(url, json=payload, timeout=10)
    ```

3. Feedback if the prediction is the same as ground truth value.

4. Create `Dockerfile` and set port number as `8501`.

5. Create `Makefile` under path `./4705_Personalized_Book_Recommender` like we did in **Assignment 6**. This `Makefile` builds two docker containers in detach mode.

6. Add FastAPU url as environment variable while using docker run. Since the exact url is unknown, we use this format `http://$(CONTAINER_BACKEND):8000` as url. `$(CONTAINER_BACKEND)` is the container name.

    ```bash
    docker run -d \
       --name $(CONTAINER_FRONTEND) \
       --network $(NETWORK) \
       -p 8501:8501 \
       -e BACKEND_URL=http://$(CONTAINER_BACKEND):8000 \
       -v $(shell pwd)/$(FRONTEND_DIR):/app \
       -v $(VOLUME):/app/logs \
       $(IMAGE_FRONTEND)
    ```

7. Activate Frontend and Backend services by the following code.

    ```bash
    cd ./4705_Personalized_Book_Recommender
    make build
    make run
    ```

    To check the real url for two services, we run this
    
    ```bash 
    make logs
    ```

    Visit two urls, if everything runs well without errors, our program is ready to go.

## 3.2: Streamlit Monitor -- Model Monitoring Dashboard

This section connects Amazon DynamoDB first, then extracts target variables from the loaded table for the final visualization.

Here are the steps to test the program in the local machine.

1. Connect to **Amazon DynamoDB** by the same way declared in **Phase 2: Backend API and Database Integration**. And load the table `Backend_Log_Cache`.

2. Apply `table.scan()` to scan all the table, use `.get('Items')` to take all data, implement `for loop` to save variables line by line in the lists.

3. Finish visualization like we did in **Assignment 6**.

4. Deploy Docker container and activate localhost to check if we develop all program successfully. 

    ```bash
    cd ./Monitor_Streamlit
    make build
    make run
    ```

5. Attention! : Add `.env` into `docker run` again in the `Makefile`

    ```bash
    docker run --env-file .env -d \
          --name $(CONTAINER_NAME) \
          -p 8501:8501 $(IMAGE_NAME)
    ```

    - Here, the port number `8501` is the same as that in Section 3.1 because we deploy the entire program on the separate EC2 Servers, which have different url. Therefore, identical port number doesn't cause confliction.

## Phase 4: Testing and CI/CD Automation

- Install `pytest` packages

  ```bash
  python3 -m pip install --upgrade pip
  python3 -m pip install pytest pytest-asyncio
  ```

## 4.1. Comprehensive Testing

There are individual test file in the four folders: `FastAPI_Backend`, `Model_Management`, `Monitor_Streamlit`, `Streamlit_Frontend`.

**Unit Tests**

`test_backend.py` in the `FastAPI_Backend`: 

1. `test_ensure_table_existing` tests if `ensure_table` function in the program can load existed table well.

2. `test_ensure_table_creates_if_missing` tests if `ensure_table` function creates new table well.

`test_manage.py` in the `Model_Management`:

3. `test_log_artifact` tests if `log_artifact` function in the program can create wandb artifact for data and model well.

`test_dashboard.py` in the `Monitor_Streamlit`:

4. `test_ensure_table_existing` tests if `ensure_table` function in the program can load existed table well.

5. `test_ensure_table_creates_if_missing` tests if `ensure_table` function creates new table well.

`test_frontend.py`  in the `Streamlit_Frontend`:

6. `test_backend_predict` tests if `backend_predict` function retrieves the prediction from FastAPI backend well.

**Integration Tests**

`test_backend.py` in the `FastAPI_Backend`: 

1. `test_predict2` tests if program runs endpoints `predict` well.
2. `test_predict3` tests if program captures wrong input type and missing input value well.

`test_manage.py` in the `Model_Management`:

3. `test_main` tests if program runs the entire `main()` well.

`test_dashboard.py` in the `Monitor_Streamlit`:

4. `test_main` tests if program runs the entire `main()` well.

# 4.2. CI/CD Pipeline

We created `.github/workflows/ci.yml` under `./4705_Personalized_Book_Recommender`

In the `ci.yml`, we have the following settings

1. Trigger Setting. The workflow automatically triggers on push requests to the dev branch and pull requests to the main branch.

```
on: 
  push:
    branches: [dev]
  pull_request:
    branches: [master]
```

2. Set up running environment and install essential packages

```
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.13'
- name: Install dependencies
  run: |
    python3 -m pip install --upgrade pip
    python3 -m pip install flake8 pytest pytest-asyncio ruff
    python3 -m pip install -r ./requirements.txt
```

3. Check flake 8 linter and execute all test suite

```
- name: Check style issue with Flake8
  run: |
    echo "Use Flake8"
    flake8 ./FastAPI_Backend/main.py
    flake8 ./Model_Management/train_model.py
    flake8 ./Monitor_Streamlit/monitor_app.py
    flake8 ./Streamlit_Frontend/frontend_app.py
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

- name: Run pytest on FastAPI_Backend
  run: |
    pytest -v ./FastAPI_Backend/test_backend.py

- name: Run pytest on Model_Management
  run: |
    pytest -v ./Model_Management/test_manage.py

- name: Run pytest on Monitor_Streamlit
  run: |
    pytest -v ./Monitor_Streamlit/test_dashboard.py

- name: Run pytest on Streamlit_Frontend
  run: |
    pytest -v ./Streamlit_Frontend/test_frontend.py
```

Then, we push all code to git repo

```bash
cd 4705_jiansun_assignment6
git init
git add README.md
git commit -m "Initial commit"
git remote add origin https://github.com/jiansfoggy/4705_Personalized_Book_Recommender.git
git branch -M master
git push -u origin master
```

```bash
git status # make sure we are on master branch
git checkout -b dev
git status # make sure we are on dev branch
git add --all
git commit -m “init”
git push
```

Go to github and set up pull request.

1. Click **Pull requests**
2. Click **New pull request**
3. Take `main` as base, `dev` as compare
4. Click **Create pull request**
5. Submit a new commit

## Phase 5: Containerization and Deployment

## 5.1. Docker Packaging

* All three folders `FastAPI_Backend`, `Monitor_Streamlit`, `Streamlit_Frontend` have individual `Dockerfile`

* `./4705_Personalized_Book_Recommender/Makefile` creates separate docker images and containers for FastAPI Backend and Streamlit Frontend

* `./4705_Personalized_Book_Recommender/Monitor_Streamlit/Makefile` creates docker image and container for Streamlit Monitor Dashboard.

## 5.2. AWS Deployment


## 5.3. Documentation

This is the required `README.md`.



---

---

## CI/CD with Github Actions
  
  Create **requirements.txt** file before pushing to Github.
  ```bash
  pip3 freeze > requirements.txt
  ```

- Code `ci.yml` file and save it under `.github/workflows`.

- `ci.yml` helps check code quality when we push the project to github.

- Create an emply `4705_jiansun_assignment6` folder and initialize it as Github repository.

  ```bash
  cd 4705_jiansun_assignment6
  git init
  git add .
  git commit -m "Initial commit"
  git remote add origin https://github.com/jiansfoggy/4705_jiansun_assignment6.git
  git branch -M master
  git push -u origin master
  ```

  Then, move all project files into `4705_jiansun_assignment6`, run the following code.

  ```bash
  git status # make sure we are on master branch
  git checkout -b dev
  git status # make sure we are on dev branch
  git pull
  git add --all
  git commit -m "Update"
  git push
  ```

- In the Github repository `4705_jiansun_assignment6`, click **Pull requests** to create new pull from `dev` to `master` branch, enter and submit commit, then yaml files runs like a script. 

- Click **Actions** to view details and debug.

---

## Build and run the API locally using **Makefile**:
  
  All docker commands are embedded into **Makefile**.

  1. **Install dependencies**

      In the `./4705_jiansun_assignment6`, Run `make build` to build the Docker image

      ```bash
      make build
      ```

  2. **Start the server**

      Run `init-volume` to initialize and copy files in Prediction_FastAPI/logs/ to Docker Volume via a one-time container.

      ```bash
      make init-volume
      ```

      If success, it shows this on screen.

      ```bash
      >> Initializing volume 'logs-volume' with existing logs...
      Unable to find image 'alpine:latest' locally
      latest: Pulling from library/alpine
      6e174226ea69: Pull complete 
      Digest: sha256:4bcff63911fcb4448bd4fdacec207030997caf25e9bea4045fa6c8c44de311d1
      Status: Downloaded newer image for alpine:latest
      >> Volume initialized.
      ```

      Run `make run` to activate and process the container from the image.

      FastAPI and Streamlit share the logs via the same Docker Volume

      ```bash
      make run
      ```
      
      Services are up:
      ```bash
      • FastAPI at http://localhost:8000
      • Streamlit Monitor at http://localhost:8501
      ```
  
  3. **Activate FastAPI**

      Run the following code to activate the localhost
      ```bash
      uvicorn main:app --reload
      ```

  4. **Delete Containers and Images**

      Run `make clean` to delete the created docker containers and images.

      ```bash
      make clean
      ```

---

## Interactive Docs

Once running, FastAPI automatically generates its documentation. Explore and test all endpoints via Swagger UI at:

```
http://127.0.0.1:8000/docs
```

---


---

## How to manually deploy the service on EC2 

1. Enter the AWS Sandbox by clicking the [link](https://awsacademy.instructure.com/courses/123950/modules/items/11736909).
2. Click `Start Lab` at top right corner in the top bar.
3. A window pops up. When "Lab status: ready" at the last line shows up, please close the window.
4. While waiting for initialization, let's turn on terminal and run these codes:

    ```bash
    cd ~/Downloads
    mkdir AWS_EC2
    ```

5. Go back to sandbox window, click `Details` at top right corner, click `show` followed by clicking `Download PEM`.
6. Move the download `labsuser.PEM` to AWS_EC2. Go back to sandbox window, click `AWS` on the left of `Start Lab` button. It turns on a new window.

    ```bash
    mv ~/Downloads/labsuser.PEM ./AWS_EC2
    ```

7. In the search bar, please type EC2 and hit enter. Then, we create new EC2 Instance by clicking `Launch Instance`.
8. Set Name as `EC2_HW6`.
9. In the **Application and OS Images**, we click `Browse more AMIs` and select `Ubuntu Server 24.04 LTS (HVM), SSD Volume Type`.
  
   In the shown `AMI from catalog`, we record the Username (it should be ubuntu) for further loging into the remote server.

10. Under the **Instance type**, keep the default `t2.micro` selected.
11. About **Key pair (login)**, please choose vockey.
12. In the **Network settings**, click `Edit`. In the **Availability Zone**, we select `us-ease-1a`.
13. About **firewall**, click **Select existing Security group** and use default one. 

    We will set up new one later.

14. We use 20GiB when setup **Configure storage**.
15. For **Advanced details**, we `Enable` **Termination protection**. Then, copy belowed code and paste it to the bottom chat box.

    ```bash
    #!/bin/bash
    dnf install -y httpd
    systemctl enable httpd
    systemctl start httpd
    echo '<html><h1>Hello From Your Web Server!</h1></html>' > /var/www/html/index.html
    ```

18. Click `Launch instance` and enter new page.
19. Click `view all instance` at the bottom right corner to enter a new page. Then, click `refresh` symbol near `connect` button and wait for initiaizing.
20. When **instance status** of new EC2 instance is running, we find **Network&Security** at the left sidebar and click `Security Groups`.
21. Click "Create security group" on the top right.
22. Enter these content -- `Security group name` = Assignment6; `Description` = SSH_FastAPI_Streamlit; use default VPC.
23. In the **Inbound rules** block, click `Add rule`. create 3 rules based on the following info.
24. Security group rule 1: 
   Type = `SSH` Port = `22` Source type = `My IP`(97.228.134.68/32) Description = `Port 22 (SSH)`
25. Security group rule 2: 
   Type = `Custom TCP` Port = `8000` Source type = `Anywhere-IPv4` Description = `Port 8000 (FastAPI)`
26. Security group rule 3: 
   Type = `Custom TCP` Port = `8501` Source type = `Anywhere-IPv4` Description = `Port 8501 (Streamlit)`
27. Click `Create security group` at the bottom right.
28. Click **Instances** from Sidebar, select **EC2_HW6** and click **Actions**. Then, go to **Security**, **Change Security Groups**. At **Associated security groups**, select `Assignment6`, click `Add security group`. click `save`.
29. We copy `Public IPV4 address` in the **Details**, and go to the terminal.
30. Under ./AWS_EC2, run code to activate EC2 ubuntu server

    ```bash
    cd ./AWS_EC2
    chmod 400 labsuser.pem
    ssh -i labsuser.pem ubuntu@<Public IPV4 address>
    # Enter yes if asked
    ```

31. Now, we connect to remote server with ubuntu 24.04 OS. Let's set up it for API deployment.
32. Set up python virtual environment
    
    ```bash
    sudo apt update
    sudo apt install python3-venv make
    # Enter yes or y if asked
    python3 -m venv ec2
    # Enter yes or y if asked
    . ec2/bin/activate
    ```

33. Install Docker

    ```bash
    # Add Docker's official GPG key:
    # Enter yes or y if asked
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Enter yes or y if asked
    sudo apt-get update
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo docker run hello-world
    ```

35. Install Git and connect to Github

    ```bash
    # Enter yes or y if asked
    sudo apt update
    sudo apt install git -y
    git --version
    git config --global user.name "jiansfoggy"
    git config --global user.email "Jian.Sun86@du.edu"
    git config --list
    ssh-keygen -t ed25519 -C "Jian.Sun86@du.edu"
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
    cat ~/.ssh/id_ed25519.pub
    ```

36. Go to Github, enter **Settings** and **SSH and GPG keys**, click **New SSH key**.
37. Set Title as `AWS-EC2`, copy ssh-rsa from the terminal, and paste it to the `Key` box.
38. Click `Add SSH key`, enter Github password, click `Confirm`.
39. Go back to terminal and download Git Repository `4705_jiansun_assignment6`.

    ```bash
    sudo usermod -aG docker $USER
    exit
    ssh -i labsuser.pem ubuntu@<Public IPV4 address>
    . ec2/bin/activate
    ssh -T git@github.com
    git clone -b dev git@github.com:jiansfoggy/4705_jiansun_assignment6.git
    cd 4705_jiansun_assignment6
    ```

40. Implement Pytest again to check code.

  ```bash
  python3 -m pip install --upgrade pip
  python3 -m pip install pytest pytest-asyncio
  python3 -m pip install -r ./Monitor_Streamlit/requirements.txt
  python3 -m pip install -r ./Prediction_FastAPI/requirements.txt
  cd Monitor_Streamlit
  pytest -v test_dashboard.py
  cd ../Prediction_FastAPI
  pytest -v test_api.py
  ```

41. Install docker images for FastAPI and Streamlit Monitor 

    ```bash
    cd ..
    make build
    make init-volume
    make run
    cd Prediction_FastAPI
    python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

40. Make sure that the service listens to `0.0.0.0` on EC2 for both ports

    ```bash
    sudo ss -tulpn | grep -E '(:8000|:8501)' || sudo netstat -tulpn | grep -E '(:8000|:8501)' || true
    sudo ufw status verbose
    sudo iptables -L -n -v
    ps -ef | grep uvicorn
    ```

    If Streamlit listens to `0.0.0.0:8000` and FastAPI listens to `0.0.0.0:8501`, we connect successfully.

41. Check if curl command returns status code like `200` or `302` in EC2 SSH terminal to make sure url are built successfully.

    ```bash
    curl http://127.0.0.1:8000/docs
    curl http://127.0.0.1:8501
    ```

42. Check streamlit monitor dashboard.

    ```bash
    http://<Public IPv4 address>:8501/
    ```

43. Check FastAPI Service

    ```bash
    http://<Public IPv4 address>:8000/docs
    ```

44. The two service are deployed successfully, if there is no error poping up.

## Notes

* Ensure `sentiment_model.pkl` and `IMDB_Dataset.csv` are in the project subdirectory `Prediction_FastAPI` and `Monitor_Streamlit`.
* The `Makefile` automates setup, training, and running tasks.
* Produced docs `/docs` reflect real-time API schema and let you issue live requests.

not test pytest on EC2 yet.
# API Service & Model Monitoring & CI/CD & Testing

## Introduction

A pipeline tests the program and application and deliveries and deploys the model to Github, which contains two services run independently but communicate with each other and their Docker build file. The entire process will be deployed on AWS EC2:

A lightweight **FastAPI**-based service that classifies text as **Positive** or **Negative**, and provides sentiment probabilities.

A **Streamlit App** that reads the logs from the shared volume to visualize model performance.

A **Docker** Volume that persist log data and share it between the two containers.

A **.github/workflows** that tests the code quality and program's bug while uploading to Github and pulling repository from `dev` branch to `master` one.

A **AWS EC2** server running Linux environment to deploy services like FastAPI backend and Streamlit monitor dashboard.

### **Project Architecture**

- `Monitor_Streamlit`: contains files to build streamlit

- `Prediction_FastAPI`: contains files to build FastAPI

- `Makefile`: builds multi-containers for this application

- ` CI/CD Pipeline`: `.github/workflows/ci.yml` lists workflows and points to check for automate code quality checking while pulling request to `master` branch.

- `README.md`: introduces the entire project and displays how to run the project.

---

## Prerequisites

To run this app, please make sure `Docker`, `Git`, `FastAPI`, `Postman`, `pytest`, `AWS Sandbox`, other essential softwares, and other essential python packages mentioned in the `requirements.txt` are installed. 

Turn on **AWS Sandbox and EC2** at step 1.

---

## Local Error Test

- Test the modules to ensure reliability
  
  1. Install `pytest` packages

  ```bash
  python3 -m pip install --upgrade pip
  python3 -m pip install pytest pytest-asyncio
  ```

  2. Enter **Monitor_Streamlit** and run `test_dashboard.py` to test if the code can launch Streamlit application without errors.

  ```bash
  cd Monitor_Streamlit
  pytest -v test_dashboard.py
  ```

  3. Enter **Prediction_FastAPI** and run `test_api.py` to test if `/predict` endpoint predicts positive and negative samples correctly and successfully tackles the improper input data, like missing or malformed data.

  ```bash
  cd Prediction_FastAPI
  pytest -v test_api.py
  ```

---

## Features & Endpoints

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

### **3. `POST /predict_proba`**

* **Purpose**: Returns sentiment along with the model’s confidence.
* **Running Example**: http://127.0.0.1:8000/predict_proba?text=This%20movie%20was%20a%20masterpiece!&true_sentiment=Positive.
* **Request Body**:

  ```json
  {
    "text": "I love this!",
    "true_sentiment":"Positive"
  }
  ```
* **Successful Response**:

  ```json
  {
    "sentiment": "Positive",
    "probability": "0.9532"
  }
  ```
* **Error Cases**:

  * `400 Bad Request` if the text is empty.
  * `503 Service Unavailable` if the model fails to load.

### **4. `GET /example`**

* **Purpose**: Returns a random review from the local IMDB dataset (`IMDB_Dataset.csv`).
* **Response**:

  ```json
  {
    "review": "This movie is fantastic—I loved every moment!"
  }
  ```

---

## CI/CD with Github Actions

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

## Usage Example

```bash
# Check health
curl http://127.0.0.1:8000/health

# Predict sentiment
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"text":"What a lovely story!!","true_sentiment":"Positive"}' \
     http://127.0.0.1:8000/predict

# Predict with probability
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"text":"What a lovely story!","true_sentiment":"Positive"}' \
     http://127.0.0.1:8000/predict_proba

# Check example
curl http://127.0.0.1:8000/example
```

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

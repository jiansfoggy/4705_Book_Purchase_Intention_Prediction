# Book Purchase Intention Prediction

## Introduction

This project analyzes the book review from Amazon Reviews Data 2023 -- Book Subset to estimate if the customer bought the book.

We want to study the relationship between review attitude and final purchase status. The conclusion is less correlated. What people say doesn't gurrantee what they do.

The entire projects stand on a pipline, which tests the program and application and deliveries and deploys the model to Github, which contains three services run independently but communicate with each other and their Docker build file. The entire process will be deployed on two AWS EC2:

A **Machine Learning Model** trains the data to generate a weight file. It saves the weight file as artifact in the WandB.

A lightweight **FastAPI**-based Backend service that classifies text as **Positive** or **Negative**.

**Positive** means user bought the book. **Negative** means not.

A **Streamlit Frontend User Interface** that takes user's review as input and sends it to backend for prediction and displays the result.

A **Streamlit Monitor Dashboard** that reads the logs from the Amazon DynamoDB to visualize model performance.

A **Docker** launches Frontend, Backend, and Monitor services.

A **.github/workflows** that tests the code quality and program's bug while uploading to Github and pulling repository from `dev` branch to `master` one.

Two **AWS EC2** servers run Linux environment to deploy FastAPI Backend and Streamlit Frontend at the same server, then it puts Streamlit monitor dashboard at the other.

### **Project Architecture**

- `data`
- `Model_Management`: contains python file to train machine learning for book recommandation, and model weight file.
- `FastAPI_Backend`: contains files to build FastAPI Backend.
- `Streamlit_Frontend`: contains files to build Streamlit Frontend.
- `Monitor_Streamlit`: contains files to build Streamlit Monitor.
- `Makefile`: builds multi-containers for this application
- `CI/CD Pipeline`: `.github/workflows/ci.yml` lists workflows and points to check for automate code quality checking while pulling request to `master` branch.
- `README.md`: introduces the entire project and displays how to run the project.

---

## Prerequisites

To run this app, please make sure `Docker`, `Git`, `WandB`, `FastAPI`, `Postman`, `pytest`, `AWS Learner Lab`, other essential softwares, and other essential python packages mentioned in the `requirements.txt` are installed. 

Turn on **AWS Learner Lab** at step 1.

---

# Please go to each folder to find the introduction of first 4 phases.

This file focuses on the last phase.

---

## Phase 5: Containerization and Deployment

## 5.1. Docker Packaging

* All three folders `FastAPI_Backend`, `Monitor_Streamlit`, `Streamlit_Frontend` have individual `Dockerfile`

* `./4705_Personalized_Book_Recommender/Makefile` creates separate docker images and containers for FastAPI Backend and Streamlit Frontend

* `./4705_Personalized_Book_Recommender/Monitor_Streamlit/Makefile` creates docker image and container for Streamlit Monitor Dashboard.

## 5.2. AWS Deployment

### 5.2.1 Manually deploy the service on first EC2 

1. Enter the **AWS Academy Learner Lab** by clicking the [link](https://awsacademy.instructure.com/courses/127314/modules/items/12104634).
2. Click `Start Lab` at top right corner in the top bar.
3. When green light near `AWS` button on the top left is on, click it to enter new tab.
4. While waiting for initialization, let's turn on terminal and run these codes:

    ```bash
    cd ~/Downloads
    mkdir AWS_EC2
    ```

5. Go back to Learner Lab window, click `Details` at top right corner, click `show` followed by clicking `Download PEM`.
6. Move the download `labsuser.PEM` to AWS_EC2. Go back to sandbox window, click `AWS` on the left of `Start Lab` button. It turns on a new window.

    ```bash
    mv ~/Downloads/labsuser.PEM ./AWS_EC2
    ```

7. In the search bar, please type EC2 and hit enter. Then, we create new EC2 Instance by clicking `Launch Instance`.
8. Set Name as `EC2_for_Frontend_Backend`.
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
22. Enter these content -- `Security group name` = Final_Porject; `Description` = SSH_FastAPI_Streamlit; use default VPC.
23. In the **Inbound rules** block, click `Add rule`. create 3 rules based on the following info.
24. Security group rule 1: 
   Type = `SSH` Port = `22` Source type = `My IP`(97.228.134.68/32) Description = `Port 22 (SSH)`
25. Security group rule 2: 
   Type = `Custom TCP` Port = `8000` Source type = `Anywhere-IPv4` Description = `Port 8000 (FastAPI)`
26. Security group rule 3: 
   Type = `Custom TCP` Port = `8501` Source type = `Anywhere-IPv4` Description = `Port 8501 (Streamlit)`
27. Click `Create security group` at the bottom right.
28. Click **Instances** from Sidebar, select **EC2_Frontend_Backend** and click **Actions**. Then, go to **Security**, **Change Security Groups**. At **Associated security groups**, select `Final_Porject`, click `Add security group`. click `save`.
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

34. Install Git and connect to Github

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

35. Go to Github, enter **Settings** and **SSH and GPG keys**, click **New SSH key**.
36. Set Title as `AWS-EC2`, copy ssh-rsa from the terminal, and paste it to the `Key` box.
37. Click `Add SSH key`, enter Github password, click `Confirm`.
38. Go back to terminal and download Git Repository `4705_Book_Purchase_Intention_Prediction`.

    ```bash
    sudo usermod -aG docker $USER
    exit
    ssh -i labsuser.pem ubuntu@<Public IPV4 address>
    . ec2/bin/activate
    ssh -T git@github.com
    git clone -b dev git@github.com:jiansfoggy/4705_Book_Purchase_Intention_Prediction.git
    cd 4705_Book_Purchase_Intention_Prediction
    ```

### 5.2.2 Download Amazon Reviews Dataset 2023

39. Run the following codes in the command line to download dataset.

To make the program runs smoothly, we have to create a subset out of Books.jsonl. 

`read_data.py` is to finish task like this, which creates `review_data.csv` and `test_data.json`

```bash
pwd # check if you are in ./4705_Book_Purchase_Intention_Prediction now.
mkdir data
cd data
curl -O https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz
```

Here you may need to increase the Volume size to continue decompressing data file.

* exit the server and stop instance for the current EC2 server
* In the Amazon Console sidebar, click Volumes under Elastic Block Store.
* Click **Actions** --> **Modify volume**
* Change the volume size to 40GB.
* Save the change. When it is done, restart instance.
* Enter the EC2 ubuntu server again and finish the resting part.

```bash
lsblk # make sure the volume size gets changed.
df -h
gunzip Books.jsonl.gz
cd ..
python3 -m pip install pandas
python3 read_data.py
rm -rf ./data/Books.jsonl
```

Now, you have two csv files, `review_data.csv` and `test_data.json`, in the `./data`.

40. Move data files to target path.

```bash
cp ./data/review_data.csv ./Monitor_Streamlit/
mv ./data/test_data.json ./FastAPI_Backend
```

### 5.2.3 Manually deploy the service on second EC2

41. Repeat Steps 7--19 and 29--40 to create second EC2 server with the following changes

* Start from step 7, since we already setup .pem file in Steps 1--6.
* At Step 8: Set Name as `EC2_for_Monitor`.
* At Step 13: About **firewall**, click **Select existing Security group** and use `Final_Porject`, the one we just created for first EC2 server. 
* Repeat Steps 20--28 to create new Security Group `Monitor` to avoid further trouble. Change description at will, but keep the other information unchanged.

### 5.2.4 Run User Interface and FastAPI services on the server `EC2_for_Frontend_Backend`

42. Setup WandB

Go to [here](https://wandb.ai/quickstart?product=models) and copy **WANDB_API_KEY** in the highlighted yellow block.

```bash
python3 -m pip install wandb
wandb login # copy and paste API Key while asking
export WANDB_API_KEY=<WANDB_API_KEY>
```

43. Setup Amazon DynamoDB: Launch **AWS Academy Learner Lab** and click **AWS Details** on the top right.
44. Click **AWS Details** on the top right. 
45. Under **Cloud Access**, click **Show**, and record `aws_access_key_id`
, `aws_secret_access_key`, `aws_session_token`, `AWSAccountId`, and `Region`.
46. Open **terminal** and install `boto3`, `awscli`, `hashlib`.
   
    ```bash
    python3 -m pip install --upgrade pip
    python3 -m pip install boto3 awscli
    ```

47. In the terminal, run `aws configure` and enter copied `aws_access_key_id`, `aws_secret_access_key`, `Region`, and `json`.
  
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
    
    If you see the following example feedback, you successfully set up a valid permit.
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

48. If Step 48 works badly, create a `.env` file under `4705_Book_Purchase_Intention_Prediction`.

- Given that the program is deployed in the Docker environment, it can't read `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc from local machine. We need to take these information to Docker environment while building it.

        ````bash
        vim .env
        WANDB_API_KEY="<paste from wandb>"
        AWS_ACCESS_KEY_ID="<paste from AWS Details>"
        AWS_SECRET_ACCESS_KEY="<paste from AWS Details>"
        AWS_SESSION_TOKEN="<paste from AWS Details>"
        AWS_REGION="<paste from AWS Details>"
        ```
- Save the file and put it into `.gitignore`.
        ````bash
        echo "*.env" >> .gitignore
        cp ./.env ./FastAPI_Backend
        cp ./.env ./Monitor_Streamlit
        ```
49. Now, Amazon DynamoDB is ready. Let's activate Frontend Streamlit User Interface and FastAPI Backend.

Make sure you are in the directory of `4705_Book_Purchase_Intention_Prediction`.

```bash
make build
make init-volume
make run
docker logs <container_names> 
```
50. Check Streamlit Frontend User Interface.

    ```bash
    http://<Public IPv4 address>:8501/
    ```
51. Check FastAPI Backend.

    ```bash
    http://<Public IPv4 address>:8000/docs
    ```
52. You will see the following after using `make logs`. It means the Backend and Frontend are built successfully.

```
Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.


  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.18.0.3:8501
  External URL: http://54.197.165.152:8501


   FastAPI   Starting production server 🚀
 
             Searching for package file structure from directories with         
             __init__.py files                                                  
             Importing from /code
 
    module   🐍 main.py
 
      code   Importing the FastAPI app object from the module with the following
             code:                                                              
 
             from main import app
 
       app   Using import string: main:app
 
    server   Server started at http://0.0.0.0:8000
    server   Documentation at http://0.0.0.0:8000/docs
 
             Logs:
 
      INFO   Started server process [1]
      INFO   Waiting for application startup.
      INFO   Application startup complete.
      INFO   Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
[DDB] Table 'Backend_Log_Cache' found.
Table status: ACTIVE
      INFO   172.18.0.3:47084 - "POST /predict HTTP/1.1" 200
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: WARNING Artifact.get_path(name) is deprecated, use Artifact.get_entry(name) instead.
Detected EC2 (Learner Lab). Using IAM Role credentials...
[DDB] Table 'Backend_Log_Cache' found.
Table status: ACTIVE
/code/artifacts/MultinomialNB-artifact:v1/purchase_model.pkl
Model 'MultinomialNB-artifact:latest' loaded successfully from W&B.
Create local log file at ./logs/prediction_logs.json
[DDB] put succeed: Cache data to DynamoDB
      INFO   172.18.0.3:37884 - "POST /predict HTTP/1.1" 200
```

### 5.2.5 Run Monitor Dashboard on the server `EC2_for_Monitor`

53. Repeat Steps 42--48 to prepare and connect to WandB and Amazon DynamoDB here.

54. Let's activate Streamlit Model Monitor Dashboard.

Make sure you are in the directory of `4705_Book_Purchase_Intention_Prediction`.

```bash
cd Monitor_Streamlit
make build
make run
docker logs <container_names> 
```

55. Check streamlit monitor dashboard.

```bash
http://<Public IPv4 address for server 2>:8501/
```
55. You will see


56. Trouble shooting. Make sure that the service listens to `0.0.0.0` on EC2 for both ports

    ```bash
    sudo apt install -y net-tools
    sudo ss -tulpn | grep -E '(:8000|:8501)' || sudo netstat -tulpn 
    sudo ss -tulpn | grep -E '(:8000|:8501)' || sudo netstat -tulpn | grep -E '(:8000|:8501)' || true
    sudo ufw status verbose
    sudo iptables -L -n -v
    ps -ef | grep uvicorn
    ```

    If Streamlit listens to `0.0.0.0:8000` and FastAPI listens to `0.0.0.0:8501`, it connects successfully.

43. Check if curl command returns status code like `200` or `302` in EC2 SSH terminal to make sure url are built successfully.

    ```bash
    curl http://127.0.0.1:8000/docs
    curl http://127.0.0.1:8501
    ```

44. Check streamlit monitor dashboard.

    ```bash
    http://<Public IPv4 address>:8501/
    ```

45. Check FastAPI Service

    ```bash
    http://<Public IPv4 address>:8000/docs
    ```

60. The two service are deployed successfully, if there is no error poping up.

## 5.3. Documentation

You are reading the `README.md` at the root path. Visit folders `FastAPI_Backend`, `Model_Management`, `Monitor_Streamlit`, and `Streamlit_Frontend` to find the README.md file for each task.



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


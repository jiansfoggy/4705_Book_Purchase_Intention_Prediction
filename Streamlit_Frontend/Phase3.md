# Book Purchase Intention Prediction

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
    cp ../.env ./
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

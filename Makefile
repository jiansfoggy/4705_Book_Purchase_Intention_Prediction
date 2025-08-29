# Define variables for the image name and tag

# Image names
IMAGE_BACKEND      := backend_api
IMAGE_FRONTEND     := frontend_user

# Container names
CONTAINER_BACKEND  := backend_api_0822
CONTAINER_FRONTEND := frontend_user_0822

# Shared network & volume
NETWORK            := app-network
VOLUME             := logs-volume

# Paths
BACKEND_DIR        := FastAPI_Backend
FRONTEND_DIR       := Streamlit_Frontend

.PHONY: all init-volume build run logs clean

all: build

init-volume:
	@echo ">> Initializing volume '$(VOLUME)' with existing logs..."
	@docker volume inspect $(VOLUME) >/dev/null 2>&1 || docker volume create $(VOLUME)
	@docker run --rm \
	    -v $(VOLUME):/data \
	    -v $(shell pwd)/$(BACKEND_DIR)/logs:/src-logs \
	    alpine \
	    sh -c "cp /src-logs/prediction_logs.json /data/ || true"
	@echo ">> Volume initialized."

build: build-api build-monitor

build-api:
	@echo "Building Docker image: $(IMAGE_BACKEND)"
	docker build -t $(IMAGE_BACKEND) ./$(BACKEND_DIR)

build-monitor:
	@echo "Building Docker image: $(IMAGE_FRONTEND)"
	docker build -t $(IMAGE_FRONTEND) ./$(FRONTEND_DIR)

run: # build
	# Create network and volume if they don't exist
	@echo "Prepare network and volume..."
	docker network inspect $(NETWORK) >/dev/null 2>&1 || docker network create $(NETWORK)
# 	docker volume inspect $(VOLUME)   >/dev/null 2>&1 || docker volume create $(VOLUME)

	# FastAPI service
	@echo "Running FastAPI Docker container..."
	docker rm -f $(CONTAINER_BACKEND) >/dev/null 2>&1 || true
	docker run --env-file .env -d \
	  --name $(CONTAINER_BACKEND) \
	  --network $(NETWORK) \
	  -p 8000:8000 \
	  -v $(shell pwd)/$(BACKEND_DIR):/app \
	  -v $(VOLUME):/app/logs \
	  $(IMAGE_BACKEND)

	# Streamlit monitoring dashboard
	@echo "Running Streamlit Docker container..."
	docker rm -f $(CONTAINER_FRONTEND) >/dev/null 2>&1 || true
	docker run -d \
	  --name $(CONTAINER_FRONTEND) \
	  --network $(NETWORK) \
	  -p 8501:8501 \
	  -e BACKEND_URL=http://$(CONTAINER_BACKEND):8000 \
	  -v $(shell pwd)/$(FRONTEND_DIR):/app \
	  -v $(VOLUME):/app/logs \
	  $(IMAGE_FRONTEND)

	@echo "Services are up:"
	@echo " • FastAPI Backend at http://localhost:8000"
	@echo " • Streamlit Frontend at http://localhost:8501"

logs:
	@echo "Check logs (Ctrl-C to stop)..."
	docker logs -f $(CONTAINER_FRONTEND) &
	docker logs -f $(CONTAINER_BACKEND) &
	@wait

clean:
	@echo "Removing Docker image: $(CONTAINER_BACKEND) $(CONTAINER_FRONTEND)"

	docker rm -f $(CONTAINER_BACKEND) $(CONTAINER_FRONTEND) || true
	docker network rm $(NETWORK)
	docker volume rm $(VOLUME)
	docker rmi $(IMAGE_BACKEND) $(IMAGE_FRONTEND) || true
	docker rmi alpine:latest || true

# http://127.0.0.1:8000
# http://127.0.0.1:8000/docs
# -v $(VOLUME):/app/logs \
# -v $(shell pwd)/$(API_DIR)/logs:/app/logs \
# -v $(shell pwd)/$(API_DIR)/logs/prediction_logs.json:/app/prediction_logs.json \
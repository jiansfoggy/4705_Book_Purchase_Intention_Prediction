# Makefile for Dockerized Streamlit App

# Define variables for the image name and tag
IMAGE_NAME     := streamlit_monitor
CONTAINER_NAME := streamlit_monitor_0822

.PHONY: build run logs clean

build:
	@echo "Building Docker image: $(IMAGE_NAME)"
	docker build -t $(IMAGE_NAME) .

run:
	@echo "Running Docker container..."
# 	docker run --rm -p 8501:8501 $(IMAGE_NAME)
	docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1 || true
	docker run --env-file .env -d \
	  --name $(CONTAINER_NAME) \
	  -p 8501:8501 $(IMAGE_NAME)

logs:
	@echo "Check logs (Ctrl-C to stop)..."
	docker logs -f $(CONTAINER_NAME) &
	@wait

clean:
	@echo "Removing Docker image: $(IMAGE_NAME)"
	docker rm -f $(CONTAINER_NAME) || true
	docker rmi -f $(IMAGE_NAME) || true
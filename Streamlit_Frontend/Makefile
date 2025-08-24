# Makefile for Dockerized Streamlit App

# Define variables for the image name and tag
IMAGE_NAME := Frontend_Streamlit

build:
	@echo "Building Docker image: $(IMAGE_NAME)"
	docker build -t $(IMAGE_NAME) .

run:
	@echo "Running Docker container..."
	docker run --rm -p 8501:8501 $(IMAGE_NAME)

clean:
	@echo "Removing Docker image: $(IMAGE_NAME)"
	docker rmi -f $(IMAGE_NAME) || true

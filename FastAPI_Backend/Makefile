# Define variables for the image name and tag
IMAGE_NAME := sentiment-api
CONTAINER_NAME := sentiment_api_0817

build:
	@echo "Building Docker image: $(IMAGE_NAME)"
	docker build -t $(IMAGE_NAME) .

run: build
	@echo "Running Docker container..."
	docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1 || true
	docker run --env-file .env -d --name $(CONTAINER_NAME) -p 8000:8000 $(IMAGE_NAME)

clean:
	@echo "Removing Docker image: $(IMAGE_NAME) and container: $(CONTAINER_NAME)"
	docker rm -f $(CONTAINER_NAME) || true
	docker rmi -f $(IMAGE_NAME) || true

# http://127.0.0.1:8000
# http://127.0.0.1:8000/docs
# 	docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1 || true
# docker ps --filter "name=sentiment_api_0817"
# docker logs sentiment_api_0817

# vim .env


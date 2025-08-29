# Define variables for the image name and tag
IMAGE_NAME := bought-api
CONTAINER_NAME := bought_0817

build:
	@echo "Building Docker image: $(IMAGE_NAME)"
	docker build -t $(IMAGE_NAME) .

run: build
	@echo "Running Docker container..."
	docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1 || true
	docker run --env-file .env -d --name $(CONTAINER_NAME) -p 8000:8000 $(IMAGE_NAME)

logs:
	@echo "Check logs (Ctrl-C to stop)..."
	docker logs -f $(CONTAINER_NAME) &
	@wait

clean:
	@echo "Removing Docker image: $(IMAGE_NAME) and container: $(CONTAINER_NAME)"
	docker rm -f $(CONTAINER_NAME) || true
	docker rmi -f $(IMAGE_NAME) || true

# http://127.0.0.1:8000
# http://127.0.0.1:8000/docs
# 	docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1 || true
# docker ps --filter "name=sentiment_api_0817"
# docker logs bought_0817

# vim .env


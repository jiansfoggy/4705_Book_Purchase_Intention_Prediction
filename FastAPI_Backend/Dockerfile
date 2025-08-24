# Use an official lightweight Python image as a parent image
FROM python:3.13-slim

# Set the working directory in the container to /app
WORKDIR /code

# Copy the requirements file into the container at /app
COPY requirements.txt /code/

# Install any needed packages specified in requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application's code into the container at /app
COPY ./main.py /code/
COPY ./sentiment_model.pkl /code/

# Make port 80 available to the world outside this container
EXPOSE 8000

# Define the command to run your app when the container starts
CMD ["fastapi", "run", "./main.py", "--port", "8000"]
# CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"， "--log-level", "debug"]

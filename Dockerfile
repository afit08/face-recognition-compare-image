# Use the official Python image as base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libgtk2.0-dev \
        pkg-config \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        && rm -rf /var/lib/apt/lists/*

# Install dlib
RUN pip install dlib

# Install Flask and other Python dependencies
RUN pip install --no-cache-dir Flask face_recognition

# Expose the port the app runs on
EXPOSE 8100

# Define the command to run the application
CMD ["python", "app.py"]

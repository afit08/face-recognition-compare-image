#!/bin/bash

IMAGE_NAME="face-recognition-compare-image"
CONTAINER_NAME="data-face-recognition-compare-image"
PORT=8100

echo "📦 Building Docker image..."
sudo docker build -t $IMAGE_NAME .


echo "🛑 Stopping old container (if running)..."
sudo docker stop $CONTAINER_NAME 2>/dev/null || true

echo "🧹 Removing old container (if exists)..."
sudo docker rm $CONTAINER_NAME 2>/dev/null || true

echo "🚀 Starting new container..."
sudo docker run -d \
  -p $PORT:$PORT \
  --restart always \
  --name $CONTAINER_NAME \
  $IMAGE_NAME

echo "✅ Deployment complete. Container logs:"
sudo docker logs -f $CONTAINER_NAME
#!/bin/bash

echo "Starting Big Data System..."

# Start Kafka and Spark
echo "Starting Kafka and Spark with Docker Compose..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Create Kafka topic
echo "Creating Kafka topic..."
docker exec -it $(docker ps -q -f name=kafka) kafka-topics --create --topic retail_data_stream --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# Start Kafka Consumer in background
echo "Starting Kafka Consumer..."
python kafka/consumer.py &
CONSUMER_PID=$!

# Wait a bit
sleep 5

# Start Kafka Producer
echo "Starting Kafka Producer..."
python kafka/producer.py &
PRODUCER_PID=$!

# Start API Server
echo "Starting API Server..."
python api/app.py &
API_PID=$!

echo "System started successfully!"
echo "Producer PID: $PRODUCER_PID"
echo "Consumer PID: $CONSUMER_PID"
echo "API PID: $API_PID"
echo ""
echo "API endpoints available at http://localhost:5000"
echo "Kafka UI available at http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "echo 'Stopping services...'; kill $PRODUCER_PID $CONSUMER_PID $API_PID; docker-compose down; exit" INT
wait
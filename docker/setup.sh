#!/bin/bash
# Setup script for RAG Tools infrastructure

set -e

echo "Setting up RAG Tools infrastructure..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed"
    exit 1
fi

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env with your configuration"
else
    echo ".env file already exists"
fi

# Create necessary directories
echo "Creating data directories..."
mkdir -p ../data ../logs

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
for i in {1..30}; do
    if docker-compose exec -T postgres pg_isready -U rag_tools > /dev/null 2>&1; then
        echo "PostgreSQL is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Error: PostgreSQL failed to start"
        exit 1
    fi
    sleep 1
done

# Wait for Qdrant to be ready
echo "Waiting for Qdrant..."
for i in {1..30}; do
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo "Qdrant is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Error: Qdrant failed to start"
        exit 1
    fi
    sleep 1
done

echo ""
echo "Setup complete!"
echo ""
echo "Services:"
echo "  - PostgreSQL: localhost:5432"
echo "  - Qdrant:     localhost:6333"
echo ""
echo "Run 'docker-compose logs -f' to view logs"

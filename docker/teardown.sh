#!/bin/bash
# Teardown script for RAG Tools infrastructure

set -e

echo "Tearing down RAG Tools infrastructure..."

# Stop and remove containers
echo "Stopping services..."
docker-compose down

# Ask about removing volumes
read -p "Remove data volumes? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing volumes..."
    docker-compose down -v
    echo "Volumes removed"
else
    echo "Volumes preserved"
fi

echo "Teardown complete!"

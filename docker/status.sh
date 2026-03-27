#!/bin/bash
# Status check script for RAG Tools infrastructure

echo "Checking RAG Tools infrastructure status..."
echo ""

# Check PostgreSQL
echo -n "PostgreSQL: "
if docker-compose exec -T postgres pg_isready -U rag_tools > /dev/null 2>&1; then
    echo "Running"
else
    echo "Not running"
fi

# Check Qdrant
echo -n "Qdrant:     "
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo "Running"
else
    echo "Not running"
fi

# Check pgAdmin (if running)
echo -n "pgAdmin:    "
if docker ps --format '{{.Names}}' | grep -q rag_tools_pgadmin; then
    echo "Running"
else
    echo "Not running"
fi

echo ""

# Show container info
echo "Container Status:"
docker-compose ps

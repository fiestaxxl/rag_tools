# Docker Setup for RAG Tools

This directory contains Docker Compose configuration for running the infrastructure required by the RAG Tools module.

## Services

### PostgreSQL
- **Image**: PostgreSQL 16 Alpine
- **Port**: 5432
- **Credentials**: See `.env.example`
- **Volumes**: `postgres_data` for persistent storage

### Qdrant
- **Image**: Qdrant v1.7.0
- **HTTP Port**: 6333
- **gRPC Port**: 6334
- **Volumes**: `qdrant_data` for persistent storage

### pgAdmin (Optional)
- **Image**: pgAdmin 4
- **Port**: 5050
- **Profile**: `admin` (must be run explicitly)

## Quick Start

### 1. Copy environment file
```bash
cp .env.example .env
```

### 2. Start services
```bash
docker-compose up -d
```

### 3. Check status
```bash
docker-compose ps
```

### 4. View logs
```bash
docker-compose logs -f
```

## Managing Services

### Start services
```bash
docker-compose up -d
```

### Stop services
```bash
docker-compose down
```

### Stop and remove volumes (clean slate)
```bash
docker-compose down -v
```

### Start with pgAdmin
```bash
docker-compose --profile admin up -d
```

### View specific service logs
```bash
docker-compose logs -f postgres
docker-compose logs -f qdrant
```

## Testing Connections

### PostgreSQL
```bash
psql -h localhost -p 5432 -U rag_tools -d rag_tools
```

### Qdrant Health Check
```bash
curl http://localhost:6333/health
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| POSTGRES_USER | rag_tools | PostgreSQL username |
| POSTGRES_PASSWORD | rag_tools_password | PostgreSQL password |
| POSTGRES_DB | rag_tools | Database name |
| POSTGRES_PORT | 5432 | PostgreSQL port |
| QDRANT_HTTP_PORT | 6333 | Qdrant HTTP port |
| QDRANT_GRPC_PORT | 6334 | Qdrant gRPC port |
| QDRANT__API_KEY | (empty) | Optional Qdrant API key |

## Production Considerations

For production deployments, consider:

1. **Security**: Use secrets management for passwords
2. **Networking**: Use custom networks with proper isolation
3. **Resources**: Set appropriate CPU/memory limits
4. **Backups**: Configure backup strategies for PostgreSQL
5. **Monitoring**: Add health checks and logging aggregation

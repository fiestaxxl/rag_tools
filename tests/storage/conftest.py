# import os

# if not os.path.exists("/var/run/docker.sock"):
#     raise RuntimeError(
#         "Docker socket not available. "
#         "Run container with -v /var/run/docker.sock:/var/run/docker.sock"
#     )

# import pytest
# import asyncio
# from testcontainers.postgres import PostgresContainer
# from testcontainers.core.container import DockerContainer

# from rag_tools.storage import PostgresClient, QdrantClientWrapper
# from rag_tools.config.settings import PostgresSettings, QdrantSettings




# @pytest.fixture(scope="session")
# def event_loop():
#     loop = asyncio.get_event_loop()
#     yield loop
#     loop.close()


# # ------------------------
# # POSTGRES
# # ------------------------
# @pytest.fixture(scope="session")
# def postgres_container():
#     with PostgresContainer("postgres:15") as postgres:
#         yield postgres


# @pytest.fixture
# async def postgres_client(postgres_container):
#     config = PostgresSettings(
#         host=postgres_container.get_container_host_ip(),
#         port=postgres_container.get_exposed_port(5432),
#         user=postgres_container.USER,
#         password=postgres_container.PASSWORD,
#         database=postgres_container.DBNAME,
#         min_connections=1,
#         max_connections=5,
#     )

#     client = PostgresClient(config)
#     await client.initialize()

#     yield client

#     await client.close()


# # ------------------------
# # QDRANT
# # ------------------------
# @pytest.fixture(scope="session")
# def qdrant_container():
#     container = DockerContainer("qdrant/qdrant:latest").with_exposed_ports(6333)
#     container.start()
#     yield container
#     container.stop()


# @pytest.fixture
# def qdrant_client(qdrant_container):
#     url = f"http://{qdrant_container.get_container_host_ip()}:{qdrant_container.get_exposed_port(6333)}"

#     config = QdrantSettings(url=url)

#     client = QdrantClientWrapper(config)
#     client.connect()
#     client.set_embedding_dim(4)

#     yield client

#     client.close()
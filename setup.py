#!/usr/bin/env python3
"""Setup script for rag_tools module."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag_tools",
    version="0.1.0",
    author="Ivan Gurev",
    author_email="ivan.gurev@skoltech.ru",
    description="RAG-based tool retrieval system for MCP servers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fiesta_xxl/rag_tools",  # replace with actual repo
    packages=find_packages(include=["rag_tools", "rag_tools.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12.11",
    install_requires=[
        "pydantic>=2.0",
        "pydantic-settings>=2.0",
        "httpx>=0.24.0",
        "asyncpg>=0.28.0",
        "sqlalchemy>=2.0",
        "qdrant-client>=1.9.0",
        "numpy>=1.24.0",
        "aiohttp>=3.8.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "local": [
            "sentence-transformers>=2.2.0",
            "torch>=2.0.0",        # sentence-transformers pulls this but we keep explicit
        ],
        "eval": [
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.6.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
        ],
        "all": [
            "sentence-transformers>=2.2.0",
            "torch>=2.0.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.6.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
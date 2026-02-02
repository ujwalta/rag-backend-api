#!/bin/bash
# Quick start script for RAG Backend

echo "🚀 RAG Backend Quick Start"
echo "=========================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker not found. You'll need to install Redis and Qdrant manually."
else
    echo "✅ Docker found: $(docker --version)"
    
    # Start services with Docker Compose
    if [ -f "docker-compose.yml" ]; then
        echo "📦 Starting services (Redis, Qdrant)..."
        docker-compose up -d redis qdrant
        echo "✅ Services started"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created. Please review and update as needed."
fi

# Initialize database
echo "🗄️  Initializing database..."
python -c "
import asyncio
from app.services.database import get_database_service

async def init_db():
    db = get_database_service()
    await db.create_tables()
    print('✅ Database initialized')

asyncio.run(init_db())
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  uvicorn app.main:app --reload"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
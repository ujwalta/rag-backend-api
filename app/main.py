"""
Main FastAPI application.
Configures routes, middleware, and startup/shutdown events.
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime

from app.core.config import settings
from app.api import documents, chat
from app.services.database import get_database_service
from app.services.redis_service import get_redis_service
from app.models.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    print("Starting up RAG Backend API...")
    
    # Initialize database
    db_service = get_database_service()
    await db_service.create_tables()
    print("Database tables created/verified")
    
    # Connect to Redis
    redis_service = get_redis_service()
    await redis_service.connect()
    print("Connected to Redis")
    
    yield
    
    # Shutdown
    print("Shutting down RAG Backend API...")
    
    # Disconnect from Redis
    await redis_service.disconnect()
    print("Disconnected from Redis")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade RAG backend with conversational AI and interview booking",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
    description="Check the health status of the API and its dependencies"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the status of the API and its services.
    """
    services = {}
    
    # Check database
    try:
        from sqlalchemy import text
        db_service = get_database_service()
        # Simple query to check connection
        async with db_service.get_session() as session:
            await session.execute(text("SELECT 1"))
        services["database"] = "healthy"
    except Exception as e:
        services["database"] = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        redis_service = get_redis_service()
        await redis_service.connect()
        if redis_service.redis_client:
            await redis_service.redis_client.ping()
            services["redis"] = "healthy"
        else:
            services["redis"] = "unhealthy: not connected"
    except Exception as e:
        services["redis"] = f"unhealthy: {str(e)}"
    
    # Check vector database
    try:
        from app.services.vector_db import VectorDBFactory
        vector_db = VectorDBFactory.get_vector_db()
        services["vector_db"] = f"configured: {settings.VECTOR_DB_TYPE}"
    except Exception as e:
        services["vector_db"] = f"unhealthy: {str(e)}"
    
    return HealthResponse(
        status="healthy" if all("healthy" in v for v in services.values()) else "degraded",
        timestamp=datetime.utcnow(),
        version=settings.APP_VERSION,
        services=services
    )


# Root endpoint
@app.get(
    "/",
    tags=["root"],
    summary="API Root",
    description="Get basic information about the API"
)
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# Include routers
app.include_router(documents.router, prefix=settings.API_V1_PREFIX)
app.include_router(chat.router, prefix=settings.API_V1_PREFIX)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
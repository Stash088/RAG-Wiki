# src/main.py - если хотите чтобы он был основным
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Создаем app здесь
app = FastAPI(
    title="RAG Wikipedia API",
    description="API для RAG системы с Wikipedia, Qdrant и Ollama",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Импортируем и подключаем роутеры
from api.router import router
from api.services import rag_service
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    import time
    
    logger.info("Ждем инициализацию Ollama...")
    time.sleep(1) 
    """Инициализация при запуске приложения"""
    rag_service.init_components()


@app.get("/")
async def root():
    return {"message": "RAG API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
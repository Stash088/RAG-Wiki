from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ============ Запросы ============

class SearchRequest(BaseModel):
    """Запрос на поиск"""
    query: str = Field(..., description="Поисковый запрос", min_length=1)
    limit: int = Field(default=5, ge=1, le=20, description="Количество результатов")


class ArticleRequest(BaseModel):
    """Запрос на получение статьи"""
    title: str = Field(..., description="Название статьи")


class IngestRequest(BaseModel):
    """Запрос на загрузку статей в базу"""
    query: str = Field(..., description="Поисковый запрос для Wikipedia")
    limit: int = Field(default=5, ge=1, le=20, description="Количество статей")


class RAGRequest(BaseModel):
    """Запрос для RAG"""
    question: str = Field(..., description="Вопрос пользователя")
    limit: int = Field(default=3, ge=1, le=10, description="Количество источников")
    temperature: float = Field(default=0.3, ge=0, le=1, description="Температура генерации")


# ============ Ответы ============

class ArticleResponse(BaseModel):
    """Информация о статье"""
    title: str
    url: str = ""
    text_preview: str = ""
    char_count: int = 0


class SearchResult(BaseModel):
    """Результат поиска"""
    id: str
    score: float
    title: str
    text_preview: str = ""


class IngestResult(BaseModel):
    """Результат загрузки статей"""
    query: str
    articles_found: int      
    articles_saved: int  
    saved_ids: List[str] = []


class RAGResponse(BaseModel):
    """Ответ RAG системы"""
    question: str
    answer: str
    sources: List[Dict[str, Any]] = []
    processing_time: float = 0.0


class HealthResponse(BaseModel):
    """Статус здоровья системы"""
    status: str
    embedder: str = "unknown"
    qdrant: str = "unknown"
    llm: str = "unknown"
    wikipedia: str = "unknown"
    stats: Dict[str, Any] = {}


class StatsResponse(BaseModel):
    """Статистика системы"""
    collection: str
    documents: int
    vector_size: int
    embedding_model: str
    llm_model: str


class ClearResponse(BaseModel):
    """Результат очистки базы"""
    success: bool
    message: str
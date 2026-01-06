# src/api/router.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
import time

from clients.wikipediaClient import WikipediaManager
from clients.embedder import Embedder
from database.qdrant import QdrantManager
from clients.llmClient import LLMClient
from api.models import (
        SearchRequest, ArticleRequest, IngestRequest, RAGRequest,
        ArticleResponse, SearchResult, IngestResult, RAGResponse,
        HealthResponse, StatsResponse, ClearResponse
    )
from api.services import rag_service

logger = logging.getLogger(__name__)

# Создаем роутеры
router = APIRouter()

# Зависимости для инъекции компонентов
def get_wikipedia_manager():
    """Зависимость для WikipediaManager - создаем новый экземпляр для каждого запроса"""
    return rag_service.create_wikipedia_manager(language="ru")


def get_embedder():
    """Зависимость для Embedder - используем общий экземпляр"""
    return rag_service.get_component('embedder')


def get_qdrant():
    """Зависимость для QdrantManager - используем общий экземпляр"""
    return rag_service.get_component('qdrant')


def get_llm():
    """Зависимость для LLMClient - используем общий экземпляр"""
    return rag_service.get_component('llm')

@router.get("/health", response_model=HealthResponse, tags=["Система"])
async def health_check(
    wikipedia: WikipediaManager = Depends(get_wikipedia_manager),
    embedder: Embedder = Depends(get_embedder),
    qdrant: QdrantManager = Depends(get_qdrant),
    llm: LLMClient = Depends(get_llm)
):
    """Проверка состояния системы"""
    try:
        # Проверяем компоненты
        embedder_status = "healthy" if embedder.test() else "unhealthy"
        
        try:
            qdrant.get_collection_info()
            qdrant_status = "healthy"
        except:
            qdrant_status = "unhealthy"
        
        llm_status = "healthy" if llm.test() else "unhealthy"
        
        try:
            wikipedia.search_articles("test", limit=1)
            wikipedia_status = "healthy"
        except:
            wikipedia_status = "unavailable"
        
        # Статистика
        stats = qdrant.get_collection_info()
        
        return HealthResponse(
            status="healthy" if all(s == "healthy" for s in [embedder_status, qdrant_status, llm_status]) else "degraded",
            embedder=embedder_status,
            qdrant=qdrant_status,
            llm=llm_status,
            wikipedia=wikipedia_status,
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"Ошибка health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=List[ArticleResponse], tags=["Wikipedia"])
async def search_wikipedia(
    request: SearchRequest,
    wikipedia: WikipediaManager = Depends(get_wikipedia_manager)
):
    """Поиск статей в Wikipedia"""
    try:
        results = wikipedia.search_articles(
            request.query, 
            limit=request.limit
        )
        
        articles = []
        for result in results:
            articles.append(ArticleResponse(
                title=result.get('title', ''),
                url=f"https://ru.wikipedia.org/wiki/{result['title'].replace(' ', '_')}",
                text_preview=result.get('snippet', '')[:25000],
                char_count=result.get('wordcount', 0) * 5
            ))
        
        return articles
        
    except Exception as e:
        logger.error(f"Ошибка поиска Wikipedia: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/article", response_model=ArticleResponse, tags=["Wikipedia"])
async def get_article(
    request: ArticleRequest,
    wikipedia: WikipediaManager = Depends(get_wikipedia_manager)
):
    """Получение статьи из Wikipedia"""
    try:
        article = wikipedia.fetch_article(request.title)
        
        if not article:
            raise HTTPException(status_code=404, detail="Статья не найдена")
        
        return ArticleResponse(
            title=article.get('title', request.title),
            url=article.get('url', ''),
            text_preview=article.get('text', '')[:25000],
            char_count=len(article.get('text', ''))
        )
        
    except Exception as e:
        logger.error(f"Ошибка получения статьи: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=IngestResult, tags=["База знаний"])
async def ingest_articles(
    request: IngestRequest,
    wikipedia: WikipediaManager = Depends(get_wikipedia_manager),
    embedder: Embedder = Depends(get_embedder),
    qdrant: QdrantManager = Depends(get_qdrant)
):
    """Загрузка статей в векторную базу Qdrant"""
    try:
        # Поиск статей
        search_results = wikipedia.search_articles(
            request.query, 
            limit=request.limit
        )
        
        if not search_results:
            return IngestResult(
                query=request.query,
                articles_found=0,          
                articles_saved=0,          
                saved_ids=[]
            )
        
        # Загрузка и сохранение
        saved_ids = []
        for result in search_results:
            try:
                article = wikipedia.fetch_article(result['title'])
                if not article or not article.get('text'):
                    continue
                
                # Векторизация
                text_to_embed = f"{article['title']}\n\n{article['text'][:1000]}"
                embedding = embedder.embed(text_to_embed)
                
                # Сохранение
                doc_id = qdrant.store_document(
                    text=article['text'][:25000],
                    embedding=embedding,
                    metadata={
                        'title': article['title'],
                        'url': article.get('url', ''),
                        'source': 'wikipedia',
                        'full_length': len(article.get('text', ''))
                    }
                )
                
                saved_ids.append(doc_id)
                logger.info(f"Сохранена статья: {article['title']}")
                
            except Exception as e:
                logger.error(f"Ошибка при сохранении статьи: {e}")
                continue
        
        return IngestResult(
            query=request.query,
            articles_found=len(search_results),      
            articles_saved=len(saved_ids),           
            saved_ids=saved_ids
        )
        
    except Exception as e:
        logger.error(f"Ошибка ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similar", response_model=List[SearchResult], tags=["Поиск"])
async def find_similar(
    request: SearchRequest,
    embedder: Embedder = Depends(get_embedder),
    qdrant: QdrantManager = Depends(get_qdrant)
):
    """Поиск похожих документов в векторной базе"""
    try:
        # Векторизация запроса
        query_embedding = embedder.embed(request.query)
        
        # Поиск в Qdrant
        results = qdrant.search_similar(
            query_vector=query_embedding,
            limit=request.limit
        )
        
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                id=str(result['id']),
                score=result['score'],
                title=result['payload'].get('title', 'Без названия'),
                text_preview=result['payload'].get('text', '')[:200]
            ))
        
        return search_results
        
    except Exception as e:
        logger.error(f"Ошибка поиска похожих: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag", response_model=RAGResponse, tags=["RAG"])
async def rag_query(
    request: RAGRequest,
    embedder: Embedder = Depends(get_embedder),
    qdrant: QdrantManager = Depends(get_qdrant),
    llm: LLMClient = Depends(get_llm)
):
    """RAG вопрос-ответ с поиском по базе и генерацией ответа"""
    start_time = time.time()
    
    try:
        # 1. Векторизация вопроса
        query_embedding = embedder.embed(request.question)
        
        # 2. Поиск релевантных документов
        results = qdrant.search_similar(
            query_vector=query_embedding,
            limit=request.limit,
            score_threshold=0.6
        )
        
        if not results:
            return RAGResponse(
                question=request.question,
                answer="К сожалению, в базе знаний нет информации по этому вопросу.",
                sources=[],
                processing_time=time.time() - start_time
            )
        
        # 3. Формирование контекста
        context_parts = []
        sources = []

        filtered_results = [r for r in results if r['score'] > 0.6]
        
        for result in filtered_results:
            title = result['payload'].get('title', 'Источник')
            text = result['payload'].get('text', '')
            
            context_parts.append(f"[{title}]\n{text}")
            sources.append({
                'title': title,
                'url': result['payload'].get('url', ''),
                'similarity': result['score']
            })
        
        context = "\n\n".join(context_parts)
        
        # 4. Формирование промпта
        prompt = f"""Ты — экспертный помощник, отвечающий на вопросы на основе предоставленного контекста.

КОНТЕКСТ:
{context}

ВОПРОС: {request.question}

ИНСТРУКЦИИ:
1. Тщательно проанализируй контекст и найди всю релевантную информацию
2. Сформулируй развернутый, информативный ответ на основе найденной информации
3. Если информации недостаточно, укажи это, но все равно попробуй дать максимально полный ответ
4. Структурируй ответ логически, используй абзацы и маркеры где это уместно
5. Будь точным и информативным

ОТВЕТ:"""
        
        # 5. Генерация ответа через LLM
        answer = llm.generate(
            prompt=prompt,
            temperature=request.temperature,
            max_tokens=500
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"RAG ответ сгенерирован за {processing_time:.2f}с")
        
        return RAGResponse(
            question=request.question,
            answer=answer.strip(),
            sources=sources,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Ошибка RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse, tags=["База знаний"])
async def get_stats(qdrant: QdrantManager = Depends(get_qdrant)):
    """Получение статистики по векторной базе"""
    try:
        info = qdrant.get_collection_info()
        
        return StatsResponse(
            collection=qdrant.collection_name,
            documents=info.get('points_count', 0),
            vector_size=info.get('vector_size', 0),
            embedding_model="nomic-embed-text:latest",
            llm_model="qwen2.5:0.5b"
        )
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear", response_model=ClearResponse, tags=["База знаний"])
async def clear_database(qdrant: QdrantManager = Depends(get_qdrant)):
    """Очистка векторной базы данных"""
    try:
        success = qdrant.clear_collection()
        
        return ClearResponse(
            success=success,
            message="База данных очищена" if success else "Не удалось очистить базу данных"
        )
            
    except Exception as e:
        logger.error(f"Ошибка очистки базы: {e}")
        raise HTTPException(status_code=500, detail=str(e))
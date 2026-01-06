# clients/embedder.py
import ollama
from ollama import Client
import logging
import numpy as np
from typing import List, Optional
from functools import lru_cache
import time
import os

logger = logging.getLogger(__name__)


class Embedder:
    """
    Векторизатор для текста с использованием модели nomic-embed-text через Ollama.
    Размерность: 768, хорошо работает с русским языком.
    """
    
    # Константы для nomic-embed-text
    EMBEDDING_DIM = 768
    MODEL_NAME = "nomic-embed-text"
    RECOMMENDED_TEXT_LIMIT = 8192
    
    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",  # ← ДОБАВИЛИ явный хост
        timeout: int = 30,
        max_retries: int = 2,
        cache_size: int = 1000,
        normalize: bool = True,
        batch_size: int = 5
    ):
        """
        Инициализация векторизатора с моделью nomic-embed-text.
        
        Args:
            host: URL Ollama сервера (по умолчанию 127.0.0.1 для избежания IPv6)
            timeout: Таймаут запроса в секундах
            max_retries: Количество попыток при ошибках
            cache_size: Размер кэша LRU
            normalize: Нормализовать векторы
            batch_size: Размер батча
        """
        self.host = host
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_size = cache_size
        self.normalize = normalize
        self.batch_size = batch_size
        
        # ← КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: создаём клиент с явным хостом
        self.client = Client(host=host)
        
        # Настройка кэша
        self._setup_cache()
        
        # Проверяем доступность модели
        self._verify_model()
        
        logger.info(f"✅ Векторизатор инициализирован с моделью: {self.MODEL_NAME}")
        logger.info(f"   Хост Ollama: {host}")
        logger.info(f"   Размерность векторов: {self.EMBEDDING_DIM}")
        logger.info(f"   Нормализация: {normalize}")
        logger.info(f"   Размер кэша: {cache_size} записей")
    
    def _setup_cache(self):
        """Настройка LRU кэша для эмбеддингов"""
        self._embed_cache = {}
        
        @lru_cache(maxsize=self.cache_size)
        def cached_embed(text: str) -> Optional[tuple]:
            """Внутренняя функция с кэшированием"""
            return self._embed_uncached(text)
        
        self.cached_embed_func = cached_embed
    
    def _verify_model(self):
        """Проверка что модель nomic-embed-text доступна"""
        try:
            # Простой тестовый запрос
            test_result = self.embed("тест", use_cache=False)
            
            if len(test_result) != self.EMBEDDING_DIM:
                logger.warning(
                    f"Размерность эмбеддинга ({len(test_result)}) "
                    f"не соответствует ожидаемой ({self.EMBEDDING_DIM})"
                )
            
            logger.debug(f"✅ Модель {self.MODEL_NAME} доступна")
            return True
            
        except Exception as e:
            logger.error(f"❌ Модель {self.MODEL_NAME} недоступна: {e}")
            logger.info(f"Установите модель: ollama pull {self.MODEL_NAME}")
            raise ConnectionError(f"Модель {self.MODEL_NAME} недоступна")
    
    def _embed_uncached(self, text: str) -> Optional[tuple]:
        """Создание эмбеддинга без кэширования"""
        if not text or not text.strip():
            return None
        
        cleaned_text = text.strip()
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # ← ИЗМЕНЕНО: используем self.client вместо ollama
                response = self.client.embeddings(
                    model=self.MODEL_NAME,
                    prompt=cleaned_text
                )
                
                embedding = response.get('embedding', [])
                
                if not embedding:
                    raise ValueError("Пустой ответ от модели")
                
                if self.normalize and embedding:
                    embedding = self._normalize_vector(embedding)
                
                elapsed = time.time() - start_time
                
                if elapsed > 1.0:
                    logger.debug(f"Долгий эмбеддинг: {elapsed:.2f}с, текст: {len(cleaned_text)} символов")
                
                return tuple(embedding)
                
            except Exception as e:
                wait_time = 1.5 ** attempt
                logger.warning(
                    f"Попытка {attempt + 1}/{self.max_retries} не удалась: {str(e)[:100]}"
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error(f"Не удалось создать эмбеддинг после {self.max_retries} попыток")
                    return None
        
        return None
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Нормализация вектора"""
        if not vector:
            return vector
        
        np_vector = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(np_vector)
        
        if norm == 0 or np.isnan(norm):
            return vector
        
        normalized = np_vector / norm
        return normalized.tolist()
    
    def embed(self, text: str, use_cache: bool = True) -> List[float]:
        """Создание эмбеддинга для текста."""
        if not text or not text.strip():
            logger.warning("Пустой текст, возвращаю нулевой вектор")
            return [0.0] * self.EMBEDDING_DIM
        
        cleaned_text = text.strip()
        
        try:
            if use_cache:
                cached_result = self.cached_embed_func(cleaned_text)
                
                if cached_result is None:
                    logger.debug(f"Не удалось создать эмбеддинг для текста: {cleaned_text[:50]}...")
                    return [0.0] * self.EMBEDDING_DIM
                
                return list(cached_result)
            else:
                result = self._embed_uncached(cleaned_text)
                return list(result) if result else [0.0] * self.EMBEDDING_DIM
                
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга: {e}")
            return [0.0] * self.EMBEDDING_DIM
    
    def embed_batch(
        self, 
        texts: List[str], 
        use_cache: bool = True,
        show_progress: bool = False
    ) -> List[List[float]]:
        """Создание эмбеддингов для списка текстов."""
        if not texts:
            return []
        
        embeddings = []
        valid_texts = [(i, text.strip()) for i, text in enumerate(texts) if text and text.strip()]
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(valid_texts, desc="Создание эмбеддингов", unit="текст")
            except ImportError:
                iterator = valid_texts
        else:
            iterator = valid_texts
        
        for idx, text in iterator:
            embedding = self.embed(text, use_cache=use_cache)
            embeddings.append(embedding)
        
        return embeddings
    
    def truncate_text(self, text: str, max_chars: int = 4000) -> str:
        """Обрезка текста до безопасного размера."""
        if len(text) <= max_chars:
            return text
        
        logger.debug(f"Текст обрезан с {len(text)} до {max_chars} символов")
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.8:
            return truncated[:last_period + 1]
        
        return truncated
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Вычисление косинусного сходства."""
        if len(vec1) != len(vec2):
            raise ValueError(f"Векторы разной длины: {len(vec1)} vs {len(vec2)}")
        
        a = np.array(vec1, dtype=np.float32)
        b = np.array(vec2, dtype=np.float32)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        return float(np.clip(similarity, -1.0, 1.0))
    
    def get_cache_info(self) -> dict:
        """Получение информации о кэше"""
        cache_info = self.cached_embed_func.cache_info()
        return {
            'hits': cache_info.hits,
            'misses': cache_info.misses,
            'maxsize': cache_info.maxsize,
            'currsize': cache_info.currsize,
            'hit_ratio': cache_info.hits / max(cache_info.hits + cache_info.misses, 1)
        }
    
    def clear_cache(self):
        """Очистка кэша"""
        self.cached_embed_func.cache_clear()
        logger.info("Кэш эмбеддингов очищен")
    



def create_embedder(
    host: str = "http://127.0.0.1:11434",
    normalize: bool = True,
    cache_size: int = 1000
) -> Embedder:
    """Создание векторизатора с настройками по умолчанию."""
    return Embedder(
        host=host,
        normalize=normalize,
        cache_size=cache_size
    )
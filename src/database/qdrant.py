from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional, Any, Union
import logging
import os
from dotenv import load_dotenv
from uuid import uuid4
import json

load_dotenv()

logger = logging.getLogger(__name__)






class QdrantManager:
    """
    Менеджер для работы с векторной базой данных Qdrant.
    Обеспечивает CRUD операции для векторов и метаданных.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: str = "wikipedia_rag",
        vector_size: int = 768,  # для nomic-embed-text
        recreate_collection: bool = False
    ):
        """
        Инициализация клиента Qdrant.
        
        Args:
            host: Хост Qdrant (по умолчанию из .env)
            port: Порт Qdrant (по умолчанию из .env)
            collection_name: Название коллекции
            vector_size: Размерность векторов
            recreate_collection: Пересоздать коллекцию если существует
        """
        self.host = host or os.getenv("HOST_QDRANT", "localhost")
        self.port = port or int(os.getenv("PORT_QDRANT", 6333))
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "wikipedia_rag")
        self.vector_size = vector_size
        
        # Инициализация клиента
        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            timeout=30  # таймаут для операций
        )
        
        logger.info(f"Qdrant клиент инициализирован: {self.host}:{self.port}")
        
        # Проверка подключения
        if not self._check_connection():
            raise ConnectionError(f"Не удалось подключиться к Qdrant по адресу {self.host}:{self.port}")
        
        # Создание/проверка коллекции
        self._setup_collection(recreate=recreate_collection)
    
    def _check_connection(self) -> bool:
        """Проверка подключения к Qdrant"""
        try:
            # Простой запрос для проверки связи
            collections = self.client.get_collections()
            logger.info(f"✅ Подключение к Qdrant успешно. Коллекций: {len(collections.collections)}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Qdrant: {e}")
            return False
    
    def _setup_collection(self, recreate: bool = False):
        """Создание или проверка коллекции"""
        try:
            # Получаем список коллекций
            collections = self.client.get_collections()
            collection_exists = any(
                c.name == self.collection_name 
                for c in collections.collections
            )
            
            if recreate and collection_exists:
                logger.info(f"Удаление существующей коллекции: {self.collection_name}")
                self.client.delete_collection(collection_name=self.collection_name)
                collection_exists = False
            
            if not collection_exists:
                # Создаем новую коллекцию
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=2,  # оптимизация для маленьких коллекций
                        indexing_threshold=0  # индексировать все векторы сразу
                    )
                )
                logger.info(f"✅ Коллекция создана: {self.collection_name}")
            else:
                logger.info(f"✅ Коллекция уже существует: {self.collection_name}")
                
                # Проверяем параметры коллекции
                collection_info = self.client.get_collection(self.collection_name)
                actual_size = collection_info.config.params.vectors.size
                
                if actual_size != self.vector_size:
                    logger.warning(
                        f"⚠️ Размерность коллекции ({actual_size}) "
                        f"не соответствует ожидаемой ({self.vector_size})"
                    )
                    
        except Exception as e:
            logger.error(f"❌ Ошибка при настройке коллекции: {e}")
            raise
    
    def store_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[Union[str, int]]] = None
    ) -> List[Union[str, int]]:
        """
        Сохранение векторов в коллекцию.
        
        Args:
            vectors: Список векторов
            payloads: Список метаданных для каждого вектора
            ids: Список ID (опционально, сгенерируются автоматически)
            
        Returns:
            Список ID сохраненных векторов
        """
        if len(vectors) != len(payloads):
            raise ValueError("Количество векторов и payload должно совпадать")
        
        # Генерируем ID если не предоставлены
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]
        elif len(ids) != len(vectors):
            raise ValueError("Количество ID должно совпадать с количеством векторов")
        
        # Преобразуем векторы в нужный формат
        points = []
        for vector, payload, point_id in zip(vectors, payloads, ids):
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )
        
        try:
            # Сохраняем векторы
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True  # ждем завершения операции
            )
            
            logger.info(f"✅ Сохранено векторов: {len(points)}")
            logger.debug(f"Статус операции: {operation_info.status}")
            
            return ids
            
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении векторов: {e}")
            raise
    
    def store_document(
        self,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> str:
        """
        Сохранение одного документа.
        
        Args:
            text: Текст документа
            embedding: Векторное представление
            metadata: Дополнительные метаданные
            document_id: ID документа (опционально)
            
        Returns:
            ID сохраненного документа
        """
        if document_id is None:
            document_id = str(uuid4())
        
        payload = {
            'text': text,
            **metadata
        }
        
        self.store_vectors(
            vectors=[embedding],
            payloads=[payload],
            ids=[document_id]
        )
        
        return document_id
    
    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[models.Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск похожих векторов.
        
        Args:
            query_vector: Вектор запроса
            limit: Количество результатов
            score_threshold: Порог сходства (0.0-1.0)
            filter_conditions: Условия фильтрации
            
        Returns:
            Список найденных документов с оценкой сходства
        """
        try:
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions,
                with_payload=True,
                with_vectors=False  # экономим трафик, обычно не нужны
            )
            
            # Преобразуем в удобный формат
            results = []
            for result in search_results.points:
                results.append({
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload,
                    'text': result.payload.get('text', '')
                })
            
            logger.debug(f"Найдено результатов: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка при поиске: {e}")
            return []
    
    def search_by_text(
        self,
        text: str,
        embedding_func,
        limit: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Поиск по текстовому запросу.
        
        Args:
            text: Текст запроса
            embedding_func: Функция для создания эмбеддинга
            limit: Количество результатов
            **kwargs: Дополнительные параметры для search_similar
            
        Returns:
            Список найденных документов
        """
        # Создаем эмбеддинг для текста запроса
        query_vector = embedding_func(text)
        
        # Ищем похожие векторы
        return self.search_similar(
            query_vector=query_vector,
            limit=limit,
            **kwargs
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Получение информации о коллекции"""
        try:
            info = self.client.get_collection(self.collection_name)
            
            return {
                'name': info.config.params.vectors.size,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance,
                'points_count': info.points_count,
                'segments_count': info.segments_count,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о коллекции: {e}")
            return {}
    
    def get_document(self, document_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Получение документа по ID"""
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[document_id],
                with_payload=True,
                with_vectors=False
            )
            
            if points and len(points) > 0:
                point = points[0]
                return {
                    'id': point.id,
                    'payload': point.payload
                }
            return None
            
        except Exception as e:
            logger.error(f"Ошибка при получении документа: {e}")
            return None
    
    def delete_document(self, document_id: Union[str, int]) -> bool:
        """Удаление документа по ID"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[document_id]
                )
            )
            logger.info(f"Документ удален: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при удалении документа: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Очистка всей коллекции"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Коллекция очищена: {self.collection_name}")
            
            # Создаем заново
            self._setup_collection()
            return True
        except Exception as e:
            logger.error(f"Ошибка при очистке коллекции: {e}")
            return False
    
    def create_index(
        self, 
        field_name: str, 
        field_schema: Optional[models.PayloadSchemaType] = None
    ):
        """
        Создание индекса для полей payload.
        
        Args:
            field_name: Название поля для индексации
            field_schema: Тип поля (keyword, integer, float, etc.)
        """
        if field_schema is None:
            field_schema = models.PayloadSchemaType.TEXT
        
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema
            )
            logger.info(f"Индекс создан для поля: {field_name}")
        except Exception as e:
            logger.error(f"Ошибка при создании индекса: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики по коллекции"""
        try:
            info = self.get_collection_info()
            
            # Получаем кластерную информацию если доступно
            cluster_info = {}
            try:
                cluster_info = self.client.cluster_info()
            except:
                pass
            
            return {
                'collection': info,
                'cluster': cluster_info,
                'aliases': self.client.get_aliases(),
                'total_collections': len(self.client.get_collections().collections)
            }
        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {e}")
            return {}
    
    def batch_ingest(
        self,
        documents: List[Dict[str, Any]],
        embedding_func,
        batch_size: int = 50,
        show_progress: bool = True
    ) -> List[str]:
        """
        Пакетная загрузка документов.
        
        Args:
            documents: Список документов
            embedding_func: Функция для создания эмбеддингов
            batch_size: Размер батча
            show_progress: Показывать прогресс-бар
            
        Returns:
            Список ID сохраненных документов
        """
        from tqdm import tqdm
        
        all_ids = []
        
        if show_progress:
            iterator = tqdm(
                range(0, len(documents), batch_size),
                desc="Загрузка документов в Qdrant"
            )
        else:
            iterator = range(0, len(documents), batch_size)
        
        for i in iterator:
            batch = documents[i:i + batch_size]
            
            vectors = []
            payloads = []
            
            for doc in batch:
                # Создаем эмбеддинг для документа
                text_to_embed = f"{doc.get('title', '')}\n\n{doc.get('text', '')}"
                embedding = embedding_func(text_to_embed[:2000])  # ограничиваем длину
                
                vectors.append(embedding)
                payloads.append({
                    'text': doc.get('text', ''),
                    'title': doc.get('title', ''),
                    'url': doc.get('url', ''),
                    'source': doc.get('source', 'wikipedia'),
                    'metadata': json.dumps(doc.get('metadata', {}))
                })
            
            # Сохраняем батч
            batch_ids = self.store_vectors(vectors, payloads)
            all_ids.extend(batch_ids)
        
        logger.info(f"✅ Загружено документов: {len(all_ids)}")
        return all_ids


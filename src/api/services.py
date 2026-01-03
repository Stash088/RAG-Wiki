import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RAGService:
    """Singleton сервис для управления всеми RAG компонентами"""
    
    _instance = None
    _components: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def init_components(cls, force_reinit: bool = False):
        """
        Инициализация всех RAG компонентов.
        
        Args:
            force_reinit: Принудительная переинициализация даже если уже инициализировано
        """
        if cls._components and not force_reinit:
            logger.debug("Компоненты уже инициализированы, пропускаем")
            return
        
        try:
            logger.info("🚀 Инициализация RAG компонентов...")
            
            # Импортируем все компоненты
            from clients.wikipediaClient import WikipediaManager
            from clients.embedder import Embedder
            from database.qdrant import QdrantManager
            from clients.llmClient import LLMClient
            
            # 1. Векторизатор (nomic-embed-text:latest)
            logger.info("1. Инициализация векторизатора...")
            embedder = Embedder()
            logger.info("   ✅ Векторизатор готов")
            
            # 2. Векторная база данных (Qdrant)
            logger.info("2. Инициализация Qdrant...")
            qdrant = QdrantManager(
                collection_name="wikipedia_rag",
                vector_size=embedder.EMBEDDING_DIM
            )
            info = qdrant.get_collection_info()
            logger.info(f"   ✅ Qdrant подключен. Документов: {info.get('points_count', 0)}")
            
            # 3. LLM модель (qwen2.5:0.5b)
            logger.info("3. Инициализация LLM...")
            llm = LLMClient(model_name="qwen2.5:0.5b")
            if not llm.test():
                logger.warning("   ⚠️ LLM модель может работать некорректно")
            else:
                logger.info(f"   ✅ LLM модель {llm.model_name} готова")
            
            # 4. Wikipedia клиент (легкий, создается при каждом запросе, но храним класс)
            logger.info("4. Инициализация Wikipedia клиента...")
            wikipedia_class = WikipediaManager  # Сохраняем класс, а не экземпляр
            
            # Сохраняем компоненты
            cls._components = {
                'embedder': embedder,
                'qdrant': qdrant,
                'llm': llm,
                'wikipedia_class': wikipedia_class,  # Класс для создания экземпляров
            }
            
            logger.info("✅ Все RAG компоненты инициализированы успешно!")
            
        except ImportError as e:
            logger.error(f"❌ Ошибка импорта компонентов: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации компонентов: {e}")
            raise
    
    @classmethod
    def get_component(cls, name: str) -> Optional[Any]:
        """Получение компонента по имени"""
        if not cls._components:
            cls.init_components()
        return cls._components.get(name)
    
    @classmethod
    def get_all_components(cls) -> Dict[str, Any]:
        """Получение всех компонентов"""
        if not cls._components:
            cls.init_components()
        return cls._components
    
    @classmethod
    def create_wikipedia_manager(cls, language: str = "ru") -> Any:
        """
        Создание нового экземпляра WikipediaManager.
        
        Args:
            language: Язык Wikipedia
            
        Returns:
            Экземпляр WikipediaManager
        """
        if 'wikipedia_class' not in cls._components:
            cls.init_components()
        
        wikipedia_class = cls._components['wikipedia_class']
        return wikipedia_class(
            language=language,
            user_agent="RAG-API/1.0"
        )
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Получение статистики по всем компонентам"""
        try:
            components = cls.get_all_components()
            
            # Статистика Qdrant
            qdrant_stats = components['qdrant'].get_collection_info()
            
            # Информация о моделях
            embedder_info = {
                'model': components['embedder'].MODEL_NAME,
                'dimension': components['embedder'].EMBEDDING_DIM
            }
            
            llm_info = {
                'model': components['llm'].model_name
            }
            
            return {
                'qdrant': qdrant_stats,
                'embedder': embedder_info,
                'llm': llm_info,
                'total_components': len(components)
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}
    
    @classmethod
    def clear_cache(cls):
        """Очистка кэша компонентов (для тестирования)"""
        cls._components.clear()
        logger.info("Кэш компонентов очищен")


# Глобальный экземпляр сервиса
rag_service = RAGService()
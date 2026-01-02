import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv 
from urllib.parse import quote

logger = logging.getLogger(__name__)



load_dotenv()

class WikipediaManager:
    """
    Менеджер для работы с Wikipedia через прямое API.
    Поддерживает поиск, получение статей и обработку ошибок.
    """
    
    def __init__(
        self, 
        user_agent: str = "WikiRAG/1.0 (https://github.com/yourusername/rag-project; contact@example.com)",
        language: str = os.getenv("LANGUAGE_WIKIPEDIA", ""),
        cache_days: int = 30,
        timeout: int = 10
    ):
        """
        Инициализация менеджера Wikipedia.
        
        Args:
            user_agent: User-Agent для запросов (обязательно для Wikipedia API)
            language: Язык Wikipedia (ru, en, etc.)
            cache_days: Количество дней для кэширования
            timeout: Таймаут запросов в секундах
        """
        self.user_agent = user_agent
        self.language = language
        self.cache_days = cache_days
        self.timeout = timeout
        
        # Базовый URL для Wikipedia API
        self.base_url = os.getenv("URL_WIKIPEDIA_API", "")
        
        # Стандартные заголовки
        self.headers = {
            'User-Agent': user_agent,
            'Accept': 'application/json'
        }
        
        # Кэш для уже полученных статей (опционально, можно использовать Redis)
        self._cache = {}
        
        logger.info(f"WikipediaManager initialized for {language}.wikipedia.org")
    
    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Выполнение HTTP запроса к Wikipedia API"""
        try:
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout: {params.get('titles', 'Unknown')}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
    
    def fetch_article(self, title: str, include_metadata: bool = True) -> Optional[Dict[str, Any]]:
        """
        Получение статьи по названию с дополнительной информацией.
        
        Args:
            title: Название статьи
            include_metadata: Включать ли метаданные (дата изменения, категории и т.д.)
            
        Returns:
            Словарь с данными статьи или None если статья не найдена
        """
        logger.info(f"Fetching article: {title}")
        
        # Проверяем кэш
        cache_key = f"article_{title}"
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if self._is_cache_valid(cached_data.get('timestamp', 0)):
                logger.debug(f"Using cached article: {title}")
                return cached_data['data']
        
        # Параметры для получения контента
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts|info|categories",
            "explaintext": True,
            "inprop": "url|displaytitle",
            "cllimit": "max",
            "utf8": 1
        }
        
        data = self._make_request(params)
        if not data:
            return None
        
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            logger.warning(f"No pages found for title: {title}")
            return None
        
        # Получаем первую страницу (должна быть только одна)
        page_id, page_data = next(iter(pages.items()))
        
        # Проверяем, существует ли статья
        if page_id == "-1" or "missing" in page_data:
            logger.warning(f"Article not found: {title}")
            return None
        
        # Формируем результат
        article = {
            'title': page_data.get('title', title),
            'pageid': int(page_id),
            'text': page_data.get('extract', ''),
            'url': page_data.get('fullurl', f"https://{self.language}.wikipedia.org/wiki/{quote(title)}"),
            'displaytitle': page_data.get('displaytitle', title)
        }
        
        # Добавляем метаданные если нужно
        if include_metadata:
            article.update({
                'last_edited': page_data.get('touched', None),
                'length': page_data.get('length', 0),
                'categories': [cat['title'] for cat in page_data.get('categories', [])]
            })
        
        # Кэшируем результат
        self._cache[cache_key] = {
            'data': article,
            'timestamp': time.time()
        }
        
        logger.info(f"Article fetched successfully: {title} ({len(article['text'])} chars)")
        return article
    
    def search_articles(self, query: str, limit: int = 5, snippet_limit: int = 200) -> List[Dict[str, Any]]:
        """
        Поиск статей по запросу.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            snippet_limit: Длина сниппета в символах
            
        Returns:
            Список словарей с результатами поиска
        """
        logger.info(f"Searching articles for query: {query}")
        
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srprop": "snippet|titlesnippet",
            "srinfo": "totalhits",
            "srwhat": "text",
            "utf8": 1
        }
        
        data = self._make_request(params)
        if not data:
            return []
        
        search_results = data.get("query", {}).get("search", [])
        results = []
        
        for result in search_results:
            article_info = {
                'title': result.get('title', ''),
                'pageid': result.get('pageid', 0),
                'snippet': result.get('snippet', '')[:snippet_limit],
                'wordcount': result.get('wordcount', 0),
                'timestamp': result.get('timestamp', '')
            }
            results.append(article_info)
        
        logger.info(f"Found {len(results)} articles for query: {query}")
        return results
    
    def search_and_fetch(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Комбинированный метод: поиск и загрузка полных статей.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество статей для загрузки
            
        Returns:
            Список полных статей
        """
        logger.info(f"Search and fetch for query: {query}")
        
        # Сначала ищем статьи
        search_results = self.search_articles(query, limit=limit)
        if not search_results:
            return []
        
        # Затем загружаем полные статьи
        articles = []
        for result in search_results[:limit]:
            title = result['title']
            article = self.fetch_article(title)
            if article:
                # Добавляем информацию из поиска
                article['search_snippet'] = result.get('snippet', '')
                articles.append(article)
            
            # Небольшая задержка чтобы не перегружать API
            time.sleep(0.1)
        
        logger.info(f"Fetched {len(articles)} full articles for query: {query}")
        return articles
    
    def get_article_sections(self, title: str) -> List[Dict[str, str]]:
        """
        Получение структуры статьи (разделы).
        
        Args:
            title: Название статьи
            
        Returns:
            Список разделов с заголовками и текстом
        """
        article = self.fetch_article(title, include_metadata=False)
        if not article or not article['text']:
            return []
        
        text = article['text']
        sections = []
        
        # Простой парсинг разделов (Wikipedia использует ==Заголовок==)
        lines = text.split('\n')
        current_section = {'title': 'Введение', 'text': ''}
        
        for line in lines:
            line_stripped = line.strip()
            
            # Определяем заголовок раздела (== Заголовок ==)
            if line_stripped.startswith('==') and line_stripped.endswith('=='):
                # Сохраняем предыдущий раздел
                if current_section['text'].strip():
                    sections.append(current_section)
                
                # Извлекаем заголовок (убираем ==)
                section_title = line_stripped.strip('= ')
                current_section = {'title': section_title, 'text': ''}
            else:
                current_section['text'] += line + '\n'
        
        # Добавляем последний раздел
        if current_section['text'].strip():
            sections.append(current_section)
        
        return sections
    
    def get_related_articles(self, title: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Получение связанных статей (через категории).
        
        Args:
            title: Название статьи
            limit: Максимальное количество связанных статей
            
        Returns:
            Список связанных статей
        """
        article = self.fetch_article(title)
        if not article:
            return []
        
        # Берем первую категорию для поиска похожих статей
        categories = article.get('categories', [])
        if not categories:
            return []
        
        # Используем первую категорию для поиска
        category = categories[0].replace('Категория:', '')
        return self.search_articles(category, limit=limit)
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """
        Проверка валидности кэша.
        
        Args:
            timestamp: Время создания кэша
            
        Returns:
            True если кэш еще действителен
        """
        if self.cache_days <= 0:
            return False
        
        cache_age_days = (time.time() - timestamp) / (60 * 60 * 24)
        return cache_age_days < self.cache_days
    
    def clear_cache(self):
        """Очистка кэша"""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Получение информации о кэше"""
        return {
            'size': len(self._cache),
            'cache_days': self.cache_days,
            'cached_articles': list(self._cache.keys())
        }

from clients.WikipediaClient import WikipediaManager
import logging


wiki_manager = WikipediaManager()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

    # 1. Поиск статей
print("\n1. Поиск статей:")
search_results = wiki_manager.search_articles("искусственный интеллект", limit=3)
for i, result in enumerate(search_results, 1):
    print(f"  {i}. {result['title']} - {result['snippet'][:100]}...")
    
# 2. Получение полной статьи
print("\n2. Получение полной статьи:")
article = wiki_manager.fetch_article("искусственный интеллект")
if article:
    print(f"  Заголовок: {article['title']}")
    print(f"  URL: {article['url']}")
    print(f"  Длина текста: {len(article['text'])} символов")
    print(f"  Первые 150 символов: {article['text'][:150]}...")
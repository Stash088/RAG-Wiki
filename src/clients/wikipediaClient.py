import requests
import os
from dotenv import load_dotenv 


load_dotenv()



def fetch_wikipedia_article(title):

    API_URL = os.getenv("URL_WIKIPEDIA_API", "")
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    headers = {
        'User-Agent': 'MyWikipediaBot/1.0 (https://myproject.org; amir@example.com)'
    }
    
    try:
        response = requests.get(API_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        if not pages:
            return "Статья не найдена"
         
        page = next(iter(pages.values()))
        
        if "missing" in page:
            return "Статья не существует"
            
        return page.get("extract", "Содержание недоступно")
        
    except requests.exceptions.RequestException as e:
        return f"Ошибка сети: {e}"
    except ValueError as e:
        return f"Ошибка декодирования JSON: {e}"

result = fetch_wikipedia_article("Python")



print(result)
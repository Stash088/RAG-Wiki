# clients/llmClient.py
from ollama import Client
import logging
import os 
from dotenv import load_dotenv 
from typing import Dict, List

logger = logging.getLogger(__name__)

load_dotenv()

class LLMClient:
    """Клиент для работы с LLM через Ollama"""
    
    def __init__(
        self, 
        model_name: str = os.getenv("OLLAMA_MODEL","qwen2.5:0.5b"),
        host: str = os.getenv("OLLAMA_HOST","http://127.0.0.1:11434"),
    ):
        self.model_name = model_name
        self.client = Client(host=host)
        self._verify_model()
        logger.info(f"LLM клиент инициализирован: {model_name} @ {host}")
    
    def _verify_model(self):
        """Проверка доступности модели"""
        try:
            response = self.client.list()
            
            # Новая версия ollama возвращает объекты, а не словари
            # Пробуем разные способы доступа
            available = []
            
            # Получаем список моделей
            models_list = getattr(response, 'models', None)
            if models_list is None:
                # Fallback для словаря
                models_list = response.get('models', []) if isinstance(response, dict) else []
            
            for m in models_list:
                # Пробуем получить имя модели разными способами
                if hasattr(m, 'name'):
                    available.append(m.name)
                elif hasattr(m, 'model'):
                    available.append(m.model)
                elif isinstance(m, dict):
                    available.append(m.get('name') or m.get('model', ''))
                else:
                    # Попробуем преобразовать в строку
                    available.append(str(m))
            
            logger.debug(f"Доступные модели: {available}")
            
            # Проверяем наличие нашей модели
            model_found = False
            for model in available:
                if self.model_name in model or model in self.model_name:
                    model_found = True
                    self.model_name = model  # Используем точное имя
                    break
            
            if not model_found:
                logger.warning(f"Модель {self.model_name} не найдена. Доступные: {available}")
                # Ищем похожую модель
                for model in available:
                    if "qwen" in model.lower():
                        self.model_name = model
                        logger.info(f"Используем модель: {self.model_name}")
                        return
                raise ValueError(f"Модель {self.model_name} недоступна. Доступные: {available}")
                
        except Exception as e:
            logger.error(f"Ошибка проверки модели: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 5000
    ) -> str:
        """Генерация текста"""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                }
            )
            
            # Поддержка и объекта, и словаря
            if hasattr(response, 'response'):
                return response.response
            elif isinstance(response, dict):
                return response.get('response', '')
            else:
                return str(response)
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return "Извините, произошла ошибка при генерации ответа."
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2
    ) -> str:
        """Чат-взаимодействие"""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={'temperature': temperature}
            )
            
            # Поддержка и объекта, и словаря
            if hasattr(response, 'message'):
                message = response.message
                if hasattr(message, 'content'):
                    return message.content
                elif isinstance(message, dict):
                    return message.get('content', '')
            elif isinstance(response, dict):
                return response.get('message', {}).get('content', '')
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Ошибка чата: {e}")
            return "Извините, произошла ошибка."
    
    def test(self) -> bool:
        """Тестирование модели"""
        try:
            response = self.generate("Скажи 'привет' одним словом.", temperature=0.1)
            if response and len(response) > 0 and "ошибка" not in response.lower():
                logger.info(f"✅ LLM модель {self.model_name} работает")
                logger.debug(f"   Тестовый ответ: {response[:100]}...")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Тест LLM не пройден: {e}")
            return False
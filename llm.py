"""Step 4: The model. A thin wrapper around the Gemini chat model."""
from typing import List

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Settings

class GeminiModel:
  def __init__(self, settings: Settings):
      self.llm = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.google_api_key,
        temperature=settings.temperature,
        timeout=30,
        max_retries=3,
    )

  def generate(self, messages: List[BaseMessage]) -> str:
      response = self.llm.invoke(messages)
      content = response.content
      # Some versions return a list of content parts instead of a plain string
      if isinstance(content, list):
          return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
      return content
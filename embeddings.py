"""Step 2: Embedding. Turns text into vectors so we can compare meaning."""
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import Settings

class EmbeddingService:
    def __init__(self, settings: Settings):
        self.model = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )

    def embed_text(self, text: str) -> List[float]:
        return self.model.embed_query(text)
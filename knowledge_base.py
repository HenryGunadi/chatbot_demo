"""Step 3: Retrieval. Stores embedded documents and finds the ones closest to the question."""
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from embeddings import EmbeddingService

SAMPLE_DOCUMENTS = [
    "An API key is a secret token that identifies your application to a service. Keep it in a .env file, never in source code.",
    "An embedding is a list of numbers that represents the meaning of a text. Similar texts have vectors that are close together.",
    "A prompt template has fixed instructions plus placeholders, such as {question}, that are filled in at runtime.",
    "LangChain is a framework that connects prompts, models, and tools into a single pipeline.",
    "Temperature controls randomness. Low values give focused answers, high values give more creative answers.",
]


class KnowledgeBase:
    def __init__(self, embedding_service: EmbeddingService, texts: List[str] | None = None):
        self.store = InMemoryVectorStore(embedding=embedding_service.model)
        self.store.add_texts(texts or SAMPLE_DOCUMENTS)

    def search(self, query: str, k: int = 2) -> List[Document]:
        return self.store.similarity_search(query, k=k)
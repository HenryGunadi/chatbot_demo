"""Orchestrator. Wires all steps together: input -> embed/retrieve -> prompt -> model -> answer."""
from dataclasses import dataclass
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import Settings
from embeddings import EmbeddingService
from knowledge_base import KnowledgeBase
from llm import GeminiModel
from prompts import PromptBuilder


@dataclass
class ChatResult:
    """Everything produced at each step, so the UI can show the pipeline to students."""
    question: str
    embedding_preview: List[float]
    embedding_size: int
    retrieved_context: List[str]
    final_prompt: List[BaseMessage]
    answer: str


class Chatbot:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = EmbeddingService(settings)
        self.knowledge_base = KnowledgeBase(self.embedder)
        self.prompt_builder = PromptBuilder()
        self.model = GeminiModel(settings)
        self.history: List[BaseMessage] = []

    def ask(self, question: str) -> ChatResult:
        # 1. Embed the question
        vector = self.embedder.embed_text(question)

        # 2. Retrieve the closest documents
        docs = self.knowledge_base.search(question, k=self.settings.top_k)
        context_texts = [d.page_content for d in docs]

        # 3. Build the prompt
        messages = self.prompt_builder.build(
            question=question,
            context="\n".join(f"- {t}" for t in context_texts),
            history=self.history,
        )

        # 4. Call the model
        answer = self.model.generate(messages)

        # 5. Remember the turn
        self.history.extend([HumanMessage(content=question), AIMessage(content=answer)])

        return ChatResult(
            question=question,
            embedding_preview=vector[:5],
            embedding_size=len(vector),
            retrieved_context=context_texts,
            final_prompt=messages,
            answer=answer,
        )

    def reset(self) -> None:
        self.history.clear()
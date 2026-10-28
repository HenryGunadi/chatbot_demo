"""Step 1: Prompt design. Decides HOW the input is shaped before it reaches the model."""
from typing import List

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class PromptBuilder:
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful teaching assistant. "
        "Answer using the context below when it is relevant. "
        "If the context does not contain the answer, say so, then answer from general knowledge. "
        "Keep answers short and clear."
    )

    def __init__(self, system_prompt: str | None = None):
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt + "\n\nContext:\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

    def build(self, question: str, context: str, history: List[BaseMessage]) -> List[BaseMessage]:
        """Fill the template and return the final list of messages sent to the LLM."""
        return self.template.format_messages(
            question=question,
            context=context or "(no relevant context found)",
            history=history,
        )
"""Central place for configuration. Reads values from the .env file."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    chat_model: str
    embedding_model: str
    temperature: float
    top_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is empty. Fill it in the .env file.")

        return cls(
            google_api_key=api_key,
            chat_model=os.getenv("GEMINI_CHAT_MODEL") or "gemini-2.5-flash",
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL") or "models/gemini-embedding-001",
            temperature=float(os.getenv("TEMPERATURE") or 0.3),
            top_k=int(os.getenv("TOP_K") or 2),
        )
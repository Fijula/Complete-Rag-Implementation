import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


@dataclass
class LLMConfig:
    api_key: str
    model: str


def get_llm_config() -> LLMConfig:
    """Read LLM configuration from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env file or export the variable first."
        )

    return LLMConfig(api_key=api_key, model=model)


def make_chat_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Construct a LangChain ChatOpenAI instance with shared config."""
    cfg = get_llm_config()
    return ChatOpenAI(
        api_key=cfg.api_key,
        model=cfg.model,
        temperature=temperature,
    )





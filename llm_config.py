import os
from dotenv import load_dotenv

load_dotenv()

def get_llm(use_mock: bool = True):
    """
    Returns an LLM instance.
    - If use_mock=True => uses local MockChatModel (free)
    - If use_mock=False => tries OpenAI; if it fails (quota/billing), falls back to MockChatModel
    """
    if use_mock:
        from mock_llm import MockChatModel
        return MockChatModel()

    # Try real OpenAI (may fail if quota/billing not enabled)
    try:
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=api_key)
    except Exception:
        from mock_llm import MockChatModel
        return MockChatModel()

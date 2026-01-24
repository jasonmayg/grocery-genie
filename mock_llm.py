from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

class MockChatModel(BaseChatModel):
    """Free offline mock model that behaves like an LLM for demos."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        content = (
            "Here is a 7-day meal plan (DEMO MODE):\n\n"
            "Day 1: Veggie Pasta (500 cal, $5)\n"
            "Day 2: Lentil Soup (450 cal, $4)\n"
            "Day 3: Tacos (650 cal, $7)\n"
            "Day 4: Chicken Stir Fry (550 cal, $6)\n"
            "Day 5: Veggie Pasta (500 cal, $5)\n"
            "Day 6: Lentil Soup (450 cal, $4)\n"
            "Day 7: Tacos (650 cal, $7)\n\n"
            "Notes: This is a placeholder response when API quota is unavailable."
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self):
        return "mock"

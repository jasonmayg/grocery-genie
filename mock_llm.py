"""Offline mock LLM used by Demo Mode.

Lets anyone run the full Grocery Genie flow (plan -> list -> inventory check ->
store strategy) without an API key or spend. It inspects the prompt to decide
which agent is calling and returns a realistic, canned response for that step.
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

MEAL_PLAN = """## Monday
- **Dinner:** Chicken & bell pepper stir-fry with brown rice — ~650 kcal, ~$4.20/serving

## Tuesday
- **Dinner:** Spinach & egg fried rice — ~550 kcal, ~$2.10/serving *(uses spinach before it spoils)*

## Wednesday
- **Dinner:** Teriyaki salmon, rice & broccoli — ~700 kcal, ~$6.50/serving

## Thursday
- **Dinner:** Tofu & vegetable curry — ~600 kcal, ~$3.00/serving

## Friday
- **Dinner:** Chicken noodle soup — ~550 kcal, ~$3.40/serving

## Saturday
- **Dinner:** Beef & broccoli rice bowl — ~720 kcal, ~$5.80/serving

## Sunday
- **Dinner:** Veggie omelette with yogurt & banana — ~500 kcal, ~$2.30/serving

**Why this fits a $100 budget:** chicken, eggs, rice and spinach already in your
kitchen cover four dinners, so new spend is concentrated on two proteins
(salmon, beef) bought in bulk. Estimated weekly total: **$78**.

*Demo Mode — simulated AI response.*"""

SHOPPING_LIST = """- 1 lb Salmon fillet
- 1 lb Beef sirloin
- 1 block Firm tofu
- 2 heads Broccoli
- 1 pack Rice noodles
- 12 Eggs
- 2 lbs Chicken breast
- 1 jar Curry paste"""

INVENTORY_CHECK = """### 🛒 Final Shopping List (To Buy)
- 1 lb Salmon fillet
- 1 lb Beef sirloin
- 1 block Firm tofu
- 2 heads Broccoli
- 1 pack Rice noodles
- 1 jar Curry paste

### ✅ Covered by Inventory
- Eggs (have 12, need 12)
- Chicken breast (have 2 lb, need 2 lb)
- Brown rice, spinach, bell peppers"""

KEYWORDS = '["Salmon fillet", "Beef sirloin", "Firm tofu", "Broccoli", "Rice noodles"]'

STORE_STRATEGY = """| Item | Recommended Store | Price | Reasoning |
|---|---|---|---|
| Salmon fillet (1 lb) | Costco | $11.99 (Est) | Bulk price beats grocery per lb |
| Beef sirloin (1 lb) | Costco | $9.49 (Est) | Freeze half for next week |
| Firm tofu | Target | $2.29 (Est) | Small quantity, no bulk benefit |
| Broccoli (2 heads) | Target | $3.98 (Est) | Fresh produce, buy as needed |
| Rice noodles | Target | $2.79 (Est) | Pantry staple |
| Curry paste | Target | $3.49 (Est) | Lasts several weeks |

**MATH:** 11.99 + 9.49 + 2.29 + 3.98 + 2.79 + 3.49 = 34.03

# 💵 TOTAL ESTIMATED COST: $34.03
(Split: Costco: $21.48, Target: $12.55)

*Demo Mode — prices are estimates, no live search was run.*"""


def _pick_response(prompt: str) -> str:
    p = prompt.lower()
    if "google shopping from" in p or "json list" in p:
        return KEYWORDS
    if "decisive shopping assistant" in p:
        return STORE_STRATEGY
    if "inventory manager" in p:
        return INVENTORY_CHECK
    if "extracts a shopping list" in p:
        return SHOPPING_LIST
    return MEAL_PLAN


class MockChatModel(BaseChatModel):
    """Free offline mock model that behaves like an LLM for demos."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt = " ".join(str(getattr(m, "content", m)) for m in messages)
        content = _pick_response(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self):
        return "mock"

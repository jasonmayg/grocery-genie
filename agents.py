from langchain_core.prompts import PromptTemplate

class SimpleAgent:
    def __init__(self, llm, prompt: PromptTemplate):
        self.llm = llm
        self.prompt = prompt

    def run(self, inputs: dict) -> str:
        prompt_text = self.prompt.format(**inputs)
        result = self.llm.invoke(prompt_text)
        return result.content if hasattr(result, "content") else str(result)

def meal_planning_agent(llm) -> SimpleAgent:
    prompt = PromptTemplate(
    input_variables=["budget", "cuisine", "diet", "calories"],
    template=(
        "You are a helpful meal planning assistant.\n\n"
        "Create a 7-day meal plan under a ${budget} budget.\n"
        "Daily calorie target: approximately {calories} calories.\n"
        "Cuisine preference: {cuisine}\n"
        "Dietary restriction: {diet}\n\n"
        "For each day, include:\n"
        "- Meal name\n"
        "- Estimated calories\n"
        "- Estimated cost\n\n"
        "Try to keep calories close to the target while staying under budget.\n"
        "Return in a clean bullet format."
    ),
)
    return SimpleAgent(llm, prompt)

def shopping_list_agent(llm) -> SimpleAgent:
    prompt = PromptTemplate(
        input_variables=["meal_plan"],
        template=(
            "You are a grocery shopping assistant.\n\n"
            "Based on this 7-day meal plan:\n"
            "{meal_plan}\n\n"
            "Create a consolidated grocery list grouped by category (Produce, Protein, Pantry, Dairy, Spices/Other).\n"
            "Add simple estimated quantities (e.g., '2 lbs', '1 carton', '3 items') where reasonable.\n"
            "Keep it concise and practical."
        ),
    )
    return SimpleAgent(llm, prompt)

from langchain_core.prompts import PromptTemplate
# Define a simple wrapper class to mimic an "Agent" behavior
class SimpleAgent:
    def __init__(self, llm, prompt_template):
        self.llm = llm
        self.prompt_template = prompt_template

    def run(self, inputs):
        prompt = self.prompt_template.format(**inputs)
        response = self.llm.invoke(prompt)
        return response.content

# --- 1. Meal Planning Agent ---
def meal_planning_agent(llm) -> SimpleAgent:
    prompt = PromptTemplate(
        input_variables=["budget", "cuisine", "diet", "calories", "pantry_items"],
        template=(
            "You are a professional Nutritionist and Budget Chef.\n"
            "Your goal is to plan a healthy 7-day meal plan for a family.\n\n"
            
            "--- INVENTORY (USE THESE FIRST) ---\n"
            "{pantry_items}\n"
            "Prioritize using these ingredients to reduce waste.\n"
            "-----------------------------------\n\n"

            "CONSTRAINTS:\n"
            "- Weekly Budget: ${budget}\n"
            "- Target Daily Calories: ~{calories} kcal\n"
            "- Cuisine Style: {cuisine}\n"
            "- Dietary Restrictions: {diet}\n\n"

            "INSTRUCTIONS:\n"
            "1. Create a diverse plan (don't repeat the same meal every day).\n"
            "2. Provide a specific logical reason why this plan fits the budget.\n"
            "3. Format the output clearly with Day headers (e.g., ## Monday).\n"
            "4. For each meal, explicitly state the estimated cost and calories.\n"
        ),
    )
    return SimpleAgent(llm, prompt)

# --- 2. Shopping List Agent (UPDATED FOR CONSOLIDATION) ---
def shopping_list_agent(llm) -> SimpleAgent:
    prompt = PromptTemplate(
        input_variables=["meal_plan"],
        template=(
            "You are a helpful assistant that extracts a shopping list from a meal plan.\n"
            "Given the following meal plan:\n"
            "{meal_plan}\n\n"
            
            "CRITICAL INSTRUCTION: You MUST consolidate duplicate ingredients.\n"
            "If multiple meals use the same item (e.g., Monday needs '1 lb Chicken' and Tuesday needs '2 lbs Chicken'), SUM their quantities into a single entry.\n"
            "Do NOT list the same ingredient twice.\n\n"

            "Format the final output as a simple, deduped list with total quantities (e.g., - 3 lbs Chicken Breast)."
        ),
    )
    return SimpleAgent(llm, prompt)

# --- 3. Inventory Check Agent (Math Logic) ---
def inventory_check_agent(llm) -> SimpleAgent:
    prompt = PromptTemplate(
        input_variables=["shopping_list", "inventory"],
        template=(
            "You are a Kitchen Inventory Manager with strong math skills.\n"
            "Your task is to reconcile the Needed Shopping List with the Current Inventory.\n\n"
            
            "--- ITEMS NEEDED ---\n"
            "{shopping_list}\n\n"
            
            "--- CURRENT INVENTORY ---\n"
            "{inventory}\n\n"
            
            "MATH INSTRUCTIONS:\n"
            "1. Needed (6) - Have (1) = Buy (5).\n"
            "2. Needed (1) - Have (1) = Buy (0) [Remove from list].\n"
            "3. Output result in two Markdown sections:\n"
            "   ### 🛒 Final Shopping List (To Buy)\n"
            "   ### ✅ Covered by Inventory\n"
        ),
    )
    return SimpleAgent(llm, prompt)

# --- 4. Store Optimization Agent ---
def store_optimization_agent(llm) -> SimpleAgent:
    """
    Analyzes list, uses real-time prices, estimates missing ones, AND calculates the total sum.
    """
    prompt = PromptTemplate(
        input_variables=["shopping_list", "real_time_prices"],
        template=(
            "You are a Decisive Shopping Assistant specialized in the US market.\n"
            "Your goal is to optimize the shopping list and calculate the TOTAL cost.\n\n"
            
            "--- SHOPPING LIST ---\n"
            "{shopping_list}\n\n"
            
            "--- REAL-TIME DATA (From Google Shopping) ---\n"
            "{real_time_prices}\n\n"
            
            "CRITICAL RULES:\n"
            "1. **NO 'N/A':** If real-time price is missing, YOU MUST ESTIMATE based on US market averages.\n"
            "2. **Best Store:** Suggest Costco for bulk, Target/Walmart for others.\n"
            "3. **Price Column:** Must always have a number (e.g. '$5.99' or '$4.50 (Est)').\n\n"
            
            "OUTPUT FORMAT:\n"
            "1. **Strategy Table**: [Item, Recommended Store, Price, Reasoning]\n"
            "2. **MATH CALCULATION**: Sum up all the prices in the table above.\n"
            "3. **FINAL OUTPUT**: Display the total clearly at the bottom like this:\n"
            "   # 💵 TOTAL ESTIMATED COST: $XX.XX\n"
            "   (Split: Costco: $XX, Target: $XX)"
        ),
    )
    return SimpleAgent(llm, prompt)
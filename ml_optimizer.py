import pulp

def optimize_meals_ml(budget, calorie_target, diet, days=7):
    """
    Uses Linear Programming (PuLP) to optimize meal selection based on cost and calories.
    Args:
        budget (float): Weekly budget constraint.
        calorie_target (int): Daily calorie target.
        diet (str): Dietary restriction filter.
        days (int): Number of days to plan for (default 7).
    """
    # 1. Define a Mock Database of potential meals with nutrition/cost info
    # In a real app, this would come from your database or API
    all_meals = [
        {"id": 1, "meal": "Chicken Breast & Rice", "cost": 3.50, "calories": 600, "tags": ["None", "Gluten-Free"]},
        {"id": 2, "meal": "Oatmeal & Berries", "cost": 1.20, "calories": 300, "tags": ["Vegetarian", "Vegan"]},
        {"id": 3, "meal": "Avocado Toast", "cost": 2.50, "calories": 400, "tags": ["Vegetarian", "Vegan"]},
        {"id": 4, "meal": "Steak & Asparagus", "cost": 8.00, "calories": 700, "tags": ["None", "Keto", "Paleo"]},
        {"id": 5, "meal": "Pasta Primavera", "cost": 2.00, "calories": 500, "tags": ["Vegetarian"]},
        {"id": 6, "meal": "Salmon Salad", "cost": 6.50, "calories": 550, "tags": ["None", "Keto", "Gluten-Free"]},
        {"id": 7, "meal": "Tofu Stir Fry", "cost": 3.00, "calories": 450, "tags": ["Vegan", "Vegetarian"]},
        {"id": 8, "meal": "Greek Yogurt Parfait", "cost": 1.80, "calories": 350, "tags": ["Vegetarian", "Gluten-Free"]},
        {"id": 9, "meal": "Lentil Soup", "cost": 1.50, "calories": 300, "tags": ["Vegan", "Vegetarian"]},
        {"id": 10, "meal": "Turkey Sandwich", "cost": 2.80, "calories": 400, "tags": ["None"]},
        {"id": 11, "meal": "Smoothie Bowl", "cost": 4.50, "calories": 350, "tags": ["Vegan", "Vegetarian"]},
        {"id": 12, "meal": "Quinoa Salad", "cost": 3.20, "calories": 400, "tags": ["Vegan", "Gluten-Free"]},
    ]

    # Filter meals based on diet
    if diet != "None":
        candidates = [m for m in all_meals if diet in m["tags"]]
    else:
        candidates = all_meals

    # If filtering leaves too few options, fallback to all
    if len(candidates) < 3:
        candidates = all_meals

    # 2. Setup Linear Programming Problem
    # Goal: Minimize Cost subject to Calorie constraints
    prob = pulp.LpProblem("MealOptimization", pulp.LpMinimize)

    # Decision Variables: How many times to eat each meal per week?
    # Continuous variable (low bound 0), realistically we might want Integer but continuous is faster for demo
    meal_vars = pulp.LpVariable.dicts("Meal", [m["id"] for m in candidates], lowBound=0, cat='Integer')

    # Objective Function: Minimize Total Cost
    prob += pulp.lpSum([meal_vars[m["id"]] * m["cost"] for m in candidates])

    # Constraints
    # 1. Total meals per week should roughly equal 3 meals * 7 days = 21 meals
    # Let's say we are planning mainly for "Dinner" optimization for simplicity, so 7 meals.
    prob += pulp.lpSum([meal_vars[m["id"]] for m in candidates]) == days

    # 2. Average daily calories should meet the target (Total Cals / 7 >= Target)
    # We treat 'calorie_target' as the target for this specific meal slot (e.g. Dinner approx 1/3 of daily)
    # For simplicity in this demo, let's assume calorie_target is passed as the Daily Target, 
    # and we want the average meal to be roughly 1/3 of that.
    per_meal_cal_target = calorie_target / 3.0
    
    total_calories = pulp.lpSum([meal_vars[m["id"]] * m["calories"] for m in candidates])
    prob += total_calories >= (per_meal_cal_target * days) * 0.9  # Lower bound (-10%)
    prob += total_calories <= (per_meal_cal_target * days) * 1.1  # Upper bound (+10%)

    # 3. Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # 4. Format Results
    selected_meals = []
    
    # Check if a valid solution was found
    if pulp.LpStatus[prob.status] == "Optimal":
        for m in candidates:
            qty = meal_vars[m["id"]].varValue
            if qty and qty > 0:
                for _ in range(int(qty)):
                    selected_meals.append({
                        "meal": m["meal"],
                        "cost": m["cost"],
                        "calories": m["calories"]
                    })
        return selected_meals[:days], round(pulp.value(prob.objective), 2)
    else:
        # Fallback if solver fails (e.g. constraints too tight)
        # Create a simple fallback list based on the variable 'days'
        fallback_cost = round(budget / days, 2)
        fallback = [
            {"meal": "Chef's Choice (Solver Fallback)", "cost": fallback_cost, "calories": int(calorie_target/3)} 
            for _ in range(days)
        ]
        return fallback, budget
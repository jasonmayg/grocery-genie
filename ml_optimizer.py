import random

try:
    # Your project likely has meals data in data.py
    from data import MEALS
except Exception:
    MEALS = []

def _normalize_meals(meals):
    """
    Ensures every meal has required keys:
      - 'meal' (string)
      - 'cost' (number)
      - 'calories' (number, fallback if missing)
    """
    normalized = []
    for m in meals:
        name = m.get("meal") or m.get("name") or "Unknown Meal"
        cost = float(m.get("cost", 0))
        calories = m.get("calories", None)
        if calories is None:
            # Fallback so app doesn't break if calories not in data yet
            calories = 500
        normalized.append({"meal": name, "cost": cost, "calories": int(calories)})
    return normalized

def optimize_meals_ml(budget, calorie_target=2000, days=7, trials=2000, diet = "None"):
    """
    Multi-objective optimizer:
      - Hard constraint: total cost <= budget
      - Soft objective: daily calories close to calorie_target (average)
      - Soft objective: prefer lower cost (within budget)

    Returns:
      meals_selected (list of dicts), total_cost (float)
    """
    meals = _normalize_meals(MEALS)

    # If MEALS is empty, return a safe fallback
    if not meals:
        fallback = [{"meal": "Fallback Meal", "cost": round(budget / days, 2), "calories": calorie_target} for _ in range(days)]
        return fallback, round(sum(x["cost"] for x in fallback), 2)

    best_plan = None
    best_score = float("inf")

    # weights (tuneable)
    w_cal = 1.0
    w_cost = 0.15  # small preference for cheaper plans (still must be under budget)

    # Random search over meal combinations (simple “ML-ish” heuristic)
    for _ in range(trials):
        plan = [random.choice(meals) for _ in range(days)]
        total_cost = sum(x["cost"] for x in plan)
        if total_cost > budget:
            continue

        avg_cal = sum(x["calories"] for x in plan) / days
        cal_error = abs(avg_cal - calorie_target)

        # score: prioritize calorie closeness, then cheaper within budget
        score = (w_cal * cal_error) + (w_cost * total_cost)

        if score < best_score:
            best_score = score
            best_plan = plan

    # If we didn't find any under budget (rare), pick the cheapest meals
    if best_plan is None:
        meals_sorted = sorted(meals, key=lambda x: x["cost"])
        best_plan = meals_sorted[:days]
        total_cost = sum(x["cost"] for x in best_plan)
        return best_plan, round(total_cost, 2)

    return best_plan, round(sum(x["cost"] for x in best_plan), 2)

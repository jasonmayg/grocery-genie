import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import json

from llm_config import get_llm
from agents import meal_planning_agent, shopping_list_agent
from ml_optimizer import optimize_meals_ml
from token_tracker import track_tokens

# ---------- Page config ----------
st.set_page_config(page_title="Grocery AI Assistant", layout="wide")
st.title("🛒 Grocery Shopping Assistant / Meal Plan Creator")
st.caption("Streamlit UI • LangChain-style Agents • ML Optimization • Token Tracking")

# ---------- Recipe library metadata ----------
conn = sqlite3.connect("grocery_ai.db")
recipe_count = conn.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
conn.close()

st.caption(f"📚 Recipe Library Size: **{recipe_count:,} recipes** (Kaggle dataset)")

# ---------- Helpers ----------
def parse_pantry(pantry_text: str):
    items = set()
    for part in pantry_text.replace("\n", ",").split(","):
        item = part.strip().lower()
        if item:
            items.add(item)
    return items

def filter_shopping_list(shopping_list_text: str, pantry_items: set):
    lines = shopping_list_text.splitlines()
    if not pantry_items:
        return shopping_list_text, []

    kept, removed = [], []
    for line in lines:
        low = line.lower()
        if any(p in low for p in pantry_items):
            removed.append(line)
        else:
            kept.append(line)

    return "\n".join(kept).strip(), removed

# ---------- Sidebar ----------
st.sidebar.header("Mode")
use_mock = st.sidebar.toggle("Demo Mode (Free / Mock LLM)", value=True)

st.sidebar.header("Preferences")
budget = st.sidebar.slider("Weekly Budget ($)", 30, 150, 60)

calorie_target = st.sidebar.slider(
    "Daily Calorie Target (kcal)",
    1200, 3000, 2000, 100
)

cuisine = st.sidebar.selectbox(
    "Cuisine Preference",
    ["Any", "Asian", "Italian", "Mexican"]
)

diet = st.sidebar.selectbox(
    "Dietary Preference",
    ["None", "Vegetarian", "Vegan"]
)

st.sidebar.header("Pantry (Optional)")
st.sidebar.caption("Items you already have — we’ll try to remove them from the shopping list.")
pantry_text = st.sidebar.text_area(
    "Pantry items (comma or new lines)",
    placeholder="e.g., rice, pasta, olive oil"
)

st.sidebar.divider()
st.sidebar.write("**What you’re demoing:**")
st.sidebar.write("✅ GenAI Agent: Meal Planning")
st.sidebar.write("✅ GenAI Agent: Shopping List")
st.sidebar.write("✅ ML: Budget + Calories optimization")
st.sidebar.write("✅ Token economics (est.)")

# ---------- LLM ----------
llm = get_llm(use_mock=use_mock)

# ---------- Header status ----------
colA, colB = st.columns(2)

with colA:
    st.subheader("Inputs")
    st.write(f"**Budget:** ${budget}")
    st.write(f"**Daily Calories:** {calorie_target} kcal")
    st.write(f"**Cuisine:** {cuisine}")
    st.write(f"**Diet:** {diet}")

with colB:
    st.subheader("System Status")
    st.write("**LLM Mode:** " + ("Demo (Mock)" if use_mock else "Production (Real if available)"))
    st.write("**Agents:** Meal Planning + Shopping List")
    st.write("**ML Layer:** Enabled")

st.divider()

# ---------- Main action ----------
if st.button("Generate Meal Plan & Shopping List"):
    pantry_items = parse_pantry(pantry_text)

    with st.spinner("Generating…"):
        # Agent 1: Meal plan
        plan_agent = meal_planning_agent(llm)
        meal_plan = plan_agent.run({
            "budget": budget,
            "cuisine": cuisine,
            "diet": diet,
            "calories": calorie_target
        })

        # Agent 2: Shopping list
        list_agent = shopping_list_agent(llm)
        shopping_list = list_agent.run({"meal_plan": meal_plan})

        # Pantry filter
        filtered_list, removed_lines = filter_shopping_list(shopping_list, pantry_items)

        # Token tracking (estimate)
        tokens_plan, cost_plan = track_tokens(meal_plan)
        tokens_list, cost_list = track_tokens(shopping_list)
        total_tokens = tokens_plan + tokens_list
        total_cost = round(cost_plan + cost_list, 4)

        # ML optimization (SQL + Kaggle recipes)
        meals, total_cost_ml = optimizea_meals_ml(
            budget,
            calorie_target=calorie_target,
            diet=diet
        )
        df = pd.DataFrame(meals)

    # ---------- Metrics ----------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Budget", f"${budget}")
    m2.metric("ML Meal Cost", f"${total_cost_ml}")
    m3.metric("Tokens (est.)", f"{total_tokens}")
    m4.metric("LLM Cost (est.)", f"${total_cost}")

    tabs = st.tabs([
        "🤖 Meal Plan",
        "🛍 Shopping List",
        "📊 ML Optimization",
        "🔢 Token Details"
    ])

    # ---------- Tabs ----------
    with tabs[0]:
        st.subheader("Meal Plan")
        st.write(meal_plan)

    with tabs[1]:
        st.subheader("Shopping List")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Original")
            st.code(shopping_list)

        with c2:
            st.markdown("### After Pantry Removal")
            st.code(filtered_list or "Everything covered by pantry items!")

        if removed_lines:
            with st.expander("Removed items"):
                for line in removed_lines:
                    st.write(f"- {line}")

    with tabs[2]:
        st.subheader("ML-Optimized Meals (Budget + Calories)")
        st.dataframe(df, use_container_width=True)

        cA, cB = st.columns(2)

        with cA:
            fig, ax = plt.subplots()
            ax.bar(df["meal"], df["cost"])
            ax.set_ylabel("Cost ($)")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with cB:
            fig2, ax2 = plt.subplots()
            ax2.bar(df["meal"], df["calories"])
            ax2.axhline(y=calorie_target, linestyle="--")
            ax2.set_ylabel("Calories (kcal)")
            plt.xticks(rotation=45)
            st.pyplot(fig2)

        avg_cal = int(df["calories"].mean())
        st.info(f"Average daily calories: **{avg_cal} kcal** | Target: **{calorie_target} kcal**")

    with tabs[3]:
        st.subheader("Token & Cost Breakdown")
        st.write(f"Meal plan tokens: **{tokens_plan}** (${cost_plan})")
        st.write(f"Shopping list tokens: **{tokens_list}** (${cost_list})")
        st.write(f"Total tokens: **{total_tokens}** (${total_cost})")

    with st.expander("Presentation-ready explanation"):
        st.write(
            "We ingest a large Kaggle recipe dataset into a SQL database, "
            "optimize meals under budget and calorie constraints using ML, "
            "and orchestrate GenAI agents to generate meal plans and shopping lists, "
            "while tracking token usage to estimate operating cost."
        )

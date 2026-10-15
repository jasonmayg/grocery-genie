import streamlit as st
import pandas as pd
import sqlite3
import json
import time
import ast
import plotly.express as px
from langchain_core.prompts import PromptTemplate
from datetime import datetime

# Import custom modules and agents
from llm_config import get_llm
from agents import (
    SimpleAgent,
    meal_planning_agent, 
    shopping_list_agent, 
    inventory_check_agent, 
    store_optimization_agent
)
from ml_optimizer import optimize_meals_ml
from token_tracker import track_tokens
from vision_agent import analyze_image_stream
from price_fetcher import fetch_grocery_price

# ---------- 0. Page Config & CSS Styling ----------
st.set_page_config(page_title="Team 5 - Grocery AI", layout="wide", page_icon="🥑")

st.markdown("""
<style>
    /* 1. Mobile-first centered layout */
    .block-container {
        max_width: 800px;
        padding-top: 2rem;
        padding-bottom: 5rem;
        margin: auto;
    }
    
    /* 2. Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E0E0E0;
    }
    
    /* 3. Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3rem;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* 4. FORCE IMAGE SIZE */
    [data-testid="stImage"] {
        width: 100%;
    }
    [data-testid="stImage"] > div {
        width: 100%;
        display: flex;
        justify_content: center;
    }
    [data-testid="stImage"] img {
        width: 100% !important;
        max-width: 100% !important;
        height: 300px !important;    /* Force a fixed large height */
        object-fit: contain !important; 
    }
    
    .price-tag {
        font-size: 24px;
        font-weight: bold;
        color: #2E7D32;
        margin: 0;
    }
    
    .source-tag {
        color: #666;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .cal-tag {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    /* Save Button Special Style */
    .save-btn > button {
        background-color: #2E7D32 !important; /* Green */
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


def md_safe(text: str) -> str:
    """Escape '$' so Streamlit doesn't render prices like $5 ... $7 as LaTeX math."""
    return (text or "").replace("\\$", "$").replace("$", "\\$")

# ---------- Session State Initialization ----------
if "gen_meal_plan" not in st.session_state: st.session_state["gen_meal_plan"] = None
if "gen_shopping_list" not in st.session_state: st.session_state["gen_shopping_list"] = None
if "gen_ml_df" not in st.session_state: st.session_state["gen_ml_df"] = None
if "gen_store_strategy" not in st.session_state: st.session_state["gen_store_strategy"] = None
if "gen_logs" not in st.session_state: st.session_state["gen_logs"] = {}
# Store search results (images/links)
if "search_results" not in st.session_state: st.session_state["search_results"] = []

# ---------- Database Initialization ----------
conn = sqlite3.connect("grocery_ai.db")
c = conn.cursor()

# 1. Inventory Table
c.execute('''CREATE TABLE IF NOT EXISTS inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT, item_name TEXT, quantity TEXT, 
    cost REAL, expiry_days INTEGER, nutrition TEXT, date_added DATE DEFAULT (date('now')))''')

# 2. History Table
c.execute('''CREATE TABLE IF NOT EXISTS shopping_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    meal_plan_text TEXT,
    strategy_text TEXT,
    products_json TEXT,
    total_cost REAL
)''')
conn.commit()
conn.close()

# ---------- Sidebar ----------
with st.sidebar:
    # --- BRANDING SECTION ---
    try:
        st.image("WishShopSave.png", use_container_width=True)
    except:
        pass 
        
    st.markdown("""
        <div style="text-align: center; margin-top: 10px;">
            <h2 style="margin: 0; padding: 0;">TEAM 5</h2>
            <span style="color:grey; font-size:14px;">Wish, Shop, Save</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    st.markdown("### 🔌 System")
    use_mock = st.toggle("Demo Mode (Mock AI)", value=False)
    if use_mock:
        st.caption("⚡ Using simulated data (Free)")
    else:
        st.caption("🟢 Using GPT-4o (Real Intelligence)")

    st.divider()

    st.markdown("### 🎯 Preferences")
    budget = st.slider("Weekly Budget", 50, 300, 100, format="$%d")
    
    col_cal, col_diet = st.columns(2)
    with col_cal:
        calorie_target = st.number_input("Kcal/Day", value=2000, step=100)
    with col_diet:
        diet = st.selectbox("Diet", ["None", "Vegan", "Keto", "Paleo"])

    # --- Multi-Select for Cuisine ---
    cuisine_options = ["Asian", "Italian", "Mexican", "American", "Mediterranean", "French", "Indian", "Thai"]
    selected_cuisines = st.multiselect("Cuisine Style", cuisine_options, default=["Asian"])
    cuisine_str = ", ".join(selected_cuisines) if selected_cuisines else "Any"

    st.divider()
    
    with st.expander("🛠️ Advanced / Reset"):
        st.caption("Clear all inventory and chat history.")
        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Reset App Data"):
            conn = sqlite3.connect("grocery_ai.db")
            conn.execute("DELETE FROM inventory")
            conn.commit()
            conn.close()
            for key in st.session_state.keys(): del st.session_state[key]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #aaa; font-size: 12px;">
            Powered by Team 5 AI<br>
            © 2026
        </div>
        """, 
        unsafe_allow_html=True
    )

# ---------- Header ----------
# ---------- Header ----------
# vertical_alignment="center" 是 Streamlit 的新特性，让图片和文字完美垂直居中
c1, c2 = st.columns([1.2, 5], gap="medium", vertical_alignment="center")

with c1: 
    try:
        # width 改成 160-180 看起来会更精致，200可能有点太大喧宾夺主
        st.image("GrocGenie.png", width=160)
    except:
        st.image("https://cdn-icons-png.flaticon.com/512/3081/3081840.png", width=100)

with c2: 
    # 使用 HTML 自定义标题样式
    st.markdown("""
        <style>
            .header-title {
                font-size: 48px !important;
                font-weight: 800 !important;
                color: #2E7D32; /* 品牌绿色 */
                margin-bottom: -10px !important; /* 拉近标题和副标题的距离 */
                line-height: 1.2;
            }
            .header-caption {
                font-size: 18px !important;
                color: #666;
                font-weight: 400;
                margin-top: 5px !important;
            }
            .beta-badge {
                background-color: #E3F2FD;
                color: #1565C0;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 14px;
                vertical-align: middle;
                margin-left: 10px;
            }
        </style>
        
        <div class="header-title">
            Grocery Genie <span class="beta-badge">AI v2.0</span>
        </div>
        <div class="header-caption">
            Your Personal Kitchen Intelligence & Supply Chain Manager
        </div>
    """, unsafe_allow_html=True)

tabs = st.tabs(["🏠 My Kitchen", "📅 Meal Plan", "🛒 Shopping", "📜 History"])

# ==========================================
# TAB 1: 🏠 My Kitchen
# ==========================================
with tabs[0]:
    st.markdown("### 📸 Inventory Tracker")
    with st.form("upload_form", border=False):
        uploaded_img = st.file_uploader("Upload Fridge Photo / Receipt", type=["jpg", "png"], label_visibility="collapsed")
        submitted = st.form_submit_button("✨ Scan & Analyze Image")

        if submitted and uploaded_img:
            if use_mock:
                st.warning("⚠️ Turn off Demo Mode for Vision API.")
            else:
                with st.spinner("🔍 AI is identifying items..."):
                    api_key = st.secrets.get("OPENAI_API_KEY")
                    try:
                        data = analyze_image_stream(uploaded_img, api_key)
                        if data and "items" in data:
                            conn = sqlite3.connect("grocery_ai.db")
                            c = conn.cursor()
                            count = 0
                            for item in data["items"]:
                                c.execute('''INSERT INTO inventory (item_name, quantity, cost, expiry_days, nutrition)
                                    VALUES (?, ?, ?, ?, ?)''', 
                                    (item["item_name"], item["quantity"], item.get("cost", 0), 
                                     item.get("expiry_days", 7), item.get("nutrition", "Other")))
                                count += 1
                            conn.commit()
                            conn.close()
                            st.toast(f"✅ Added {count} items!", icon="🥑")
                    except Exception as e:
                        st.error(f"Vision Error: {str(e)}")
    
    # Inventory Dashboard
    conn = sqlite3.connect("grocery_ai.db")
    df = pd.read_sql("SELECT * FROM inventory ORDER BY id DESC", conn)
    conn.close()
    
    if not df.empty:
        with st.container(border=True):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.metric("Total Items", len(df))
                st.metric("Total Value", f"${df['cost'].sum():.0f}")
            with c2:
                fig = px.pie(df, names='nutrition', hole=0.7, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=100)
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                df[['item_name', 'quantity', 'expiry_days']], 
                column_config={
                    "expiry_days": st.column_config.ProgressColumn("Freshness", min_value=0, max_value=14, format="%d days"),
                    "item_name": "Item",
                    "quantity": "Qty"
                },
                hide_index=True, use_container_width=True
            )
    else:
        st.info("Your kitchen is empty. Scan a receipt or fridge photo above!")

    # --- GMAIL SYNC SECTION ---
    st.divider()

    st.markdown("### 🔗 Digital Connect (Beta)")
    st.caption("Auto-sync e-receipts from Gmail (Costco, Target, Amazon).")
    
    col_gmail, col_status = st.columns([1, 2])
    
    with col_gmail:
        if st.button("🔄 Sync Gmail Receipts"):
            if use_mock:
                st.warning("⚠️ Please turn off Demo Mode.")
            else:
                st.toast("📨 Connecting to Gmail API...", icon="🔒")
                
                mock_email_body = """
                From: service@costco.com
                Subject: E-Receipt for your visit on Jan 28, 2026
                Item            Qty     Price
                KS ALMOND MILK  2       $8.99
                ORG SPINACH     1       $4.49
                RIBEYE STEAK    1       $42.15
                AVOCADO BAG     1       $5.99
                Total: $61.62
                """
                
                with st.spinner("🔍 Found 1 new receipt from Costco... Parsing..."):
                    time.sleep(1.5)
                    
                    def email_parser_agent(llm):
                        prompt = PromptTemplate(
                            input_variables=["email_text"],
                            template=(
                                "Extract grocery items from this email receipt.\n"
                                "Email Content:\n{email_text}\n\n"
                                "Output strictly valid JSON list only. Use double quotes.\n"
                                "Example: [{{\"item_name\": \"Milk\", \"quantity\": \"2\", \"cost\": 5.99, \"expiry_days\": 7}}]"
                            )
                        )
                        return SimpleAgent(llm, prompt)
                    
                    llm = get_llm(use_mock=False)
                    parser = email_parser_agent(llm)
                    json_str = parser.run({"email_text": mock_email_body})
                    
                    if "```json" in json_str: json_str = json_str.replace("```json", "").replace("```", "")
                    
                    try:
                        try:
                            items = json.loads(json_str)
                        except json.JSONDecodeError:
                            items = ast.literal_eval(json_str)

                        conn = sqlite3.connect("grocery_ai.db")
                        c = conn.cursor()
                        count = 0
                        for item in items:
                            c.execute('''INSERT INTO inventory (item_name, quantity, cost, expiry_days, nutrition)
                                VALUES (?, ?, ?, ?, ?)''', 
                                (item["item_name"], str(item["quantity"]), item.get("cost", 0), 
                                 item.get("expiry_days", 7), "From Costco Email"))
                            count += 1
                        conn.commit()
                        conn.close()
                        
                        st.success(f"✅ Synced {count} items from Costco E-Receipt!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Parsing Failed: {e}")

    with col_status:
        st.info("✅ Gmail Account: jason.g***@gmail.com (Connected)")
        st.caption("Last synced: Just now")

# ==========================================
# TAB 2: 📅 Meal Plan
# ==========================================
with tabs[1]:
    st.markdown("### 👨‍🍳 Chef's Recommendation")
    
    with st.expander("➕ Add manual items (Optional)"):
        pantry_text = st.text_area("Type items here", placeholder="e.g., Salt, Pepper, Olive Oil")

    if st.button("🚀 Generate Weekly Plan"):
        # Gather pantry items
        pantry_list = []
        if pantry_text: pantry_list.extend([x.strip() for x in pantry_text.split(",") if x.strip()])
        try:
            conn = sqlite3.connect("grocery_ai.db")
            db_data = pd.read_sql("SELECT item_name, quantity, expiry_days FROM inventory", conn)
            conn.close()
            for _, row in db_data.iterrows():
                item_s = f"{row['item_name']} (Qty: {row['quantity']})"
                if row['expiry_days'] < 5: item_s += " [Use Soon!]"
                pantry_list.append(item_s)
        except: pass
        pantry_str = ", ".join(pantry_list) if pantry_list else "None"

        llm = get_llm(use_mock=use_mock)
        
        with st.status("Thinking...", expanded=True) as status:
            st.write("🧠 Analyzing inventory...")
            plan_agent = meal_planning_agent(llm)
            meal_plan = plan_agent.run({
                "budget": budget, 
                "cuisine": cuisine_str, 
                "diet": diet, 
                "calories": calorie_target, 
                "pantry_items": pantry_str
            })
            st.session_state["gen_meal_plan"] = meal_plan
            
            st.write("📝 Creating shopping list (Consolidating)...")
            list_agent = shopping_list_agent(llm)
            raw_list = list_agent.run({"meal_plan": meal_plan})
            
            st.write("🧮 Checking inventory quantities...")
            checker = inventory_check_agent(llm)
            checked_report = checker.run({"shopping_list": raw_list, "inventory": pantry_str})
            st.session_state["gen_shopping_list"] = (raw_list, checked_report)
            
            st.write("📊 Optimizing budget (Linear Programming)...")
            meals, ml_cost = optimize_meals_ml(budget, calorie_target, diet)
            st.session_state["gen_ml_df"] = pd.DataFrame(meals)
            
            st.session_state["gen_logs"] = {"ml_cost": ml_cost}
            st.session_state["gen_store_strategy"] = None
            
            status.update(label="Planning Complete!", state="complete", expanded=False)

    if st.session_state["gen_meal_plan"]:
        with st.container(border=True):
            st.markdown(md_safe(st.session_state["gen_meal_plan"]))
            
        with st.expander("📊 View Budget & Nutrition Analysis"):
            if st.session_state["gen_ml_df"] is not None:
                df = st.session_state["gen_ml_df"]
                c1, c2 = st.columns(2)
                with c1:
                    fig_cost = px.bar(df, x='meal', y='cost', color='cost', title="Cost Analysis")
                    st.plotly_chart(fig_cost, use_container_width=True)
                with c2:
                    fig_cal = px.scatter(df, x='meal', y='calories', size='calories', title="Calorie Analysis")
                    st.plotly_chart(fig_cal, use_container_width=True)
    else:
        st.info("Tap 'Generate' to create your plan.")

# ==========================================
# TAB 3: 🛒 Shopping Strategy
# ==========================================
with tabs[2]:
    st.markdown("### 🛍️ Smart Shopping List")
    
    if st.session_state["gen_shopping_list"]:
        _, checked_report = st.session_state["gen_shopping_list"]
        sub1, sub2 = st.tabs(["📝 Checklist", "🚀 Execution Strategy"])
        
        with sub1:
            st.info("Items auto-deducted based on your inventory.")
            st.markdown(md_safe(checked_report))

        with sub2:
            st.caption("AI compares prices from Google Shopping to find the best deals.")
            
            if st.button("🔎 Analyze Prices & Find Products"):
                llm = get_llm(use_mock=use_mock)
                
                with st.spinner("📑 Extracting search keywords..."):
                    def extractor(llm):
                        return SimpleAgent(llm, PromptTemplate(input_variables=["text"], 
                            template="Extract top 5 item names for Google Shopping from: {text}. Output JSON list strings."))
                    try:
                        items_str = extractor(llm).run({"text": checked_report}).replace("```json","").replace("```","")
                        items_to_search = json.loads(items_str)
                    except: 
                        items_to_search = ["Milk", "Eggs", "Bread", "Chicken", "Rice"]
                
                results_for_llm = []
                full_results_objects = []
                
                if items_to_search:
                    progress_text = "Searching Google Shopping..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    for i, item in enumerate(items_to_search):
                        my_bar.progress((i + 1) / len(items_to_search), text=f"Searching price for: {item}")
                        res = fetch_grocery_price(item, store_preference="Any")
                        
                        if res:
                            results_for_llm.append(f"- {item}: Found at {res['source']} for {res['price']}")
                            full_results_objects.append(res)
                        else:
                            results_for_llm.append(f"- {item}: Not found online (Estimate price).")
                    
                    my_bar.empty()
                
                st.session_state["search_results"] = full_results_objects
                
                with st.spinner("🤖 AI is calculating the best purchasing strategy..."):
                    store_agent = store_optimization_agent(llm)
                    strategy_text = store_agent.run({
                        "shopping_list": checked_report, 
                        "real_time_prices": "\n".join(results_for_llm)
                    })
                    st.session_state["gen_store_strategy"] = strategy_text

            if st.session_state["gen_store_strategy"]:
                # Calculate simple total for DB storage
                current_total = 0.0
                for r in st.session_state["search_results"]:
                    try:
                        p = float(r['price'].replace('$','').replace(',','').split()[0])
                        current_total += p
                    except: pass

                # Display Strategy
                with st.container(border=True):
                    st.markdown("### 📋 Buying Strategy")
                    st.markdown(md_safe(st.session_state["gen_store_strategy"]))
                
                # SAVE TO DB BUTTON
                st.markdown('<div class="save-btn">', unsafe_allow_html=True)
                if st.button("💾 Save this Run to History"):
                    conn = sqlite3.connect("grocery_ai.db")
                    c = conn.cursor()
                    products_json = json.dumps(st.session_state["search_results"])
                    c.execute('''INSERT INTO shopping_history 
                                 (meal_plan_text, strategy_text, products_json, total_cost)
                                 VALUES (?, ?, ?, ?)''',
                                 (st.session_state["gen_meal_plan"], 
                                  st.session_state["gen_store_strategy"], 
                                  products_json,
                                  current_total))
                    conn.commit()
                    conn.close()
                    st.success("✅ Saved! Check the 'History' tab.")
                st.markdown('</div>', unsafe_allow_html=True)

                st.divider()

                # Visual Grid (2 Columns + Calories + No Google Button)
                if st.session_state["search_results"]:
                    st.markdown("### 📸 Verified Product Matches")
                    
                    cols = st.columns(2)
                    for i, res in enumerate(st.session_state["search_results"]):
                        with cols[i % 2]:
                            with st.container(border=True):
                                # 1. Image
                                if res.get("thumbnail"): 
                                    st.image(res["thumbnail"], use_container_width=True)
                                else: 
                                    st.markdown("🥑 *No Image Available*")
                                
                                # 2. Info
                                title_text = res.get('title', 'Unknown')
                                st.markdown(f"**{title_text}**")
                                st.markdown(f'<p class="price-tag">{res.get("price", "N/A")}</p>', unsafe_allow_html=True)
                                st.markdown(f'<p class="source-tag">📍 {res.get("source", "Unknown")}</p>', unsafe_allow_html=True)
                                
                                # 3. Calories (Simulated based on hash for consistency)
                                # Simple hash to get a consistent number between 50 and 450
                                try:
                                    simulated_cal = (hash(title_text) % 400) + 50
                                except:
                                    simulated_cal = 150
                                st.markdown(f'<p class="cal-tag">🔥 {simulated_cal} kcal / serving</p>', unsafe_allow_html=True)

                                # 4. Buy Button (Only if valid) - Removed Google Fallback
                                link_url = res.get("link")
                                if link_url and isinstance(link_url, str) and link_url.startswith("http"):
                                    st.link_button("Buy Now", link_url, use_container_width=True)
                                # No Else block here (removed Find on Google)

    else:
        st.info("Please generate a meal plan first.")

# ==========================================
# TAB 4: 📜 History
# ==========================================
with tabs[3]:
    st.markdown("### 🕰️ Decision History")
    
    conn = sqlite3.connect("grocery_ai.db")
    history_df = pd.read_sql("SELECT id, timestamp, total_cost FROM shopping_history ORDER BY id DESC", conn)
    
    if not history_df.empty:
        for idx, row in history_df.iterrows():
            with st.expander(f"📅 Run on {row['timestamp']} - Est. Total: ${row['total_cost']:.2f}"):
                details = pd.read_sql(f"SELECT * FROM shopping_history WHERE id = {row['id']}", conn).iloc[0]
                
                st.markdown("#### 📋 Strategy Snapshot")
                st.markdown(details['strategy_text'])
                
                st.divider()
                st.markdown("#### 📸 Products Found")
                try:
                    products = json.loads(details['products_json'])
                    if products:
                        cols = st.columns(3)
                        for i, p in enumerate(products):
                            with cols[i % 3]:
                                if p.get('thumbnail'): st.image(p['thumbnail'], width=100)
                                st.caption(f"{p.get('price')} - {p.get('source')}")
                except:
                    st.error("Could not load product images.")
                    
                if st.button("🗑️ Delete Record", key=f"del_{row['id']}"):
                    conn.execute(f"DELETE FROM shopping_history WHERE id = {row['id']}")
                    conn.commit()
                    st.rerun()
    else:
        st.info("No history yet. Go to 'Shopping' and click 'Save this Run'.")
    
    conn.close()
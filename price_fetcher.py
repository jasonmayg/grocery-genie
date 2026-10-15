from serpapi import GoogleSearch
import streamlit as st

def fetch_grocery_price(item_name, store_preference="Any"):
    """
    Uses SerpApi (Google Shopping) to find the real-time price of an item.
    """
    api_key = st.secrets.get("SERPAPI_KEY")
    
    if not api_key:
        # No key (e.g. Demo Mode): signal "not found" so the agent estimates the price
        return None

    # 构建搜索词：比如 "Costco Organic Milk" 或 "Walmart Eggs"
    query = f"{item_name}"
    if store_preference != "Any":
        query += f" at {store_preference}"

    params = {
        "engine": "google_shopping",
        "q": query,
        "location": "Santa Clara, California, United States", # 你所在的区域
        "hl": "en",
        "gl": "us",
        "num": 5, # 只取前5个结果
        "api_key": api_key
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        shopping_results = results.get("shopping_results", [])
        
        if shopping_results:
            # 找到第一个结果 (通常是最匹配的)
            best_match = shopping_results[0]
            return {
                "title": best_match.get("title"),
                "price": best_match.get("price"),
                "source": best_match.get("source"),
                "thumbnail": best_match.get("thumbnail"),
                "link": best_match.get("link")
            }
        else:
            return None
            
    except Exception as e:
        print(f"SerpApi Error: {e}")
        return None
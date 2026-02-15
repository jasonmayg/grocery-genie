# 文件名: vision_agent.py
import base64
import json
from openai import OpenAI
import streamlit as st

def analyze_image_stream(uploaded_file, api_key):
    """
    专门处理 Streamlit 上传的图片流
    """
    if not api_key:
        return None

    # 1. 直接读取内存中的图片并转为 base64
    bytes_data = uploaded_file.getvalue()
    base64_image = base64.b64encode(bytes_data).decode('utf-8')
    
    client = OpenAI(api_key=api_key)

    # 2. 同样的 Prompt，为了稳定性稍微精简一下
    system_prompt = """
    Analyze the grocery image. Return a JSON object with key "items".
    For each item:
    - "item_name": string
    - "quantity": string (estimate)
    - "cost": float (estimate in USD if not visible)
    - "expiry_days": int (estimate days until spoiled)
    - "nutrition": One of ["Protein", "Vegetable", "Fruit", "Carb", "Dairy", "Spice"]
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # 必须是 gpt-4o
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract data."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Vision API Error: {e}")
        return None
# 文件名: vision_test.py
import base64
import os
import json
from openai import OpenAI

# ---------------- 配置区域 ----------------
# 为了测试脚本简单，你可以直接在这里填 Key，或者从环境变量读取
# 如果你已经配置了 .streamlit/secrets.toml，通常我们在 app 里读，
# 这里为了独立运行，请把 Key 填在下面引号里，或者确保环境变量里有 OPENAI_API_KEY
API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR_OPENAI_API_KEY"
IMAGE_PATH = "test_image.jpg"  # 确保你文件夹里有这张图
# ----------------------------------------

client = OpenAI(api_key=API_KEY)

# 1. 辅助函数：把图片转成 Base64 编码 (AI 只能读这种格式)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 2. 核心函数：万能视觉分析
def analyze_fridge_or_receipt(image_path):
    print(f"🔍 正在分析图片: {image_path} ...")
    
    base64_image = encode_image(image_path)

    # 这里的 Prompt 是关键！一定要明确定义 JSON 结构
    system_prompt = """
    You are an AI Food Data Analyst. 
    Analyze the provided image (which could be a grocery receipt or a photo of food).
    Extract inventory items and return a JSON object with a key "items" containing a list.
    
    For EACH item detected, provide:
    - "item_name": Concise name (e.g., "Spinach", "Milk").
    - "quantity": Estimated quantity (e.g., "1 bag", "1 gallon").
    - "cost": The price on the receipt (float), OR estimate the average market cost if not visible.
    - "expiry_days": Estimated days until it spoils (int). Logic: Leafy greens~5, Milk~7, Eggs~21, Canned~365.
    - "nutrition": Main category (Select one: "Protein", "Vegetable", "Fruit", "Carb", "Dairy", "Snack", "Spice").
    
    Output strictly valid JSON.
    """

    response = client.chat.completions.create(
        model="gpt-4o",  # 必须用支持视觉的模型
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract data from this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        response_format={"type": "json_object"}, # 强制 JSON 模式，防止 AI 废话
        temperature=0.0, # 越低越精准
    )
    
    return response.choices[0].message.content

# 3. 运行测试
if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 错误：找不到文件 {IMAGE_PATH}，请先放一张图片进去。")
    else:
        try:
            result_json = analyze_fridge_or_receipt(IMAGE_PATH)
            
            # 解析并漂亮地打印出来
            data = json.loads(result_json)
            print("\n✅ AI 识别成功！数据如下：\n")
            print(json.dumps(data, indent=4, ensure_ascii=False))
            
            print(f"\n[统计] 一共识别出 {len(data['items'])} 个物品。")
            
        except Exception as e:
            print(f"❌ 发生错误: {e}")
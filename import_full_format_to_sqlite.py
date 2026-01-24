import json, zipfile, sqlite3, math
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "grocery_ai.db"

MEAT_WORDS = {
    "chicken","turkey","beef","pork","bacon","ham","sausage","salami",
    "lamb","veal","steak","rib","ribs","anchovy","anchovies","tuna",
    "salmon","shrimp","prawn","prawns","crab","lobster","fish"
}
DAIRY_EGG_WORDS = {"milk","cheese","butter","yogurt","cream","egg","eggs","whey"}

def estimate_cost(ingredients, calories):
    # Simple, deterministic demo cost estimate:
    # base + ingredient count + a tiny calorie factor
    n = len(ingredients) if ingredients else 6
    cals = float(calories) if calories not in (None, "") else 600.0
    cost = 2.0 + 0.35 * n + 0.002 * cals
    # clamp to a sensible range
    return round(max(3.0, min(12.0, cost)), 2)

def infer_diet(ingredients, categories):
    text = " ".join((ingredients or [])).lower()
    cats = " ".join(categories or []).lower()

    has_meat = any(w in text for w in MEAT_WORDS)
    has_dairy_egg = any(w in text for w in DAIRY_EGG_WORDS)

    # categories hints (some datasets include these words)
    if "vegan" in cats:
        return "Vegan"
    if "vegetarian" in cats:
        return "Vegetarian"

    if has_meat:
        return "None"
    if has_dairy_egg:
        return "Vegetarian"
    return "Vegan"

def init_table(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS recipes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        source TEXT NOT NULL,
        title TEXT NOT NULL,
        calories REAL,
        protein REAL,
        fat REAL,
        sodium REAL,
        rating REAL,
        diet_tag TEXT,
        categories_json TEXT,
        ingredients_json TEXT,
        directions_json TEXT,
        est_cost REAL,
        n_ingredients INTEGER
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_recipes_diet ON recipes(diet_tag)")
    conn.commit()

def import_zip(zip_path: str):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("Zip file is empty")
        json_name = names[0]
        raw = zf.read(json_name)
        data = json.loads(raw)

    conn = sqlite3.connect(DB_PATH)
    init_table(conn)
    cur = conn.cursor()

    inserted = 0
    now = datetime.now().isoformat(timespec="seconds")

    for r in data:
        title = (r.get("title") or "").strip()
        if not title:
            continue

        ingredients = r.get("ingredients") or []
        directions = r.get("directions") or []
        categories = r.get("categories") or []

        calories = r.get("calories")
        protein = r.get("protein")
        fat = r.get("fat")
        sodium = r.get("sodium")
        rating = r.get("rating")

        diet_tag = infer_diet(ingredients, categories)
        est_cost = estimate_cost(ingredients, calories)
        n_ingredients = len(ingredients)

        cur.execute("""
            INSERT INTO recipes (
                created_at, source, title,
                calories, protein, fat, sodium, rating,
                diet_tag, categories_json, ingredients_json, directions_json,
                est_cost, n_ingredients
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now, f"kaggle:{zip_path.name}", title,
            calories, protein, fat, sodium, rating,
            diet_tag,
            json.dumps(categories, ensure_ascii=False),
            json.dumps(ingredients, ensure_ascii=False),
            json.dumps(directions, ensure_ascii=False),
            est_cost, n_ingredients
        ))
        inserted += 1

    conn.commit()
    conn.close()
    print(f"✅ Imported {inserted} recipes into {DB_PATH}")

if __name__ == "__main__":
    import_zip("data/full_format_recipes.json.zip")

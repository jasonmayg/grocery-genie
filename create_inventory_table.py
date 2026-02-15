# 文件名: create_inventory_table.py
import sqlite3

def init_inventory_db():
    conn = sqlite3.connect("grocery_ai.db")
    c = conn.cursor()
    
    # 创建一个功能强大的 inventory 表
    # 包含了我们在 JSON 里看到的所有字段
    c.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT,
            quantity TEXT,
            cost REAL,
            expiry_days INTEGER,
            nutrition TEXT,
            date_added DATE DEFAULT (date('now'))
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ 库存表 (Inventory Table) 创建成功！")

if __name__ == "__main__":
    init_inventory_db()
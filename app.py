import json
import pandas as pd
import psycopg2
import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Calorie Tracker", page_icon="🍽️", layout="centered")

# ---------- CONFIG (Secrets) ----------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
DB_URL = st.secrets.get("DB_URL", "")

if not GEMINI_API_KEY or not DB_URL:
    st.warning("Missing secrets. Add GEMINI_API_KEY and DB_URL in Streamlit secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# Store all entries under one user (since login removed)
DEFAULT_USER_EMAIL = "default"

# ---------- DB ----------
def db_conn():
    return psycopg2.connect(DB_URL)

def insert_entry(user_email: str, raw_text: str) -> int:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO meal_entries (user_email, raw_text) VALUES (%s, %s) RETURNING id",
                (user_email, raw_text),
            )
            entry_id = cur.fetchone()[0]
            conn.commit()
            return entry_id

def insert_items(entry_id: int, items: list[dict]):
    rows = []
    for it in items:
        rows.append((
            entry_id,
            it.get("name", "unknown"),
            it.get("quantity", None),
            it.get("unit", "serving"),
            float(it.get("calories", 0) or 0),
            it.get("protein_g", None),
            it.get("carbs_g", None),
            it.get("fat_g", None),
            it.get("confidence", None),
        ))

    if not rows:
        return

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO meal_items
                (entry_id, name, quantity, unit, calories, protein_g, carbs_g, fat_g, confidence)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                rows,
            )
            conn.commit()

def fetch_today(user_email: str):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.id, e.created_at, e.raw_text
                FROM meal_entries e
                WHERE e.user_email = %s
                  AND e.created_at::date = CURRENT_DATE
                ORDER BY e.created_at DESC
                """,
                (user_email,),
            )
            entries = cur.fetchall()

            if not entries:
                return pd.DataFrame(), pd.DataFrame()

            entry_ids = [e[0] for e in entries]
            cur.execute(
                """
                SELECT entry_id, name, quantity, unit, calories, protein_g, carbs_g, fat_g, confidence
                FROM meal_items
                WHERE entry_id = ANY(%s)
                """,
                (entry_ids,),
            )
            items = cur.fetchall()

    entries_df = pd.DataFrame(entries, columns=["entry_id", "created_at", "raw_text"])
    items_df = pd.DataFrame(
        items,
        columns=["entry_id", "name", "quantity", "unit", "calories", "protein_g", "carbs_g", "fat_g", "confidence"],
    )
    return entries_df, items_df

# ---------- GEMINI PARSER ----------
def parse_meal(text: str) -> dict:
    prompt = f"""
Return JSON only (no markdown, no extra text).

Convert this messy food text into estimated nutrition.
User may not give grams. Use typical portions.
Prefer Pakistani/South Asian defaults when relevant (roti, paratha, biryani, karahi, chai).

Schema:
{{
  "items": [
    {{
      "name": "string",
      "quantity": number|null,
      "unit": "string",
      "calories": number,
      "protein_g": number|null,
      "carbs_g": number|null,
      "fat_g": number|null,
      "confidence": number
    }}
  ],
  "total_calories": number,
  "assumptions": ["string"]
}}

Rules:
- If quantity unclear: quantity=null and unit="serving"
- confidence is 0..1
- Output must be valid JSON

User text: "{text}"
""".strip()

    model = genai.GenerativeModel(MODEL_NAME)
    res = model.generate_content(prompt)
    raw = res.text.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Gemini did not return JSON.")
        return json.loads(raw[start:end + 1])

# ---------- UI ----------
st.title("🍽️ Calorie Tracker")

target = st.number_input("Daily calorie target", min_value=800, max_value=5000, value=2000, step=50)

st.subheader("Add what you ate (random text is fine)")
food_text = st.text_area(
    "Example: 2 paratha, chai 1 cup, chicken karahi 1 bowl",
    height=90,
)

col1, col2 = st.columns([1, 1])
with col1:
    add_btn = st.button("Add meal", use_container_width=True)
with col2:
    demo_btn = st.button("Use demo text", use_container_width=True)

if demo_btn:
    st.session_state["demo_text"] = "2 paratha, chai 1 cup, chicken karahi 1 bowl"
    st.rerun()

if "demo_text" in st.session_state and not food_text:
    food_text = st.session_state["demo_text"]

if add_btn:
    if not food_text.strip():
        st.error("Type your food first.")
    else:
        with st.spinner("Estimating calories with Gemini..."):
            parsed = parse_meal(food_text.strip())

        entry_id = insert_entry(DEFAULT_USER_EMAIL, food_text.strip())
        insert_items(entry_id, parsed.get("items", []))

        st.success("Saved ✅")
        st.caption("Assumptions: " + "; ".join(parsed.get("assumptions", [])[:3]))
        st.rerun()

st.divider()

entries_df, items_df = fetch_today(DEFAULT_USER_EMAIL)

st.subheader("Today")
if items_df.empty:
    st.write("No meals logged today.")
else:
    total = float(items_df["calories"].sum())
    remaining = max(0, target - total)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total calories", f"{total:.0f}")
    c2.metric("Target", f"{target:.0f}")
    c3.metric("Remaining", f"{remaining:.0f}")

    st.write("Today’s entries")
    for _, row in entries_df.iterrows():
        entry_id = row["entry_id"]
        entry_items = items_df[items_df["entry_id"] == entry_id]
        entry_total = float(entry_items["calories"].sum()) if not entry_items.empty else 0

        with st.expander(f"{row['created_at']} — {entry_total:.0f} kcal"):
            st.write(row["raw_text"])
            if entry_items.empty:
                st.write("No parsed items.")
            else:
                show = entry_items[["name", "quantity", "unit", "calories", "confidence"]].copy()
                st.dataframe(show, use_container_width=True, hide_index=True)

if st.button("Test DB"):
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("select now();")
                st.success(f"DB OK: {cur.fetchone()[0]}")
    except Exception as e:
        st.error(e)

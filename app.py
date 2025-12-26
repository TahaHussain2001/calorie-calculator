import os
import json
import base64
from typing import List, Dict, Optional

import pandas as pd
import psycopg2
import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Calorie Tracker", page_icon="🍽️", layout="centered")

# ---------------- CONFIG ----------------
def get_env(key: str, default: str = "") -> str:
    # Priority: system env vars -> Streamlit secrets
    return os.environ.get(key) or st.secrets.get(key, default)

GEMINI_API_KEY = get_env("GEMINI_API_KEY")
DB_URL = get_env("DB_URL")

if not GEMINI_API_KEY or not DB_URL:
    st.error(
        "Missing configuration.\n\n"
        "Set GEMINI_API_KEY and DB_URL in Environment Variables or Streamlit secrets."
    )
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

DEFAULT_USER_EMAIL = "default"  # all entries under one user for now

# ---------------- UI THEME / BACKGROUND ----------------
def set_bg(image_path: str):
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background:
                    linear-gradient(rgba(0,0,0,.70), rgba(0,0,0,.75)),
                    url("data:image/jpg;base64,{b64}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            .block-container {{
                padding-top: 2rem !important;
                padding-bottom: 2rem !important;
            }}

            .glass {{
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.16);
                border-radius: 18px;
                padding: 18px 18px;
                backdrop-filter: blur(10px);
            }}

            .hero-title {{
                font-size: 42px;
                font-weight: 800;
                color: #fff;
                margin: 0 0 6px 0;
            }}
            .hero-subtitle {{
                color: rgba(255,255,255,0.85);
                font-size: 16px;
                margin: 0 0 16px 0;
            }}

            label, .stMarkdown, .stText {{
                color: rgba(255,255,255,0.92) !important;
            }}

            textarea, input {{
                border-radius: 12px !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.warning("Background image not found. Put it at: assets/bg.jpg")

set_bg("bg.jpg")


# ---------------- DB ----------------
def db_conn():
    return psycopg2.connect(DB_URL, sslmode="require", connect_timeout=10)

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

def insert_items(entry_id: int, items: List[Dict]):
    rows = []
    for it in items:
        rows.append(
            (
                entry_id,
                it.get("name", "unknown"),
                it.get("quantity", None),
                it.get("unit", "serving"),
                float(it.get("calories", 0) or 0),
                float(it.get("protein_g", 0) or 0),
                float(it.get("carbs_g", 0) or 0),
                float(it.get("fat_g", 0) or 0),
                float(it.get("confidence", 0) or 0),
            )
        )

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

def fetch_today_items(user_email: str):
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

def get_first_meal_date(user_email: str) -> Optional[pd.Timestamp]:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MIN(created_at::date)
                FROM meal_entries
                WHERE user_email = %s
                """,
                (user_email,),
            )
            row = cur.fetchone()
    return row[0] if row and row[0] else None

def fetch_last_7_days_totals(user_email: str) -> pd.DataFrame:
    first_day = get_first_meal_date(user_email)
    if not first_day:
        return pd.DataFrame()

    today = pd.Timestamp.utcnow().date()

    days_since_start = (today - first_day).days
    if days_since_start < 6:
        start_day = first_day
    else:
        start_day = today - pd.Timedelta(days=6)

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  e.created_at::date as day,
                  COALESCE(SUM(e.total_calories), 0) as total_calories,
                  COALESCE(SUM(e.total_protein_g), 0) as total_protein_g,
                  COALESCE(SUM(e.total_carbs_g), 0) as total_carbs_g,
                  COALESCE(SUM(e.total_fat_g), 0) as total_fat_g,
                  COUNT(*) as meals_logged
                FROM meal_entries e
                WHERE e.user_email = %s
                  AND e.created_at::date >= %s
                  AND e.created_at::date <= %s
                GROUP BY day
                ORDER BY day ASC
                """,
                (user_email, start_day, today),
            )
            rows = cur.fetchall()

    df = pd.DataFrame(
        rows,
        columns=[
            "day",
            "total_calories",
            "total_protein_g",
            "total_carbs_g",
            "total_fat_g",
            "meals_logged",
        ],
    )

    full_days = pd.date_range(start=start_day, end=today, freq="D").date
    full_df = pd.DataFrame({"day": full_days})
    df = full_df.merge(df, on="day", how="left").fillna(0)

    for col in ["total_calories", "total_protein_g", "total_carbs_g", "total_fat_g", "meals_logged"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ---------------- GEMINI PARSER ----------------
def parse_meal(text: str) -> dict:
    prompt = f"""
Return JSON only (no markdown, no extra text).

Convert the user's messy food text into estimated nutrition with macros.
User may not give grams. Use typical portion sizes.
Prefer Pakistani/South Asian defaults when relevant (roti, paratha, biryani, karahi, chai).

Schema (must match exactly):
{{
  "items": [
    {{
      "name": "string",
      "quantity": number|null,
      "unit": "string",
      "calories": number,
      "protein_g": number,
      "carbs_g": number,
      "fat_g": number,
      "confidence": number
    }}
  ],
  "total_calories": number,
  "total_protein_g": number,
  "total_carbs_g": number,
  "total_fat_g": number,
  "assumptions": ["string"]
}}

Rules:
- If quantity unclear: quantity=null and unit="serving"
- protein_g, carbs_g, fat_g must be numbers (use 0 if unknown)
- confidence is 0..1
- Output must be valid JSON.

User text: "{text}"
""".strip()

    model = genai.GenerativeModel(MODEL_NAME)
    res = model.generate_content(prompt)
    raw = (res.text or "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Gemini did not return JSON.")
        return json.loads(raw[start:end + 1])


# ---------------- APP UI ----------------
st.markdown('<div class="hero-title">🍽️ Calorie Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Type what you ate. Get calories + macros (estimated) instantly.</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    cA, cB = st.columns([1, 1])
    with cA:
        target_kcal = st.number_input("Daily calorie target", min_value=800, max_value=5000, value=2000, step=50)
    with cB:
        protein_target = st.number_input("Protein target (g)", min_value=0, max_value=400, value=120, step=5)

    st.subheader("Add what you ate")
    food_text = st.text_area(
        "Example: 2 paratha, chai 1 cup, chicken karahi 1 bowl",
        height=90,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        add_btn = st.button("Add meal", use_container_width=True)
    with col2:
        demo_btn = st.button("Use demo text", use_container_width=True)
    with col3:
        test_db_btn = st.button("Test DB", use_container_width=True)

    if demo_btn:
        st.session_state["demo_text"] = "2 paratha, chai 1 cup, chicken karahi 1 bowl"
        st.rerun()

    if "demo_text" in st.session_state and not food_text:
        food_text = st.session_state["demo_text"]

    if test_db_btn:
        try:
            with db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("select now();")
                    st.success(f"DB OK: {cur.fetchone()[0]}")
        except Exception as e:
            st.error(e)

    st.markdown("</div>", unsafe_allow_html=True)

if add_btn:
    if not food_text.strip():
        st.error("Type your food first.")
    else:
        with st.spinner("Estimating calories + macros with Gemini..."):
            parsed = parse_meal(food_text.strip())

        entry_id = insert_entry(DEFAULT_USER_EMAIL, food_text.strip())
        insert_items(entry_id, parsed.get("items", []))

        st.success("Saved ✅")
        assumptions = parsed.get("assumptions", [])
        if assumptions:
            st.caption("Assumptions: " + "; ".join(assumptions[:3]))
        st.rerun()

st.divider()

# ---------------- LAST 7 DAYS ----------------
st.subheader("Last 7 Days")

week_df = fetch_last_7_days_totals(DEFAULT_USER_EMAIL)

if week_df.empty:
    st.write("No history yet. Add your first meal to start tracking.")
else:
    wk_kcal = float(week_df["total_calories"].sum())
    wk_p = float(week_df["total_protein_g"].sum())
    wk_c = float(week_df["total_carbs_g"].sum())
    wk_f = float(week_df["total_fat_g"].sum())

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("7-day calories", f"{wk_kcal:.0f} kcal")
    a2.metric("7-day protein", f"{wk_p:.0f} g")
    a3.metric("7-day carbs", f"{wk_c:.0f} g")
    a4.metric("7-day fat", f"{wk_f:.0f} g")

    st.line_chart(week_df.set_index("day")[["total_calories"]])

    show_week = week_df.copy()
    show_week["day"] = show_week["day"].astype(str)

    st.dataframe(
        show_week[["day", "meals_logged", "total_calories", "total_protein_g", "total_carbs_g", "total_fat_g"]],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ---------------- TODAY (DETAIL) ----------------
entries_df, items_df = fetch_today_items(DEFAULT_USER_EMAIL)

st.subheader("Today (Details)")

if items_df.empty:
    st.write("No meals logged today.")
else:
    for col in ["calories", "protein_g", "carbs_g", "fat_g", "confidence"]:
        items_df[col] = pd.to_numeric(items_df[col], errors="coerce").fillna(0)

    total_kcal = float(items_df["calories"].sum())
    total_p = float(items_df["protein_g"].sum())
    total_c = float(items_df["carbs_g"].sum())
    total_f = float(items_df["fat_g"].sum())

    kcal_progress = min(1.0, max(0.0, total_kcal / float(target_kcal)))
    p_progress = 1.0 if protein_target == 0 else min(1.0, max(0.0, total_p / float(protein_target)))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calories", f"{total_kcal:.0f} kcal")
    m2.metric("Protein", f"{total_p:.0f} g")
    m3.metric("Carbs", f"{total_c:.0f} g")
    m4.metric("Fat", f"{total_f:.0f} g")

    st.caption("Calories progress")
    st.progress(kcal_progress)

    st.caption("Protein progress")
    st.progress(p_progress)

    st.write("Meals")
    for _, row in entries_df.iterrows():
        entry_id = row["entry_id"]
        entry_items = items_df[items_df["entry_id"] == entry_id].copy()

        entry_kcal = float(entry_items["calories"].sum()) if not entry_items.empty else 0
        entry_p = float(entry_items["protein_g"].sum()) if not entry_items.empty else 0
        entry_c = float(entry_items["carbs_g"].sum()) if not entry_items.empty else 0
        entry_f = float(entry_items["fat_g"].sum()) if not entry_items.empty else 0

        with st.expander(
            f"{row['created_at']} — {entry_kcal:.0f} kcal | P {entry_p:.0f}g C {entry_c:.0f}g F {entry_f:.0f}g"
        ):
            st.write(row["raw_text"])
            show = entry_items[
                ["name", "quantity", "unit", "calories", "protein_g", "carbs_g", "fat_g", "confidence"]
            ].copy()
            st.dataframe(show, use_container_width=True, hide_index=True)

import os
import json
import base64
from datetime import datetime, date, time
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional, Tuple

import pandas as pd
import psycopg2
import streamlit as st
import google.generativeai as genai

# ---------------- APP CONFIG ----------------
st.set_page_config(page_title="Calorie Tracker", page_icon="🍽️", layout="centered")

LOCAL_TZ = ZoneInfo("Asia/Karachi")
TZ_NAME = "Asia/Karachi"

# ---------------- CONFIG ----------------
def get_env(key: str, default: str = "") -> str:
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

DEFAULT_USER_EMAIL = "default"  # only you

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
              /* Targets row: push the button down to align with number inputs */
      .targets-row [data-testid="stButton"] {
          margin-top: 26px;
      }

      /* Small tweak: remove extra top padding that some themes add */
      .targets-row .stButton > button {
          height: 42px; /* optional, looks closer to input height */
      }
            </style>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.warning("Background image not found. Put it at: bg.jpg")

set_bg("bg.jpg")

# ---------------- DB ----------------
def db_conn():
    return psycopg2.connect(DB_URL, sslmode="require", connect_timeout=10)

# ---------- Daily Targets (per date) ----------
def get_daily_targets(user_email: str, target_date: date) -> Tuple[int, int]:
    """
    Returns (target_kcal, protein_target_g) for a specific date.
    If missing, returns defaults (2000, 120).
    """
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT target_kcal, protein_target_g
                FROM daily_targets
                WHERE user_email = %s AND target_date = %s
                """,
                (user_email, target_date),
            )
            row = cur.fetchone()

    if row:
        return int(row[0]), int(row[1])

    return 2000, 120

def upsert_daily_targets(user_email: str, target_date: date, target_kcal: int, protein_target_g: int):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO daily_targets (user_email, target_date, target_kcal, protein_target_g)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_email, target_date)
                DO UPDATE SET
                    target_kcal = EXCLUDED.target_kcal,
                    protein_target_g = EXCLUDED.protein_target_g,
                    updated_at = NOW()
                """,
                (user_email, target_date, int(target_kcal), int(protein_target_g)),
            )
            conn.commit()

# ---------- Meals ----------
def insert_entry(
    user_email: str,
    raw_text: str,
    created_at_local: datetime,
    totals: Dict[str, float],
) -> int:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO meal_entries
                  (user_email, raw_text, created_at,
                   total_calories, total_protein_g, total_carbs_g, total_fat_g)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    user_email,
                    raw_text,
                    created_at_local,
                    float(totals.get("total_calories", 0) or 0),
                    float(totals.get("total_protein_g", 0) or 0),
                    float(totals.get("total_carbs_g", 0) or 0),
                    float(totals.get("total_fat_g", 0) or 0),
                ),
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
                  (entry_id, name, quantity, unit,
                   calories, protein_g, carbs_g, fat_g, confidence)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                rows,
            )
            conn.commit()

def fetch_day_items(user_email: str, day_local: date):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT e.id, e.created_at, e.raw_text
                FROM meal_entries e
                WHERE e.user_email = %s
                  AND (e.created_at AT TIME ZONE %s)::date = %s
                ORDER BY e.created_at DESC
                """,
                (user_email, TZ_NAME, day_local),
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

def fetch_last_7_days_totals(user_email: str) -> pd.DataFrame:
    today_local = datetime.now(LOCAL_TZ).date()
    start_local = today_local - pd.Timedelta(days=6)

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                  (e.created_at AT TIME ZONE %s)::date AS day,
                  COALESCE(SUM(i.calories), 0) AS total_calories,
                  COALESCE(SUM(i.protein_g), 0) AS total_protein_g,
                  COALESCE(SUM(i.carbs_g), 0) AS total_carbs_g,
                  COALESCE(SUM(i.fat_g), 0) AS total_fat_g,
                  COUNT(DISTINCT e.id) AS meals_logged
                FROM meal_entries e
                LEFT JOIN meal_items i ON i.entry_id = e.id
                WHERE e.user_email = %s
                  AND (e.created_at AT TIME ZONE %s)::date >= %s
                  AND (e.created_at AT TIME ZONE %s)::date <= %s
                GROUP BY day
                ORDER BY day ASC
                """,
                (TZ_NAME, user_email, TZ_NAME, start_local, TZ_NAME, today_local),
            )
            rows = cur.fetchall()

    df = pd.DataFrame(
        rows,
        columns=["day", "total_calories", "total_protein_g", "total_carbs_g", "total_fat_g", "meals_logged"],
    )

    full_days = pd.date_range(start=start_local, end=today_local, freq="D").date
    df = pd.DataFrame({"day": full_days}).merge(df, on="day", how="left").fillna(0)

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

# ---------------- DEMO CALLBACK ----------------
def fill_demo():
    st.session_state["food_text"] = "2 paratha, chai 1 cup, chicken karahi 1 bowl"

# ---------------- APP UI ----------------
st.markdown('<div class="hero-title">🍽️ Calorie Tracker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Log meals on any date. Save daily calorie/protein targets.</div>',
    unsafe_allow_html=True
)

# ✅ Manual date/time for logging + viewing
st.subheader("Log settings")
now_local = datetime.now(LOCAL_TZ)

d1, d2 = st.columns(2)
with d1:
    log_date = st.date_input("Log date", value=now_local.date())
with d2:
    log_time = st.time_input("Log time", value=now_local.time().replace(microsecond=0))

# ✅ Load targets for this selected day into session state (so switching date loads new targets)
day_targets = get_daily_targets(DEFAULT_USER_EMAIL, log_date)
if "target_kcal" not in st.session_state or st.session_state.get("_targets_for_date") != log_date:
    st.session_state["target_kcal"] = day_targets[0]
    st.session_state["protein_target"] = day_targets[1]
    st.session_state["_targets_for_date"] = log_date

st.subheader("Daily targets (for selected date)")

st.markdown('<div class="targets-row">', unsafe_allow_html=True)
t1, t2, t3 = st.columns([1, 1, 1])

with t1:
    st.number_input(
        "Daily calorie target",
        min_value=800,
        max_value=5000,
        step=50,
        key="target_kcal",
    )

with t2:
    st.number_input(
        "Protein target (g)",
        min_value=0,
        max_value=400,
        step=5,
        key="protein_target",
    )

with t3:
    if st.button("Save targets", use_container_width=True):
        upsert_daily_targets(
            DEFAULT_USER_EMAIL,
            log_date,
            st.session_state["target_kcal"],
            st.session_state["protein_target"],
        )
        st.success("Targets saved ✅")

st.markdown("</div>", unsafe_allow_html=True)


st.subheader("Add what you ate")

st.text_area(
    "Example: 2 paratha, chai 1 cup, chicken karahi 1 bowl",
    height=90,
    key="food_text",
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    add_btn = st.button("Add meal", use_container_width=True)
with col2:
    st.button("Use demo text", use_container_width=True, on_click=fill_demo)
with col3:
    test_db_btn = st.button("Test DB", use_container_width=True)

if test_db_btn:
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("select now();")
                st.success(f"DB OK: {cur.fetchone()[0]}")
    except Exception as e:
        st.error(e)

current_text = st.session_state.get("food_text", "")

if add_btn:
    if not current_text.strip():
        st.error("Type your food first.")
    else:
        with st.spinner("Estimating calories + macros with Gemini..."):
            parsed = parse_meal(current_text.strip())

        # timezone-aware datetime for timestamptz
        created_at_local = datetime.combine(log_date, log_time).replace(tzinfo=LOCAL_TZ)

        totals = {
            "total_calories": parsed.get("total_calories", 0),
            "total_protein_g": parsed.get("total_protein_g", 0),
            "total_carbs_g": parsed.get("total_carbs_g", 0),
            "total_fat_g": parsed.get("total_fat_g", 0),
        }

        entry_id = insert_entry(DEFAULT_USER_EMAIL, current_text.strip(), created_at_local, totals)
        insert_items(entry_id, parsed.get("items", []))

        st.success(f"Saved ✅ (Logged: {log_date} {log_time})")
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

    a1, a2 = st.columns(2)
    a1.metric("7-day calories", f"{wk_kcal:.0f} kcal")
    a2.metric("7-day protein", f"{wk_p:.0f} g")

    st.line_chart(week_df.set_index("day")[["total_calories"]])

    show_week = week_df.copy()
    show_week["day"] = show_week["day"].astype(str)
    st.dataframe(
        show_week[["day", "meals_logged", "total_calories", "total_protein_g", "total_carbs_g", "total_fat_g"]],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ---------------- SELECTED DAY (DETAIL) ----------------
st.subheader(f"Details for {log_date}")

entries_df, items_df = fetch_day_items(DEFAULT_USER_EMAIL, log_date)

if items_df.empty:
    st.write("No meals logged for this date.")
else:
    for col in ["calories", "protein_g", "carbs_g", "fat_g", "confidence"]:
        items_df[col] = pd.to_numeric(items_df[col], errors="coerce").fillna(0)

    total_kcal = float(items_df["calories"].sum())
    total_p = float(items_df["protein_g"].sum())

    target_kcal = float(st.session_state.get("target_kcal", 2000))
    protein_target = float(st.session_state.get("protein_target", 120))

    kcal_progress = min(1.0, max(0.0, total_kcal / target_kcal)) if target_kcal > 0 else 0.0
    p_progress = min(1.0, max(0.0, total_p / protein_target)) if protein_target > 0 else 0.0

    m1, m2 = st.columns(2)
    m1.metric("Calories", f"{total_kcal:.0f} kcal")
    m2.metric("Protein", f"{total_p:.0f} g")

    st.caption(f"Calories progress (target {int(target_kcal)} kcal)")
    st.progress(kcal_progress)

    st.caption(f"Protein progress (target {int(protein_target)} g)")
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



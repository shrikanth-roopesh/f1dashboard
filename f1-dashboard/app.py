import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import google.generativeai as genai


st.set_page_config(page_title="F1 Race Strategy Intelligence", layout="wide")
st.title("F1 Race Strategy Intelligence")
with st.sidebar:
    st.markdown("### F1 Race Strategy Intelligence")
    st.markdown("A tire strategy decision support tool built on official F1 timing data.")
    st.markdown("**Built with:** Python · FastF1 · pandas · Streamlit")
    st.markdown("---")
    st.markdown("**Shrikanth Roopesh**")
    st.markdown("[LinkedIn](https://linkedin.com/in/shrikanth-roopesh)")
    st.markdown("[GitHub](https://github.com/shrikanth2406/f1-strategy-intelligence)")
    st.markdown("[Portfolio](https://shrikanth-roopesh.netlify.app)")

df = pd.read_csv(os.path.join(BASE_DIR, 'f1_laps.csv'))

# Adding filters 
col1, col2, col3 = st.columns(3)
with col1:
    race = st.selectbox("Race", ["All"] + sorted(df['Race'].unique().tolist()))
with col2:
    driver = st.selectbox("Driver", ["All"] + sorted(df['Driver'].unique().tolist()))
with col3:
    compound = st.selectbox("Compound", ["All"] + sorted(df[df['Compound'].notna()]['Compound'].unique().tolist()))

filtered = df.copy()
if race != "All":
    filtered = filtered[filtered['Race'] == race]
if driver != "All":
    filtered = filtered[filtered['Driver'] == driver]
if compound != "All":
    filtered = filtered[filtered['Compound'] == compound]


# KPI Cards

clean = filtered[filtered['LapTimeSeconds'] > 60]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Laps", f"{len(clean):,}")
col2.metric("Races", filtered['Race'].nunique())
col3.metric("Drivers", filtered['Driver'].nunique())
col4.metric("Avg Lap Time", f"{clean['LapTimeSeconds'].mean():.1f}s")

st.markdown("---")

st.subheader("Tire Degradation By Compound")
deg = clean.groupby(['Compound', 'TyreLife'])['LapTimeSeconds'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#0E0E0E')
ax.set_facecolor('#0E0E0E')
colors = {'SOFT': '#E10600', 'MEDIUM': '#F59E0B', 'HARD': '#2563EB'}
for comp in ['SOFT', 'MEDIUM', 'HARD']:
    subset = deg[deg['Compound'] == comp]
    if len(subset) > 0:
        ax.plot(subset['TyreLife'], subset['LapTimeSeconds'], label=comp, color=colors[comp], linewidth=2)
ax.set_xlabel('Tire Age (laps)', color='white')
ax.set_ylabel('Avg Lap Time (seconds)', color='white')
ax.tick_params(colors='white')
ax.legend()
for spine in ax.spines.values():
    spine.set_color('#444')
st.pyplot(fig)

st.markdown("---")

st.subheader("Driver Consistency Leaderboard")
driver_stats = clean.groupby(['Driver', 'Compound'])['LapTimeSeconds'].agg(['mean', 'std', 'count']).reset_index()
driver_stats.columns = ['Driver', 'Compound', 'Avg Lap', 'Std Dev', 'Laps']
driver_stats['CV'] = (driver_stats['Std Dev'] / driver_stats['Avg Lap']).round(4)
driver_stats['Avg Lap'] = driver_stats['Avg Lap'].round(2)
driver_stats = driver_stats[driver_stats['Laps'] >= 8].sort_values('CV')
st.dataframe(
    driver_stats[['Driver', 'Compound', 'Avg Lap', 'CV', 'Laps']].head(15),
    use_container_width=True, hide_index=True
)

st.markdown("---")

st.subheader("Stint Length By Compound")
stint_lengths = clean.groupby(['Driver', 'Race', 'Stint', 'Compound']).size().reset_index(name='StintLength')
avg_stint = stint_lengths.groupby('Compound')['StintLength'].mean().reset_index()
avg_stint = avg_stint.sort_values('StintLength')
st.bar_chart(avg_stint, x='Compound', y='StintLength', horizontal=True, color="#E10600")

st.markdown("---")

st.subheader("Pit Stop Count By Driver")
df_sorted = filtered.sort_values(['Driver', 'Race', 'LapNumber'])
df_sorted['StintChange'] = df_sorted.groupby(['Driver', 'Race'])['Stint'].diff().fillna(0) != 0
pit_counts = df_sorted[df_sorted['StintChange']].groupby('Driver').size().reset_index(name='PitStops')
pit_counts = pit_counts.sort_values('PitStops', ascending=False).head(10)
st.bar_chart(pit_counts, x='Driver', y='PitStops', color="#E10600")

# Chatbot

st.markdown("---")
st.subheader("Ask The Data")
st.caption("Ask a question about the race data for data backed answers.")

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-flash-latest")

# Tell the model what your database looks like
SCHEMA = """
Table: laps
Columns:
- Driver (text, 3-letter code like VER, HAM, LEC)
- Team (text)
- LapNumber (integer)
- LapTime (text)
- LapTimeSeconds (float, lap time in seconds)
- Compound (text: SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
- TyreLife (integer, how many laps old the tire is)
- Stint (integer, which stint this lap belongs to)
- Race (text: Bahrain, Monaco, Monza)
"""

st.caption("Try one of these:")
c1, c2, c3 = st.columns(3)
examples = [
    "Average lap time by compound?",
    "Which driver was fastest?",
    "Most laps on soft tires?"
]
for col, ex in zip([c1, c2, c3], examples):
    if col.button(ex):
        st.session_state.preset_question = ex

question = st.text_input("Your question:", value=st.session_state.get("preset_question", ""),
                         placeholder="Which driver had the most consistent tire management?")

if question:
    with st.spinner("Thinking..."):
        # Step 1: ask Gemini to write SQL
        sql_prompt = f"""You are a SQL expert. Given this SQLite schema:
{SCHEMA}

Write a single SQLite query to answer this question: "{question}"

Rules:
- Return ONLY the SQL query, no explanation, no markdown, no backticks.
- Only use the 'laps' table and the columns listed.
- If the question cannot be answered from this data, return exactly: CANNOT_ANSWER
- Filter out invalid laps with WHERE LapTimeSeconds > 60 when dealing with lap times.
"""
        sql_response = model.generate_content(sql_prompt).text.strip()
        sql_response = sql_response.replace("```sql", "").replace("```", "").strip()

        # Guardrail: if Gemini can't answer, say so
        if "CANNOT_ANSWER" in sql_response:
            st.warning("I can't answer that from the available race data. Try asking about lap times, tire compounds, drivers, stints, or pit strategy.")
        else:
            try:
                # Step 2: run the SQL on your database
                conn = sqlite3.connect(os.path.join(BASE_DIR, "f1_strategy.db"))
                result = pd.read_sql_query(sql_response, conn)
                conn.close()

                if result.empty:
                    st.warning("The query ran but found no matching data. Try rephrasing your question.")
                else:
                    # Step 3: ask Gemini to explain the result in plain English
                    explain_prompt = f"""The user asked: "{question}"
The data returned this result:
{result.to_string()}

Explain this answer in 2-3 clear sentences for a casual F1 fan. Be specific with numbers. Do not make up anything not in the data."""
                    answer = model.generate_content(explain_prompt).text.strip()

                    st.markdown(f"**Answer:** {answer}")
                    with st.expander("See the data and query"):
                        st.code(sql_response, language="sql")
                        st.dataframe(result, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error("I couldn't run that query. Try rephrasing your question.")
                st.caption(f"(Technical detail: {e})")
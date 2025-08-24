import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import sqlite3
import json
import io
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Journaling Co-Pilot",
    page_icon="🧠",
    layout="wide"
)

# --- Initialize Session State ---
if 'api_status' not in st.session_state:
    st.session_state.api_status = "Disconnected"
    st.session_state.model = None

# --- Database Setup ---
DB_FILE = "journal.db"

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS wheel_of_life (id INTEGER PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, scores TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS eisenhower_tasks (id INTEGER PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, tasks TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS journal_entries (id INTEGER PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, entry_type TEXT, content TEXT)''')
    conn.commit()
    conn.close()

# --- Database Helper Functions ---
def save_data(table, data):
    """Saves data to the specified table in the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if table == "wheel_of_life": c.execute("INSERT INTO wheel_of_life (scores) VALUES (?)", (json.dumps(data),))
    elif table == "eisenhower_tasks": c.execute("INSERT INTO eisenhower_tasks (tasks) VALUES (?)", (json.dumps(data),))
    elif table == "journal_entries": c.execute("INSERT INTO journal_entries (entry_type, content) VALUES (?, ?)", (data['type'], json.dumps(data['content'])))
    conn.commit()
    conn.close()

def load_latest_data(table):
    """Loads the most recent entry from a specified table."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    query_map = {
        "wheel_of_life": "SELECT scores FROM wheel_of_life ORDER BY timestamp DESC LIMIT 1",
        "eisenhower_tasks": "SELECT tasks FROM eisenhower_tasks ORDER BY timestamp DESC LIMIT 1",
        "five_minute_journal": "SELECT content FROM journal_entries WHERE entry_type = 'five_minute' ORDER BY timestamp DESC LIMIT 1",
        "general_journal": "SELECT content FROM journal_entries WHERE entry_type = 'general' ORDER BY timestamp DESC LIMIT 1"
    }
    c.execute(query_map.get(table))
    row = c.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None

def get_user_data_summary():
    """Fetches a summary of all recent user data to provide context to the AI."""
    summary = "Here is a summary of the user's recent journal data:\n\n"
    latest_wheel = load_latest_data("wheel_of_life")
    if latest_wheel:
        summary += "## Latest Wheel of Life Scores:\n" + "\n".join([f"- {cat}: {score}/10" for cat, score in latest_wheel.items()]) + "\n\n"
    latest_tasks = load_latest_data("eisenhower_tasks")
    if latest_tasks and any(latest_tasks.values()):
        summary += "## Latest Eisenhower Matrix Tasks:\n"
        for quadrant, tasks in latest_tasks.items():
            if tasks:
                summary += f"### {quadrant}:\n" + "\n".join([f"- {task}" for task in tasks]) + "\n"
        summary += "\n"
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT entry_type, content FROM journal_entries ORDER BY timestamp DESC LIMIT 3")
    entries = c.fetchall()
    conn.close()
    if entries:
        summary += "## Last 3 Journal Entries:\n"
        for entry_type, content in entries:
            summary += f"### Entry Type: {entry_type.replace('_', ' ').title()}\n"
            content_dict = json.loads(content)
            if isinstance(content_dict, dict):
                 summary += "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in content_dict.items() if value]) + "\n\n"
            else:
                 summary += f"- {content_dict}\n\n"
    return summary if len(summary) > 70 else "No data has been entered by the user yet."

# --- Gemini AI Configuration & Helpers ---
def configure_gemini(api_key):
    """Configures the Gemini model and returns the model object and status."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Perform a quick test to see if the key is valid
        model.generate_content("test", request_options={'timeout': 10}) 
        return model, "Connected"
    except Exception as e:
        if "API_KEY_INVALID" in str(e):
            return None, "API Key is invalid."
        return None, "Connection failed."

def get_ai_response(model, system_instruction, user_prompt):
    try:
        response = model.generate_content(f"{system_instruction}\n\nUSER PROMPT: {user_prompt}")
        return response.text
    except Exception as e: return f"AI Error: {e}"

def get_improvement_tips(model, low_scores):
    if not model or not low_scores: return ""
    context = "The user has identified areas with low satisfaction:\n"
    context += "\n".join([f"- {area}: {score}/10" for area, score in low_scores.items()])
    prompt = f"As a life coach, provide 3-5 concise, actionable tips for these areas. Use markdown headings for each area."
    return get_ai_response(model, f"{context}\n\n{prompt}", "")

# --- HTML Export Function ---
def generate_html_report(tool_name, data):
    """Generates a self-contained HTML report for the given tool and data."""
    # --- CSS Styling ---
    css = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background-color: #f0f2f6; }
        .container { max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1, h2 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }
        h1 { font-size: 2.5em; text-align: center; color: #007bff; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #007bff; color: white; }
        ul { list-style-type: none; padding: 0; }
        li { background: #f8f9fa; margin-bottom: 10px; padding: 15px; border-radius: 4px; border-left: 5px solid #007bff; }
        img { max-width: 100%; height: auto; display: block; margin: 20px auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """
    
    # --- HTML Body Generation ---
    body_content = ""
    if data:
        if tool_name == "Wheel of Life":
            scores = data
            categories = list(scores.keys())
            # Generate and embed chart
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            score_values = list(scores.values()) + [scores[categories[0]]]
            angles += angles[:1]
            ax.fill(angles, score_values, color='teal', alpha=0.25); ax.plot(angles, score_values, color='teal', linewidth=2)
            ax.set_yticklabels([]); ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories); ax.set_rlim(0, 10)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png'); plt.close(fig)
            chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            body_content += f'<h2>Chart</h2><img src="data:image/png;base64,{chart_base64}" alt="Wheel of Life Chart">'
            # Scores table
            body_content += "<h2>Scores</h2><table><tr><th>Category</th><th>Score</th></tr>"
            for cat, score in scores.items():
                body_content += f"<tr><td>{cat}</td><td>{score} / 10</td></tr>"
            body_content += "</table>"
        elif tool_name == "Eisenhower Matrix":
            tasks = data
            body_content += "<h2>Tasks by Quadrant</h2>"
            for quadrant, task_list in tasks.items():
                body_content += f"<h3>{quadrant.replace('&', ' & ')}</h3>"
                if task_list:
                    body_content += "<ul>" + "".join([f"<li>{task}</li>" for task in task_list]) + "</ul>"
                else:
                    body_content += "<p>No tasks in this quadrant.</p>"
        elif tool_name == "The Five-Minute Journal":
            entry = data
            body_content += "<h2>Morning</h2><ul>"
            for key in ["grateful_1", "great_today", "affirmation"]:
                if entry.get(key): body_content += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {entry[key]}</li>"
            body_content += "</ul><h2>Evening</h2><ul>"
            for key in ["amazing_things", "better_today"]:
                if entry.get(key): body_content += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {entry[key]}</li>"
            body_content += "</ul>"
        elif tool_name == "General Journal Entry":
            entry = data
            body_content += "<h2>Entry</h2>"
            body_content += f"<p style='white-space: pre-wrap; padding: 15px; background: #f8f9fa; border-radius: 4px;'>{entry}</p>"
    else:
        body_content = "<p>No data available to generate a report. Please enter some data in the tool first.</p>"

    # --- Final Assembly ---
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{tool_name} Report</title>
        {css}
    </head>
    <body>
        <div class="container">
            <h1>{tool_name} Report</h1>
            <p>Generated on: {datetime.now().strftime('%B %d, %Y, %I:%M %p')}</p>
            {body_content}
        </div>
    </body>
    </html>
    """
    return html

# --- UI for Tools ---
def tool_wheel_of_life(model):
    st.header("Wheel of Life")
    if 'wheel_scores' not in st.session_state:
        st.session_state.wheel_scores = load_latest_data("wheel_of_life") or {}
    
    categories = ["Career", "Finances", "Health", "Growth", "Relationships", "Romance", "Fun", "Environment"]
    cols = st.columns(4)
    for i, cat in enumerate(categories):
        with cols[i % 4]:
            st.session_state.wheel_scores[cat] = st.slider(cat, 1, 10, st.session_state.wheel_scores.get(cat, 5))
    
    if st.button("📊 Save & Show Analysis"):
        save_data("wheel_of_life", st.session_state.wheel_scores); st.success("Saved!")
        st.session_state.show_analysis = True

    if st.session_state.get('show_analysis'):
        current_scores = st.session_state.wheel_scores
        st.subheader("Analysis")
        # Display chart
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        score_values = list(current_scores.values()) + [current_scores[categories[0]]]
        angles += angles[:1]
        ax.fill(angles, score_values, color='teal', alpha=0.25); ax.plot(angles, score_values, color='teal', linewidth=2)
        ax.set_yticklabels([]); ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories); ax.set_rlim(0, 10)
        st.pyplot(fig)
        
        # AI Tips
        if model:
            low_scores = {area: score for area, score in current_scores.items() if score <= 5}
            if low_scores:
                with st.spinner("Generating improvement tips..."):
                    tips = get_improvement_tips(model, low_scores); st.markdown(tips)
            else:
                st.balloons(); st.success("Great job! You have a well-balanced wheel.")
    return st.session_state.wheel_scores

def tool_eisenhower_matrix(model):
    st.header("Eisenhower Matrix")
    if 'matrix_tasks' not in st.session_state:
        st.session_state.matrix_tasks = load_latest_data("eisenhower_tasks") or {"Urgent & Important": [], "Not Urgent & Important": [], "Urgent & Not Important": [], "Not Urgent & Not Important": []}
    
    tasks = st.session_state.matrix_tasks
    c1, c2 = st.columns(2)
    quadrants = {"Urgent & Important": ("🟢 Do First", c1), "Not Urgent & Important": ("🔵 Schedule", c2), "Urgent & Not Important": ("🟡 Delegate", c1), "Not Urgent & Not Important": ("🔴 Delete", c2)}
    for key, (title, col) in quadrants.items():
        with col.container(border=True):
            st.subheader(title)
            task = st.text_input(f"Add to '{key}'", key=f"task_{key}")
            if st.button("Add", key=f"add_{key}"): tasks[key].append(task); st.rerun()
            for t in tasks[key]: st.markdown(f"- {t}")
    if st.button("💾 Save All Tasks"): save_data("eisenhower_tasks", tasks); st.success("Saved!")
    return tasks

def tool_five_minute_journal():
    st.header("The Five-Minute Journal")
    if '5min_entry' not in st.session_state:
        st.session_state['5min_entry'] = load_latest_data("five_minute_journal") or {}
    
    entry = st.session_state['5min_entry']
    with st.form("five_minute_form"):
        st.subheader("☀️ Morning")
        entry["grateful_1"] = st.text_input("I am grateful for...", entry.get("grateful_1", ""))
        entry["great_today"] = st.text_input("What would make today great?", entry.get("great_today", ""))
        entry["affirmation"] = st.text_input("Daily affirmation. I am...", entry.get("affirmation", ""))
        st.subheader("🌙 Evening")
        entry["amazing_things"] = st.text_area("3 Amazing things that happened today...", entry.get("amazing_things", ""))
        entry["better_today"] = st.text_area("How could I have made today even better?", entry.get("better_today", ""))
        
        if st.form_submit_button("💾 Save Journal Entry"):
            save_data("journal_entries", {"type": "five_minute", "content": entry}); st.success("Saved!")
    return entry

def tool_general_journal():
    st.header("General Journal Entry")
    if 'general_entry' not in st.session_state:
        st.session_state['general_entry'] = load_latest_data("general_journal") or "What's on your mind today?"
    
    entry_text = st.session_state['general_entry']
    with st.form("general_journal_form"):
        entry_text_area = st.text_area("Your thoughts:", value=entry_text, height=300)
        if st.form_submit_button("💾 Save Entry"):
            st.session_state['general_entry'] = entry_text_area
            save_data("journal_entries", {"type": "general", "content": entry_text_area}); st.success("Saved!")
    return st.session_state['general_entry']

# --- Main App ---
init_db()
st.title("Intelligent Journaling Co-Pilot 🧠")

with st.sidebar:
    st.header("Configuration")
    
    def update_api_status():
        api_key = st.session_state.api_key_input
        if api_key:
            model, status = configure_gemini(api_key)
            st.session_state.model = model
            st.session_state.api_status = status
        else:
            st.session_state.api_status = "Disconnected"
            st.session_state.model = None

    st.text_input(
        "Enter your Google Gemini API Key", 
        type="password", 
        key="api_key_input", 
        on_change=update_api_status
    )

    status_indicator = st.empty()
    if st.session_state.api_status == "Connected":
        status_indicator.success("🟢 Connected to Gemini")
    elif st.session_state.api_status == "Disconnected":
        status_indicator.warning("🔴 Disconnected")
    else:
        status_indicator.error(f"🔴 {st.session_state.api_status}")

    st.divider()

    st.header("Tools")
    tool_selection = st.radio("Select a Tool:", ("Wheel of Life", "Eisenhower Matrix", "The Five-Minute Journal", "General Journal Entry"), key="tool_selector")
    
    st.divider()
    
    # --- Export Section ---
    st.header("Export Current View")
    st.info("The button below will export the data currently displayed on the main page.")
    
    # We will get the current data later in the script
    st.session_state.current_tool_data = None
    
    # This will be populated after the tool runs
    download_button_placeholder = st.empty()
    
    st.divider()

    st.header("💬 AI Co-Pilot")
    st.info("Ask me for insights on your saved data!")
    
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])
    if prompt := st.chat_input("Ask about your data..."):
        if not st.session_state.model: 
            st.info("Please connect with a valid API key to chat.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    context = get_user_data_summary()
                    instruction = "You are the 'Journaling Co-Pilot'. Provide insights based *only* on the user's data provided. If asked about unrelated topics, politely decline."
                    response = get_ai_response(st.session_state.model, f"{instruction}\n\nUSER DATA:\n{context}", prompt)
                    st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# --- Main Content Area ---
model = st.session_state.model
current_data_for_export = None

if tool_selection == "Wheel of Life":
    current_data_for_export = tool_wheel_of_life(model)
elif tool_selection == "Eisenhower Matrix":
    current_data_for_export = tool_eisenhower_matrix(model)
elif tool_selection == "The Five-Minute Journal":
    current_data_for_export = tool_five_minute_journal()
elif tool_selection == "General Journal Entry":
    current_data_for_export = tool_general_journal()

# --- Populate the Download Button in the Sidebar ---
if current_data_for_export:
    report_html = generate_html_report(tool_selection, current_data_for_export)
    download_button_placeholder.download_button(
        label=f"⬇️ Download {tool_selection} Report",
        data=report_html,
        file_name=f"{tool_selection.lower().replace(' ', '_')}_report.html",
        mime="text/html",
    )
Of course. Here is a comprehensive `README.md` file for the Intelligent Journaling Co-Pilot application we've built. You can save this directly into your project folder.

-----

# Intelligent Journaling Co-Pilot 🧠

An AI-powered, local journaling application built with Streamlit and Google Gemini. This tool goes beyond a simple digital notebook, acting as an interactive partner for your self-reflection and personal growth journey.

## Features

  * ✅ **Multi-Tool Journaling**: Switch between different proven methods:
      * **Wheel of Life**: For a high-level view of life balance.
      * **Eisenhower Matrix**: To prioritize tasks effectively.
      * **The Five-Minute Journal**: For structured daily gratitude and intention.
      * **General Journal**: For free-form expressive writing.
  * ✅ **Persistent Storage**: Your data is saved locally in a `journal.db` (SQLite) file, so your journal entries persist across sessions.
  * ✅ **AI Co-Pilot**: A sidebar chatbot that can access your saved data to provide personalized insights and answer questions about your progress.
  * ✅ **Proactive AI Insights**: The Wheel of Life tool automatically generates actionable tips from the AI to help you improve in low-scoring areas.
  * ✅ **HTML Export**: Download a clean, self-contained HTML report of your current tool's view, perfect for printing or archiving.
  * ✅ **Real-time API Validation**: Get immediate feedback on your Gemini API key to know if you're connected.

-----

## Technology Stack

  * **Framework**: Streamlit
  * **AI Language Model**: Google Gemini API (`google-generativeai`)
  * **Database**: SQLite3
  * **Data Visualization**: Matplotlib

-----

## Setup and Installation

Follow these steps to get the application running on your local machine.

### 1\. Prerequisites

  * Python 3.8 or higher
  * An active Google account

### 2\. Clone the Repository

Clone this project to your local machine:

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 3\. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

  * **On macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
  * **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

### 4\. Install Dependencies

Create a file named `requirements.txt` in the project directory and paste the following lines into it:

```
streamlit
google-generativeai
matplotlib
numpy
```

Now, install all the required libraries at once:

```bash
pip install -r requirements.txt
```

### 5\. Get Your Google Gemini API Key

1.  Go to [Google AI Studio](https://aistudio.google.com/).
2.  Sign in with your Google account.
3.  Click on "**Get API key**" and then "**Create API key in new project**".
4.  Copy the generated key. You will need to paste this into the app's sidebar.

### 6\. Run the Application

Once the dependencies are installed, run the Streamlit app from your terminal:

```bash
streamlit run app.py
```

Your web browser should automatically open a new tab with the application running.

-----

## How to Use

1.  **Enter API Key**: Paste your Gemini API key into the input box in the sidebar. A status indicator will confirm if the connection is successful.
2.  **Select a Tool**: Use the radio buttons in the sidebar to choose a journaling tool.
3.  **Enter Your Data**: Interact with the tool on the main page. Fill out your scores, tasks, or journal entries.
4.  **Save Your Progress**: Click the "Save" button within a tool to persist your data to the local `journal.db` file.
5.  **Get AI Insights**: Use the "AI Co-Pilot" chat in the sidebar to ask questions about your saved data (e.g., "What patterns do you see in my journal?").
6.  **Export Your View**: Use the "Export Current View" button in the sidebar to download a clean HTML report of the data currently displayed on the page.
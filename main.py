import os
import json
import streamlit as st
from typing import Union
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
from datetime import datetime

class Color(BaseModel):
    r: int
    g: int
    b: int

class Graph_Point(BaseModel):
    x: int
    y: int

class Graph_Line(BaseModel):
    line_title: str
    line_color: Color
    points: list[Graph_Point]

class Chart(BaseModel):
    chart_title: str
    x_axis_label: str
    y_axis_label: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    lines: list[Graph_Line]

class Output(BaseModel):
    work: str
    response: str
    chart: Chart
    display_chart: bool

def get_uploaded_csv_data(uploaded_files):
    data_paths = []

    for uploaded_file in uploaded_files:
        try:
            content = uploaded_file.read().decode('utf-8')
            data_paths.append({
                "filename": uploaded_file.name,
                "content": content
            })
        except Exception as e:
            st.error(f"Error reading file {uploaded_file.name}: {e}")

    return data_paths

def create_chart_from_ai_output(chart_data: dict):
    try:
        chart = Chart(**chart_data)

        fig, ax = plt.subplots(figsize=(12, 8))

        for line in chart.lines:
            x_coords = [point.x for point in line.points]
            y_coords = [point.y for point in line.points]

            color = (line.line_color.r/255, line.line_color.g/255, line.line_color.b/255)

            ax.plot(x_coords, y_coords,
                   label=line.line_title,
                   color=color,
                   marker='o',
                   linewidth=2,
                   markersize=6)

        ax.set_title(chart.chart_title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(chart.x_axis_label, fontsize=12)
        ax.set_ylabel(chart.y_axis_label, fontsize=12)

        ax.set_xlim(chart.x_min, chart.x_max)
        ax.set_ylim(chart.y_min, chart.y_max)

        ax.grid(True, alpha=0.3)

        ax.legend(fontsize=10)

        plt.tight_layout()

        return fig

    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def initialize_gemini():
    load_dotenv()

    api_key = os.getenv("KEY") or st.secrets.get("KEY")

    if not api_key:
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    return model

def generate_chart_from_query(model, data_paths, user_query):
    formatted_content = f"Here are {len(data_paths)} CSV files:\n\n"
    for i, item in enumerate(data_paths, 1):
        formatted_content += f"File {i}: {item['filename']}\n"
        formatted_content += f"```csv\n{item['content']}\n```\n\n"

    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=Output
    )

    prompt = f"""
    {formatted_content}

    User Query: "{user_query}"

    Based on the user's query and the CSV data provided, create an appropriate chart. Return the response as a JSON object that matches the Chart schema exactly.

    Requirements:
    - Analyze the user's query to determine what type of visualization they want
    - Use the Chart schema structure
    - Include proper x/y axis labels relevant to the query
    - Set appropriate min/max values for axes based on the data range (add some padding)
    - Create data points from the CSV data that best answer the user's question
    - Use meaningful colors for lines (RGB values 0-255)
    - Title should describe what the chart shows in relation to the user's query
    - For time-based data, use appropriate time intervals for x-axis
    - Ensure all numeric values are integers as required by the schema
    - Show your analytical work in the "work" field
    - Double check the accuracy of all calculations
    - Include a clear summary explaining the chart in the "response" field
    - Set display_chart to true if a meaningful chart can be created, false otherwise

    If the query cannot be answered with a meaningful chart from the available data, explain why in the response field and set display_chart to false.
    """

    try:
        response = model.generate_content(
            contents=[prompt],
            generation_config=generation_config
        )

        chart_output = json.loads(response.text)
        return chart_output

    except Exception as e:
        st.error(f"Error generating content: {e}")
        return None

def main():
    st.set_page_config(
        page_title="AI Chart Generator",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 AI-Powered Chart Generator")
    st.markdown("Upload your CSV files and chat with AI to generate insightful charts!")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = []

    with st.sidebar:
        st.header("📁 Upload CSV Files")
        uploaded_files = st.file_uploader(
            "",
            accept_multiple_files=True,
            type=['csv'],
            help="Upload one or more CSV files to analyze"
        )

        if uploaded_files:
            st.session_state.csv_data = get_uploaded_csv_data(uploaded_files)
            st.success(f"✅ Loaded {len(st.session_state.csv_data)} CSV file(s)")

            with st.expander("📋 File Previews"):
                for data in st.session_state.csv_data:
                    st.write(f"**{data['filename']}**")
                    try:
                        df = pd.read_csv(io.StringIO(data['content']))
                        st.dataframe(df.head(3), use_container_width=True)
                    except:
                        st.text(data['content'][:200] + "..." if len(data['content']) > 200 else data['content'])
                    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Chat Interface")

        chat_container = st.container()
        with chat_container:
            for i, chat in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(chat['user_message'])

                with st.chat_message("assistant"):
                    st.write(chat['ai_response'])
                    if chat.get('chart_fig'):
                        st.pyplot(chat['chart_fig'], use_container_width=True)

        user_input = st.chat_input("Ask me to create a chart from your data...")

        if user_input and st.session_state.csv_data:
            model = initialize_gemini()

            if not model:
                st.error("❌ Please provide a valid API key in the sidebar or set KEY in your .env file")
                return

            with st.spinner("🤖 AI is analyzing your data and generating chart..."):
                chart_output = generate_chart_from_query(model, st.session_state.csv_data, user_input)

                if chart_output:
                    ai_response = chart_output['response']

                    chart_fig = None
                    if chart_output['display_chart']:
                        chart_fig = create_chart_from_ai_output(chart_output['chart'])

                    st.session_state.chat_history.append({
                        'user_message': user_input,
                        'ai_response': ai_response,
                        'chart_fig': chart_fig,
                        'work_shown': chart_output.get('work', ''),
                        'timestamp': datetime.now()
                    })

                    st.rerun()

        elif user_input and not st.session_state.csv_data:
            st.warning("⚠️ Please upload CSV files first!")

if __name__ == "__main__":
    main()

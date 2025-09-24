import os
from typing import Union
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel

# Move classes outside the function for better accessibility
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

def get_csv_files(directory_path="./sample_data"):
    """Get all CSV files from the specified directory"""
    csv_files = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv") and os.path.isfile(os.path.join(directory_path, filename)):
            csv_files.append(os.path.join(directory_path, filename))
    return csv_files

# Get CSV files
file_paths = get_csv_files()
data_paths = []

for file_path in file_paths:
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            data_paths.append({
                "filename": os.path.basename(file_path),
                "content": content
            })
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
    except Exception as e:
        print(f"Warning: Error reading file {file_path}: {e}")

# Check if we found any files
if not data_paths:
    print("No CSV files found or processed successfully")
    exit()

formatted_content = f"Here are {len(data_paths)} CSV files:\n\n"
for i, item in enumerate(data_paths, 1):
    formatted_content += f"File {i}: {item['filename']}\n"
    formatted_content += f"```csv\n{item['content']}\n```\n\n"

load_dotenv()
api_key = os.getenv("KEY")
if not api_key:
    raise ValueError("KEY not found in environment variables")

# Configure with API key
genai.configure(api_key=api_key)

# Use a more recent model that supports structured output
model = genai.GenerativeModel('gemini-2.5-flash')

# Create generation config with response schema
generation_config = genai.GenerationConfig(
    response_mime_type="application/json",  # Explicitly request JSON
    response_schema=Chart
)

# Enhanced prompt for better structured response
prompt = f"""
{formatted_content}

Create a chart showing budget data over time. Return the response as a JSON object that matches the Chart schema exactly.

Requirements:
- Use the Chart schema structure
- Include proper x/y axis labels
- Set appropriate min/max values for axes
- Create data points from the CSV data
- Use meaningful colors for lines (RGB values 0-255)
- Title should describe what the chart shows

Make a graph of the budget over time.
"""

try:
    response = model.generate_content(
        contents=[prompt],
        generation_config=generation_config
    )

    print("Raw response:")
    print(response.text)

    # Try to parse as JSON to verify structure
    import json
    try:
        parsed = json.loads(response.text)
        print("\nSuccessfully parsed as JSON!")
        print("Chart title:", parsed.get('chart_title', 'Not found'))
    except json.JSONDecodeError as e:
        print(f"\nFailed to parse as JSON: {e}")

except Exception as e:
    print(f"Error generating content: {e}")

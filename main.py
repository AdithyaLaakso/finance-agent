import os
import json
from typing import Union
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel
import matplotlib.pyplot as plt
import numpy as np

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

class Output(BaseModel):
    work: str
    response: str
    chart: Chart
    display_chart: bool

def get_csv_files(directory_path="./sample_data"):
    """Get all CSV files from the specified directory"""
    csv_files = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv") and os.path.isfile(os.path.join(directory_path, filename)):
            csv_files.append(os.path.join(directory_path, filename))
    return csv_files

def create_chart_from_ai_output(chart_data: dict, save_path: str = None):
    """Create a matplotlib chart from AI-generated chart data"""
    try:
        # Parse the chart data using Pydantic model
        chart = Chart(**chart_data)

        # Create the figure and axis
        plt.figure(figsize=(12, 8))

        # Plot each line
        for line in chart.lines:
            # Extract x and y coordinates
            x_coords = [point.x for point in line.points]
            y_coords = [point.y for point in line.points]

            # Convert RGB color to matplotlib format (0-1 range)
            color = (line.line_color.r/255, line.line_color.g/255, line.line_color.b/255)

            # Plot the line
            plt.plot(x_coords, y_coords,
                    label=line.line_title,
                    color=color,
                    marker='o',
                    linewidth=2,
                    markersize=6)

        # Set chart properties
        plt.title(chart.chart_title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(chart.x_axis_label, fontsize=12)
        plt.ylabel(chart.y_axis_label, fontsize=12)

        # Set axis limits
        plt.xlim(chart.x_min, chart.x_max)
        plt.ylim(chart.y_min, chart.y_max)

        # Add grid for better readability
        plt.grid(True, alpha=0.3)

        # Add legend
        plt.legend(fontsize=10)

        # Improve layout
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")

        # Show the plot
        plt.show()

        return True

    except Exception as e:
        print(f"Error creating chart: {e}")
        return False

def main():
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
        return

    # Format content for AI
    formatted_content = f"Here are {len(data_paths)} CSV files:\n\n"
    for i, item in enumerate(data_paths, 1):
        formatted_content += f"File {i}: {item['filename']}\n"
        formatted_content += f"```csv\n{item['content']}\n```\n\n"

    # Load environment and configure AI
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
        response_mime_type="application/json",
        response_schema=Output
    )

    # Enhanced prompt for better structured response
    prompt = f"""
    {formatted_content}

    Create a chart showing budget data over time. Return the response as a JSON object that matches the Chart schema exactly.

    Requirements:
    - Use the Chart schema structure
    - Include proper x/y axis labels
    - Set appropriate min/max values for axes based on the data range (add some padding)
    - Create data points from the CSV data
    - Use meaningful colors for lines (RGB values 0-255)
    - Title should describe what the chart shows
    - For time-based data, use appropriate time intervals for x-axis
    - Ensure all numeric values are integers as required by the schema
    - Show your work in the "work" field
    - Double check the accuracy of all calculations
    - Include a summary of the work in the "response" field

    Show Gross Margin % trend for the last 3 months.
    """

    try:
        print("Generating chart data with AI...")
        response = model.generate_content(
            contents=[prompt],
            generation_config=generation_config
        )

        print("Raw AI response:")
        print(response.text)
        print("-" * 50)

        # Parse the JSON response
        try:
            print(response.text)
            chart_data = json.loads(response.text)['chart']
            print("Successfully parsed AI response as JSON!")
            print(f"Chart title: {chart_data.get('chart_title', 'Not found')}")
            print(f"Number of lines: {len(chart_data.get('lines', []))}")

            # Create the actual chart
            print("\nCreating visual chart...")
            success = create_chart_from_ai_output(
                chart_data,
                save_path="ai_generated_chart.png"
            )

            if success:
                print("Chart created successfully!")
            else:
                print("Failed to create chart")

        except json.JSONDecodeError as e:
            print(f"Failed to parse AI response as JSON: {e}")
            print("Raw response:", response.text)

    except Exception as e:
        print(f"Error generating content: {e}")

if __name__ == "__main__":
    main()

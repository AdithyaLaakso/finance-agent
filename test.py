import pytest
import json
import os
import io
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import ValidationError

from main import (
    Color, Graph_Point, Graph_Line, Chart, Output,
    get_uploaded_csv_data, create_chart_from_ai_output,
    initialize_gemini, generate_chart_from_query
)

class TestPydanticModels:
    def test_color_model_valid(self):
        color = Color(r=255, g=128, b=0)
        assert color.r == 255
        assert color.g == 128
        assert color.b == 0

    def test_graph_point_model(self):
        point = Graph_Point(x=10, y=20)
        assert point.x == 10
        assert point.y == 20

    def test_graph_line_model(self):
        color = Color(r=255, g=0, b=0)
        points = [Graph_Point(x=1, y=2), Graph_Point(x=3, y=4)]
        line = Graph_Line(
            line_title="Test Line",
            line_color=color,
            points=points
        )
        assert line.line_title == "Test Line"
        assert len(line.points) == 2
        assert line.points[0].x == 1

    def test_chart_model_complete(self):
        color = Color(r=255, g=0, b=0)
        points = [Graph_Point(x=1, y=10), Graph_Point(x=2, y=20)]
        line = Graph_Line(line_title="Revenue", line_color=color, points=points)

        chart = Chart(
            chart_title="Monthly Revenue",
            x_axis_label="Month",
            y_axis_label="Revenue ($)",
            x_min=0,
            y_min=0,
            x_max=12,
            y_max=100,
            lines=[line]
        )

        assert chart.chart_title == "Monthly Revenue"
        assert len(chart.lines) == 1
        assert chart.x_max == 12

    def test_output_model(self):
        color = Color(r=0, g=255, b=0)
        points = [Graph_Point(x=1, y=5)]
        line = Graph_Line(line_title="Test", line_color=color, points=points)
        chart = Chart(
            chart_title="Test Chart",
            x_axis_label="X",
            y_axis_label="Y",
            x_min=0, y_min=0, x_max=10, y_max=10,
            lines=[line]
        )

        output = Output(
            work="Analysis complete",
            response="Chart shows test data",
            chart=chart,
            display_chart=True
        )

        assert output.work == "Analysis complete"
        assert output.display_chart is True


class TestCSVHandling:
    def create_mock_uploaded_file(self, filename, content):
        mock_file = Mock()
        mock_file.name = filename
        mock_file.read.return_value = content.encode('utf-8')
        return mock_file

    def test_get_uploaded_csv_data_single_file(self):
        csv_content = "Name,Age,Salary\nJohn,30,50000\nJane,25,45000"
        mock_file = self.create_mock_uploaded_file("test.csv", csv_content)

        result = get_uploaded_csv_data([mock_file])

        assert len(result) == 1
        assert result[0]['filename'] == "test.csv"
        assert result[0]['content'] == csv_content

    def test_get_uploaded_csv_data_multiple_files(self):
        csv_content1 = "A,B\n1,2\n3,4"
        csv_content2 = "X,Y\n5,6\n7,8"

        mock_file1 = self.create_mock_uploaded_file("file1.csv", csv_content1)
        mock_file2 = self.create_mock_uploaded_file("file2.csv", csv_content2)

        result = get_uploaded_csv_data([mock_file1, mock_file2])

        assert len(result) == 2
        assert result[0]['filename'] == "file1.csv"
        assert result[1]['filename'] == "file2.csv"

    def test_get_uploaded_csv_data_empty_list(self):
        result = get_uploaded_csv_data([])
        assert result == []

    def test_get_uploaded_csv_data_decode_error(self):
        mock_file = Mock()
        mock_file.name = "bad_file.csv"
        mock_file.read.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte')

        result = get_uploaded_csv_data([mock_file])
        assert result == []


class TestChartCreation:
    def create_sample_chart_data(self):
        return {
            "chart_title": "Sample Chart",
            "x_axis_label": "Time",
            "y_axis_label": "Value",
            "x_min": 0,
            "y_min": 0,
            "x_max": 10,
            "y_max": 100,
            "lines": [{
                "line_title": "Series 1",
                "line_color": {"r": 255, "g": 0, "b": 0},
                "points": [
                    {"x": 1, "y": 10},
                    {"x": 2, "y": 20},
                    {"x": 3, "y": 30}
                ]
            }]
        }

    def test_create_chart_from_ai_output_success(self):
        chart_data = self.create_sample_chart_data()

        fig = create_chart_from_ai_output(chart_data)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        ax = fig.get_axes()[0]
        assert ax.get_title() == "Sample Chart"
        assert ax.get_xlabel() == "Time"
        assert ax.get_ylabel() == "Value"

        plt.close(fig)

    def test_create_chart_from_ai_output_multiple_lines(self):
        chart_data = self.create_sample_chart_data()
        chart_data["lines"].append({
            "line_title": "Series 2",
            "line_color": {"r": 0, "g": 255, "b": 0},
            "points": [
                {"x": 1, "y": 15},
                {"x": 2, "y": 25},
                {"x": 3, "y": 35}
            ]
        })

        fig = create_chart_from_ai_output(chart_data)

        assert fig is not None
        ax = fig.get_axes()[0]
        lines = ax.get_lines()
        assert len(lines) == 2

        plt.close(fig)

    def test_create_chart_from_ai_output_invalid_data(self):
        invalid_data = {"invalid": "data"}

        fig = create_chart_from_ai_output(invalid_data)

        assert fig is None

    def test_create_chart_from_ai_output_empty_points(self):
        chart_data = self.create_sample_chart_data()
        chart_data["lines"][0]["points"] = []

        fig = create_chart_from_ai_output(chart_data)

        assert fig is not None
        plt.close(fig)


class TestChartGeneration:
    def create_sample_data_paths(self):
        return [{
            "filename": "sales.csv",
            "content": "Month,Revenue\nJan,10000\nFeb,15000\nMar,12000"
        }]

    def create_sample_ai_response(self):
        return {
            "work": "Analyzed sales data for Q1",
            "response": "Chart shows monthly revenue trend",
            "chart": {
                "chart_title": "Monthly Revenue",
                "x_axis_label": "Month",
                "y_axis_label": "Revenue",
                "x_min": 0,
                "y_min": 0,
                "x_max": 3,
                "y_max": 20000,
                "lines": [{
                    "line_title": "Revenue",
                    "line_color": {"r": 0, "g": 100, "b": 200},
                    "points": [
                        {"x": 1, "y": 10000},
                        {"x": 2, "y": 15000},
                        {"x": 3, "y": 12000}
                    ]
                }]
            },
            "display_chart": True
        }

    def test_generate_chart_from_query_success(self):
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.create_sample_ai_response())
        mock_model.generate_content.return_value = mock_response

        data_paths = self.create_sample_data_paths()
        user_query = "Show me revenue trends"

        result = generate_chart_from_query(mock_model, data_paths, user_query)

        assert result is not None
        assert result["display_chart"] is True
        assert result["chart"]["chart_title"] == "Monthly Revenue"
        mock_model.generate_content.assert_called_once()

    def test_generate_chart_from_query_json_error(self):
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "invalid json"
        mock_model.generate_content.return_value = mock_response

        data_paths = self.create_sample_data_paths()
        user_query = "Show me revenue trends"

        result = generate_chart_from_query(mock_model, data_paths, user_query)

        assert result is None

    def test_generate_chart_from_query_api_error(self):
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")

        data_paths = self.create_sample_data_paths()
        user_query = "Show me revenue trends"

        result = generate_chart_from_query(mock_model, data_paths, user_query)

        assert result is None


class TestIntegration:
    def test_full_chart_generation_pipeline(self):
        csv_content = "Date,Sales,Profit\n2023-01,1000,200\n2023-02,1200,250\n2023-03,1100,220"
        mock_file = Mock()
        mock_file.name = "financial_data.csv"
        mock_file.read.return_value = csv_content.encode('utf-8')
        data_paths = get_uploaded_csv_data([mock_file])

        assert len(data_paths) == 1
        assert data_paths[0]['filename'] == "financial_data.csv"
        assert "Sales,Profit" in data_paths[0]['content']

        chart_data = {
            "chart_title": "Sales & Profit Trends",
            "x_axis_label": "Month",
            "y_axis_label": "Amount ($)",
            "x_min": 0,
            "y_min": 0,
            "x_max": 4,
            "y_max": 1500,
            "lines": [{
                "line_title": "Sales",
                "line_color": {"r": 0, "g": 150, "b": 255},
                "points": [
                    {"x": 1, "y": 1000},
                    {"x": 2, "y": 1200},
                    {"x": 3, "y": 1100}
                ]
            }]
        }

        fig = create_chart_from_ai_output(chart_data)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        ax = fig.get_axes()[0]
        assert ax.get_title() == "Sales & Profit Trends"
        assert len(ax.get_lines()) == 1

        plt.close(fig)


@pytest.fixture
def sample_csv_content():
    return "Date,Revenue,Expenses\n2023-01,5000,3000\n2023-02,6000,3500\n2023-03,5500,3200"

@pytest.fixture
def sample_chart_data():
    return {
        "chart_title": "Revenue vs Expenses",
        "x_axis_label": "Month",
        "y_axis_label": "Amount",
        "x_min": 0,
        "y_min": 0,
        "x_max": 4,
        "y_max": 7000,
        "lines": [{
            "line_title": "Revenue",
            "line_color": {"r": 0, "g": 255, "b": 0},
            "points": [{"x": 1, "y": 5000}, {"x": 2, "y": 6000}, {"x": 3, "y": 5500}]
        }]
    }


class TestPerformance:
    def test_multiple_lines_chart_performance(self):
        chart_data = {
            "chart_title": "Multi-line Chart",
            "x_axis_label": "X",
            "y_axis_label": "Y",
            "x_min": 0,
            "y_min": 0,
            "x_max": 100,
            "y_max": 1000,
            "lines": []
        }

        for line_idx in range(10):
            points = [{"x": i, "y": i * (line_idx + 1)} for i in range(100)]
            chart_data["lines"].append({
                "line_title": f"Series {line_idx + 1}",
                "line_color": {"r": line_idx * 25, "g": 100, "b": 200},
                "points": points
            })

        fig = create_chart_from_ai_output(chart_data)

        assert fig is not None
        assert len(fig.get_axes()[0].get_lines()) == 10

        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

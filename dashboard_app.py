"""#Interactive Dashboard"""

import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output



# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Vehicle CO2 Emissions Dashboard"

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Interactive Dashboard: Vehicle CO2 Emissions Analysis", style={"textAlign": "center"}),

    # Dropdown for selecting numerical variables
    html.Div([
        html.Label("Select a numerical variable for X-axis:"),
        dcc.Dropdown(
            id="x-axis-variable",
            options=[
                {"label": "Engine Size", "value": "Engine Size(L)"},
                {"label": "Fuel Consumption (City)", "value": "Fuel Consumption City (L/100 km)"},
                {"label": "Fuel Consumption (Highway)", "value": "Fuel Consumption Hwy (L/100 km)"},
                {"label": "Fuel Consumption (Combined)", "value": "Fuel Consumption Comb (L/100 km)"},
                {"label": "CO2 Emissions", "value": "CO2 Emissions (g/km)"}
            ],
            value="Engine Size(L)",
            clearable=False
        )
    ], style={"width": "48%", "display": "inline-block"}),

    # Dropdown for selecting categorical variables
    html.Div([
        html.Label("Select a categorical variable for grouping:"),
        dcc.Dropdown(
            id="grouping-variable",
            options=[
                {"label": "Fuel Type", "value": "Fuel Type"},
                {"label": "Vehicle Type", "value": "Vehicle Class"},
                {"label": "Brand", "value": "Make"}
            ],
            value="Fuel Type",
            clearable=False
        )
    ], style={"width": "48%", "display": "inline-block"}),

    # Slider for filtering CO2 Emissions range
    html.Div([
        html.Label("Filter by CO2 Emissions Range (g/km):"),
        dcc.RangeSlider(
            id="co2-slider",
            min=data["CO2 Emissions(g/km)"].min(),
            max=data["CO2 Emissions(g/km)"].max(),
            step=1,
            marks={int(i): str(int(i)) for i in np.linspace(data["CO2 Emissions(g/km)"].min(), data["CO2 Emissions(g/km)"].max(), 10)},
            value=[data["CO2 Emissions(g/km)"].min(), data["CO2 Emissions(g/km)"].max()]
        )
    ], style={"marginTop": 20}),

    # Graph for visualizations
    dcc.Graph(id="scatter-plot"),

    # KPI section
    html.Div([
        html.Div(id="kpi-average-co2", style={"width": "33%", "display": "inline-block", "textAlign": "center"}),
        html.Div(id="kpi-total-vehicles", style={"width": "33%", "display": "inline-block", "textAlign": "center"}),
        html.Div(id="kpi-unique-brands", style={"width": "33%", "display": "inline-block", "textAlign": "center"})
    ], style={"marginTop": 20, "padding": 20, "border": "1px solid #ddd", "borderRadius": 10})
])

# Callback to update the scatter plot and KPIs based on user input
@app.callback(
    [
        Output("scatter-plot", "figure"),
        Output("kpi-average-co2", "children"),
        Output("kpi-total-vehicles", "children"),
        Output("kpi-unique-brands", "children")
    ],
    [
        Input("x-axis-variable", "value"),
        Input("grouping-variable", "value"),
        Input("co2-slider", "value")
    ]
)
def update_dashboard(x_variable, group_variable, co2_range):
    # Filter data based on CO2 slider range
    filtered_data = data[(data["CO2 Emissions(g/km)"] >= co2_range[0]) & (data["CO2 Emissions(g/km)"] <= co2_range[1])]

    # Create scatter plot
    fig = px.scatter(
        filtered_data,
        x=x_variable,
        y="CO2 Emissions(g/km)",
        color=group_variable,
        title=f"{x_variable} vs CO2 Emissions Grouped by {group_variable}",
        labels={x_variable: x_variable, "CO2 Emissions(g/km)": "CO2 Emissions(g/km)"},
        hover_data=["Make", "Vehicle Class"]
    )

    # Calculate KPIs
    avg_co2 = filtered_data["CO2 Emissions(g/km)"].mean()
    total_vehicles = len(filtered_data)
    unique_brands = filtered_data["Make"].nunique()

    return (
        fig,
        f"Average CO2 Emissions: {avg_co2:.2f} g/km",
        f"Total Vehicles: {total_vehicles}",
        f"Unique Brands: {unique_brands}"
    )

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8060)

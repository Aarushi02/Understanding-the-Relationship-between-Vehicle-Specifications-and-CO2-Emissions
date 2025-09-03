
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('co2.csv')

"""# Section 1. Data Wrangling and Cleaning: Inspect the structure of the dataset"""

print("\nDataset Info:")
data.info()

"""## Check for missing values"""

print("\nMissing Values:\n")
print(data.isnull().sum())

"""## Check for duplicate rows"""

duplicates = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Remove duplicate rows
data = data.drop_duplicates()
print("\nDuplicate rows removed.")
data.info()

"""# Summary Statistics
## Summary for numerical features
"""

print("\nSummary Statistics for Numerical Features:\n")
print(data.describe())

"""## Count unique values for categorical variables"""

categorical_columns = ['Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
print("\nUnique Values in Categorical Features:\n")
for col in categorical_columns:
    print(f"{col}: {data[col].nunique()} unique values")

"""## Remove Outliers for CO2 Emissions using IQR"""

Q1 = data['CO2 Emissions(g/km)'].quantile(0.25)
Q3 = data['CO2 Emissions(g/km)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataset
data = data[(data['CO2 Emissions(g/km)'] >= lower_bound) & (data['CO2 Emissions(g/km)'] <= upper_bound)]

# Further refine outlier removal for CO2 Emissions using Z-Score
z_scores = (data['CO2 Emissions(g/km)'] - data['CO2 Emissions(g/km)'].mean()) / data['CO2 Emissions(g/km)'].std()
data = data[(z_scores >= -3) & (z_scores <= 3)]

"""## Feature Engineering: Create new feature - Fuel Efficiency Ratio"""

# Combines fuel consumption metrics to give a holistic measure of efficiency
data['Fuel Efficiency Ratio'] = data['Fuel Consumption Comb (L/100 km)'] / data['Engine Size(L)']
print("\nFeature 'Fuel Efficiency Ratio' added.")

data.head()

"""# Section 2. SQL Database Design and Querying

## Creating a Relational Databse
"""

import sqlite3
import pandas as pd
import numpy as np

from IPython.display import display
vehicles = data[['Make','Model','Vehicle Class']].drop_duplicates().reset_index(drop = True)
vehicles['Vehicle_ID'] = vehicles.index + 1

engine_specs = data[['Make','Model','Engine Size(L)','Cylinders','Fuel Type']].drop_duplicates().reset_index(drop = True)
engine_specs = engine_specs.merge(vehicles[['Make','Model','Vehicle_ID']], on = ['Make', 'Model'], how = 'left')
fuel_emissions = data[['Make', 'Model',
                       'Fuel Consumption City (L/100 km)',
                       'Fuel Consumption Hwy (L/100 km)',
                       'Fuel Consumption Comb (L/100 km)',
                       'Fuel Consumption Comb (mpg)',
                       'CO2 Emissions(g/km)',
                       'Fuel Efficiency Ratio']].drop_duplicates().reset_index(drop=True)
fuel_emissions = fuel_emissions.merge(vehicles[['Make', 'Model', 'Vehicle_ID']], on=['Make', 'Model'], how='left')

#Create SQLite database and coonect
conn = sqlite3.connect('vehicle_emissions_normalised.db')
cursor = conn.cursor()

cursor.execute('DROP TABLE IF EXISTS vehicles')
cursor.execute('DROP TABLE IF EXISTS engine_specs')
cursor.execute('DROP TABLE IF EXISTS fuel_emissions')
# Create tables for normalised schema
cursor.execute('''
CREATE TABLE IF NOT EXISTS vehicles(
  vehicle_id INTEGER PRIMARY KEY,
  make TEXT,
  model TEXT,
  vehicle_class TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS engine_specs(
  engine_id INTEGER PRIMARY KEY AUTOINCREMENT,
  vehicle_id INTEGER,
  engine_size REAL,
  cylinders INTEGER,
  fuel_type TEXT,
  FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id)
)
''')
cursor.execute('''
CREATE TABLE IF NOT EXISTS fuel_emissions (
    fuel_id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id INTEGER,
    fuel_consumption_city REAL,
    fuel_consumption_hwy REAL,
    fuel_consumption_comb REAL,
    fuel_consumption_comb_mpg INTEGER,
    co2_emissions INTEGER,
    fuel_efficiency INTEGER,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id)
)
''')

"""## Import data from CSV File"""

vehicles.to_sql('vehicles', conn, if_exists='replace', index=False)
engine_specs[['Vehicle_ID', 'Engine Size(L)', 'Cylinders', 'Fuel Type']].to_sql('engine_specs', conn, if_exists='replace', index=False)
fuel_emissions[['Vehicle_ID',
                'Fuel Consumption City (L/100 km)',
                'Fuel Consumption Hwy (L/100 km)',
                'Fuel Consumption Comb (L/100 km)',
                'Fuel Consumption Comb (mpg)',
                'CO2 Emissions(g/km)',
               'Fuel Efficiency Ratio']].to_sql('fuel_emissions', conn, if_exists='replace', index=False)

queries=['SELECT * FROM vehicles LIMIT 5;' , 'SELECT * FROM engine_specs LIMIT 5;','SELECT * FROM fuel_emissions LIMIT 5;']
for query in queries:
    df=pd.read_sql_query(query,conn)
    display(df)

"""# Section 3. Exploratory Data Analysis (EDA) using SQL queries"""

# Query to detect vehicles with the highest combined fuel consumption


query = '''
SELECT v.make, v.model, f."Fuel Consumption Comb (L/100 km)"
FROM vehicles v
JOIN fuel_emissions f ON v.vehicle_id = f.vehicle_id
ORDER BY f."Fuel Consumption Comb (L/100 km)" DESC
LIMIT 10;
'''

df=pd.read_sql_query(query,conn)
display(df)

# Query to detect vehicles with highest CO2 emissions
query = '''
SELECT v.make, v.model, f."CO2 Emissions(g/km)"
FROM vehicles v
JOIN fuel_emissions f ON v.vehicle_id = f.vehicle_id
ORDER BY f."CO2 Emissions(g/km)" DESC
LIMIT 10;
'''

df=pd.read_sql_query(query,conn)
display(df)

conn.commit()
conn.close()

"""## 1. Distribution of CO2 Emissions"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to the SQLite database
conn = sqlite3.connect('vehicle_emissions_normalised.db')


query = 'SELECT "CO2 Emissions(g/km)" FROM fuel_emissions'
co2_emissions = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 6))
sns.histplot(co2_emissions['CO2 Emissions(g/km)'], kde=True, bins=30, color='blue')
plt.title("Distribution of CO2 Emissions (g/km)")
plt.xlabel("CO2 Emissions (g/km)")
plt.ylabel("Frequency")
plt.show()

"""## 2. CO2 Emissions vs Engine Size"""

query = '''
SELECT e."Engine Size(L)" AS engine_size, f."CO2 Emissions(g/km)" AS co2_emissions, e."Fuel Type" AS fuel_type
FROM engine_specs e
JOIN fuel_emissions f ON e.vehicle_id = f.vehicle_id
'''
engine_emissions = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=engine_emissions, x='engine_size', y='co2_emissions', hue='fuel_type', alpha=0.7)
plt.title("CO2 Emissions vs Engine Size")
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend(title="Fuel Type")
plt.show()

"""## 3. Average CO2 Emissions by Vehicle Class"""

query = '''
SELECT v."Vehicle Class" AS vehicle_class, AVG(f."CO2 Emissions(g/km)") AS avg_emissions
FROM vehicles v
JOIN fuel_emissions f ON v.vehicle_id = f.vehicle_id
GROUP BY v."Vehicle Class"
ORDER BY avg_emissions
'''
avg_emissions_by_class = pd.read_sql_query(query, conn)

plt.figure(figsize=(12, 6))
avg_emissions_by_class.plot(kind='bar', x='vehicle_class', y='avg_emissions', legend=False, color='orange')
plt.title("Average CO2 Emissions by Vehicle Class")
plt.xlabel("Vehicle Class")
plt.ylabel("Average CO2 Emissions (g/km)")
plt.xticks(rotation=45)
plt.show()

"""## 4. Fuel Consumption vs CO2 Emissions"""

query = '''
SELECT f."Fuel Consumption Comb (L/100 km)" AS fuel_consumption, f."CO2 Emissions(g/km)" AS co2_emissions, v."Vehicle Class" AS vehicle_class
FROM fuel_emissions f
JOIN vehicles v ON f.vehicle_id = v.vehicle_id
'''
fuel_consumption = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=fuel_consumption, x='fuel_consumption', y='co2_emissions', hue='vehicle_class', alpha=0.7)
plt.title("Fuel Consumption vs CO2 Emissions")
plt.xlabel("Fuel Consumption Comb (L/100 km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend(title="Vehicle Class", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

"""## 5. Number of Vehicles by Fuel Type"""

query = '''
SELECT e."Fuel Type" AS fuel_type, COUNT(*) AS count
FROM engine_specs e
GROUP BY e."Fuel Type"
ORDER BY count DESC
'''
fuel_type_count = pd.read_sql_query(query, conn)

plt.figure(figsize=(8, 6))
sns.barplot(data=fuel_type_count, x='fuel_type', y='count',hue='fuel_type')
plt.title("Number of Vehicles by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Count")
plt.show()

"""## 6. Fuel Consumption (mpg) vs CO2 Emissions"""

query = '''
SELECT
   f."Fuel Consumption Comb (mpg)" AS fuel_consumption,
   f."CO2 Emissions(g/km)" AS co2_emissions
FROM fuel_emissions f;
'''
fuel_efficiency = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=fuel_efficiency, x='fuel_consumption', y='co2_emissions')
plt.title("Fuel Consumption vs CO2 Emissions")
plt.xlabel("Fuel Consumption Comb (mpg)")
plt.ylabel("CO2 Emissions (g/km)")
plt.show()

"""## 7. Correlation Heatmap"""

query = """
SELECT
    e."Engine Size(L)" AS engine_size,
    e.cylinders AS cylinders,
    f."Fuel Consumption City (L/100 km)" AS fuel_consumption_city,
    f."Fuel Consumption Hwy (L/100 km)" AS fuel_consumption_hwy,
    f."Fuel Consumption Comb (L/100 km)" AS fuel_consumption_comb,
    f."Fuel Consumption Comb (mpg)" AS fuel_consumption_comb_mpg,
    f."CO2 Emissions(g/km)" AS co2_emissions
FROM engine_specs e
JOIN fuel_emissions f ON e.vehicle_id = f.vehicle_id;
"""

dats = pd.read_sql_query(query, conn)

correlation_matrix = dats.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()

"""# Section 4. Advanced Data Analysis with Machine Learning

## Predict CO2 Emissions Based on Vehicle Features using Random Forest
"""

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Connect to SQLite database
conn = sqlite3.connect('vehicle_emissions_normalised.db')

# SQL query to fetch all relevant features
query = """
SELECT
    e."Engine Size(L)" AS engine_size,
    e.cylinders AS cylinders,
    e."Fuel Type" AS fuel_type,
    f."Fuel Consumption Comb (L/100 km)" AS fuel_consumption_comb,
    f."Fuel Consumption Comb (mpg)" AS fuel_efficiency_mpg,
    f."CO2 Emissions(g/km)" AS co2_emissions
FROM engine_specs e
JOIN fuel_emissions f ON e.vehicle_id = f.vehicle_id;
"""

# Load the data into a Pandas DataFrame
dats = pd.read_sql_query(query, conn)

# Drop rows with missing values
dats = dats.dropna()

# Separate features and target variable
X = dats[['engine_size', 'cylinders', 'fuel_type', 'fuel_consumption_comb', 'fuel_efficiency_mpg']]
y = dats['co2_emissions']

# One-hot encode the categorical 'fuel_type' feature
X_encoded = pd.get_dummies(X, columns=['fuel_type'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
rf = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Compare actual vs predicted CO2 emissions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted CO2 Emissions', fontsize=16)
plt.xlabel('Actual CO2 Emissions (g/km)', fontsize=12)
plt.ylabel('Predicted CO2 Emissions (g/km)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
# Close the database connection
conn.close()

"""## Feature Importance using Gradient Boosting"""

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Connect to SQLite database
conn = sqlite3.connect('vehicle_emissions_normalised.db')

# SQL query to fetch features and target variable
query = """
SELECT
    e."Engine Size(L)" AS engine_size,
    e.cylinders AS cylinders,
    f."Fuel Consumption Comb (L/100 km)" AS fuel_consumption_comb,
    f."Fuel Consumption Comb (mpg)" AS fuel_efficiency_mpg,
    f."CO2 Emissions(g/km)" AS co2_emissions
FROM engine_specs e
JOIN fuel_emissions f ON e.vehicle_id = f.vehicle_id;
"""

# Load the data into a Pandas DataFrame
dats = pd.read_sql_query(query, conn)

# Drop rows with missing values
dats = dats.dropna()

# Features and target variable
X = dats[['engine_size', 'cylinders', 'fuel_consumption_comb', 'fuel_efficiency_mpg']]
y = dats['co2_emissions']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3)
gbr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gbr.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature importance visualization
feature_importance = gbr.feature_importances_
features = X.columns

plt.figure(figsize=(8, 6))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Gradient Boosting Feature Importance')
plt.tight_layout()
plt.show()

# Close the database connection
conn.close()

"""# Section 5. Interactive Dashboard"""

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


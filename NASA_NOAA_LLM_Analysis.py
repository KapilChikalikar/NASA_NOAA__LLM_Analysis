# NASA & NOAA Climate Analysis using LLM and Data Science Concepts
# Public datasets used:
# - NASA GISTEMP Global Temperature Anomalies
# - NOAA Mauna Loa Atmospheric CO₂ Levels

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Dataset URLs
temp_url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
co2_url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"

# Load and clean NASA temperature anomaly data
temp_df = pd.read_csv(temp_url, skiprows=1)
temp_df = temp_df[['Year', 'J-D']].rename(columns={'J-D': 'TempAnomaly'})
temp_df = temp_df[temp_df['TempAnomaly'] != '***']
temp_df['TempAnomaly'] = temp_df['TempAnomaly'].astype(float)

# Load and clean NOAA CO₂ data
co2_df = pd.read_csv(co2_url, comment='#')
co2_df = co2_df[['year', 'month', 'average']]
co2_df = co2_df[co2_df['average'] != -99.99]  # filter invalid
co2_annual = co2_df.groupby('year')['average'].mean().reset_index().rename(columns={'average': 'CO2'})

# Merge datasets by year
merged_df = pd.merge(temp_df, co2_annual, left_on='Year', right_on='year')
merged_df = merged_df[['Year', 'TempAnomaly', 'CO2']]

# Linear regression model
X = merged_df[['CO2']]
y = merged_df['TempAnomaly']
model = LinearRegression()
model.fit(X, y)
merged_df['PredictedTemp'] = model.predict(X)

# Plot with labels
plt.figure(figsize=(12, 7))
sns.set(style="whitegrid")
plt.scatter(merged_df['CO2'], merged_df['TempAnomaly'], label="Observed Temp", color="blue")
plt.plot(merged_df['CO2'], merged_df['PredictedTemp'], label="Linear Fit", color="red")

# Add year labels (every 3rd year for clarity)
for i in range(0, len(merged_df), 3):
    plt.text(merged_df['CO2'].iloc[i], merged_df['TempAnomaly'].iloc[i] + 0.01,
             str(merged_df['Year'].iloc[i]), fontsize=8, ha='center', rotation=45)

plt.title("CO₂ vs Global Temperature Anomaly\n(NASA & NOAA Public Data)", fontsize=14)
plt.xlabel("Atmospheric CO₂ (ppm)", fontsize=12)
plt.ylabel("Temperature Anomaly (°C)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("nasa_noaa_labeled_plot.png")
plt.show()

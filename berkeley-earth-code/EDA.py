
# Berkeley Earth Temperature Dataset Exploratory Data Analysis
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.cluster import KMeans
from scipy import stats
import os
from datetime import datetime
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)

# Data paths - update these to your file locations
data_dir = r"C:\Users\Richard Nuckols\Desktop\Personal\JHU\Neural Networks\Module7\Data"
global_temp_file = os.path.join(data_dir, "GlobalTemperatures.csv")
countries_file = os.path.join(data_dir, "GlobalLandTemperaturesByCountry.csv")
states_file = os.path.join(data_dir, "GlobalLandTemperaturesByState.csv")
major_cities_file = os.path.join(data_dir, "GlobalLandTemperaturesByMajorCity.csv")
cities_file = os.path.join(data_dir, "GlobalLandTemperaturesByCity.csv")

# Create output directory for figures
output_dir = os.path.join(data_dir, "EDA_Results")
os.makedirs(output_dir, exist_ok=True)

# Function to save figures
def save_fig(fig, filename):
    fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close(fig)

# =========================================================
# Part 1: Data Loading and Initial Inspection
# =========================================================

def load_and_inspect_data():
    """Load all datasets and perform initial inspection"""
    print("Loading datasets...")
    
    # Dictionary to store all dataframes
    dfs = {}
    
    # Load Global Temperatures
    dfs['global'] = pd.read_csv(global_temp_file)
    
    # Load Countries
    dfs['countries'] = pd.read_csv(countries_file)
    
    # Load States 
    dfs['states'] = pd.read_csv(states_file)
    
    # Load Major Cities
    dfs['major_cities'] = pd.read_csv(major_cities_file)
    
    # For the cities dataset, which is very large, we'll load a sample first
    # You can adjust this or load the full dataset if your system can handle it
    dfs['cities'] = pd.read_csv(cities_file, nrows=100000)  # Sample 100K rows
    
    # Process dates for all dataframes
    for key in dfs:
        dfs[key]['dt'] = pd.to_datetime(dfs[key]['dt'])
    
    # Display basic info for each dataset
    for name, df in dfs.items():
        print(f"\n=== {name.upper()} Dataset ===")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['dt'].min()} to {df['dt'].max()}")
        print(f"Missing values:\n{df.isnull().sum()}")
        
        # For temperature columns, show basic statistics
        temp_cols = [col for col in df.columns if 'Temperature' in col and 'Uncertainty' not in col]
        if temp_cols:
            print("\nTemperature Statistics:")
            print(df[temp_cols].describe())
    
    return dfs

# =========================================================
# Part 2: Temporal Analysis
# =========================================================

def temporal_analysis(dfs):
    """Analyze temperature trends over time"""
    print("\nPerforming temporal analysis...")
    
    # 2.1 Global temperature trends over time
    global_df = dfs['global']
    
    # Resample to annual averages
    annual_global = global_df.set_index('dt').resample('Y')['LandAverageTemperature'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(annual_global['dt'], annual_global['LandAverageTemperature'], 'o-', linewidth=2)
    ax.set_title('Global Land Average Temperature (Annual)', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Temperature (°C)', fontsize=14)
    
    # Add trend line
    z = np.polyfit(range(len(annual_global)), annual_global['LandAverageTemperature'], 1)
    p = np.poly1d(z)
    ax.plot(annual_global['dt'], p(range(len(annual_global))), "r--", linewidth=2)
    
    # Add the equation of the trend line to the plot
    trend_eq = f'Trend: {z[0]:.4f}°C/year'
    ax.text(0.05, 0.95, trend_eq, transform=ax.transAxes, fontsize=14, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    save_fig(fig, "global_annual_trend.png")
    
    # 2.2 Seasonal decomposition
    # Select a recent period with complete data
    recent_period = global_df[(global_df['dt'] >= '1950-01-01') & (global_df['dt'] <= '2010-12-31')]
    recent_period = recent_period.set_index('dt')
    
    try:
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(recent_period['LandAverageTemperature'], model='additive', period=12)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16))
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('Observed', fontsize=14)
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('Trend', fontsize=14)
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal', fontsize=14)
        decomposition.resid.plot(ax=ax4)
        ax4.set_title('Residual', fontsize=14)
        plt.tight_layout()
        save_fig(fig, "seasonal_decomposition.png")
    except Exception as e:
        print(f"Seasonal decomposition failed: {e}")
    
    # 2.3 Decade-over-decade temperature changes (global)
    global_df['decade'] = (global_df['dt'].dt.year // 10) * 10
    decade_avg = global_df.groupby('decade')['LandAverageTemperature'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='decade', y='LandAverageTemperature', data=decade_avg, ax=ax)
    ax.set_title('Global Land Temperature by Decade', fontsize=16)
    ax.set_xlabel('Decade', fontsize=14)
    ax.set_ylabel('Average Temperature (°C)', fontsize=14)
    plt.xticks(rotation=45)
    save_fig(fig, "global_temp_by_decade.png")
    
    # 2.4 Country-level temperature trends for select major countries
    countries_df = dfs['countries']
    
    # Select a few major countries from different continents
    major_countries = ['United States', 'China', 'Russia', 'Brazil', 'Australia', 'Germany', 'South Africa', 'India']
    
    # Filter and create annual averages
    major_country_data = countries_df[countries_df['Country'].isin(major_countries)]
    major_country_data['Year'] = major_country_data['dt'].dt.year
    annual_country_temps = major_country_data.groupby(['Country', 'Year'])['AverageTemperature'].mean().reset_index()
    
    # Plot temperature trends for selected countries
    fig, ax = plt.subplots(figsize=(14, 10))
    for country in major_countries:
        country_data = annual_country_temps[annual_country_temps['Country'] == country]
        if not country_data.empty:
            ax.plot(country_data['Year'], country_data['AverageTemperature'], 'o-', label=country, linewidth=2, markersize=1)
    
    ax.set_title('Temperature Trends by Country (Annual Average)', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Temperature (°C)', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    save_fig(fig, "country_temperature_trends.png")
    
    return annual_global, annual_country_temps

# =========================================================
# Part 3: Spatial Analysis
# =========================================================

def spatial_analysis(dfs):
    """Analyze spatial patterns in temperature data"""
    print("\nPerforming spatial analysis...")
    
    # 3.1 Create a global temperature map using the most recent complete year
    cities_df = dfs['major_cities']
    
    # Get the most recent complete year
    max_year = cities_df['dt'].dt.year.max()
    recent_cities = cities_df[cities_df['dt'].dt.year == max_year]
    
    # Aggregate to get average for the year by city
    recent_cities = recent_cities.groupby(['City', 'Country', 'Latitude', 'Longitude'])['AverageTemperature'].mean().reset_index()
    
    # Convert latitude and longitude strings to numeric values
    recent_cities['Latitude_num'] = recent_cities['Latitude'].str.replace('[NS]', '', regex=True).astype(float)
    recent_cities['Latitude_num'] = recent_cities['Latitude_num'] * recent_cities['Latitude'].str.contains('S').map({True: -1, False: 1})
    
    recent_cities['Longitude_num'] = recent_cities['Longitude'].str.replace('[EW]', '', regex=True).astype(float)
    recent_cities['Longitude_num'] = recent_cities['Longitude_num'] * recent_cities['Longitude'].str.contains('W').map({True: -1, False: 1})
    
    # Create a scatter plot on world map
    try:
        # This requires the geopandas package
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        
        fig, ax = plt.subplots(figsize=(15, 10))
        world.plot(ax=ax, color='lightgray', edgecolor='black')
        
        scatter = ax.scatter(recent_cities['Longitude_num'], recent_cities['Latitude_num'], 
                        c=recent_cities['AverageTemperature'], cmap='coolwarm',
                        s=30, alpha=0.7)
        
        plt.colorbar(scatter, label='Average Temperature (°C)')
        ax.set_title(f'Global Temperature Distribution ({max_year})', fontsize=16)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        plt.tight_layout()
        save_fig(fig, "global_temperature_map.png")
    except Exception as e:
        print(f"Error creating world map: {e}")
        print("Creating alternative visualization without map background...")
        
        # Alternative visualization without geopandas
        fig, ax = plt.subplots(figsize=(15, 10))
        scatter = ax.scatter(recent_cities['Longitude_num'], recent_cities['Latitude_num'], 
                        c=recent_cities['AverageTemperature'], cmap='coolwarm',
                        s=30, alpha=0.7)
        
        plt.colorbar(scatter, label='Average Temperature (°C)')
        ax.set_title(f'Global Temperature Distribution ({max_year})', fontsize=16)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.grid(True)
        plt.tight_layout()
        save_fig(fig, "global_temperature_scatter.png")
    
    # 3.2 Temperature by latitude bands
    cities_df = dfs['cities']  # Use the larger cities dataset
    
    # Extract numeric latitude values
    cities_df['Latitude_num'] = cities_df['Latitude'].str.replace('[NS]', '', regex=True).astype(float)
    cities_df['Latitude_num'] = cities_df['Latitude_num'] * cities_df['Latitude'].str.contains('S').map({True: -1, False: 1})
    
    # Create latitude bands (10-degree increments)
    cities_df['Latitude_band'] = pd.cut(cities_df['Latitude_num'], 
                                       bins=range(-90, 91, 10),
                                       labels=[f'{i}° to {i+10}°' for i in range(-90, 90, 10)])
    
    # Calculate average temperature by latitude band for the most recent year
    recent_year = cities_df['dt'].dt.year.max()
    lat_band_temps = cities_df[cities_df['dt'].dt.year == recent_year].groupby('Latitude_band')['AverageTemperature'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Latitude_band', y='AverageTemperature', data=lat_band_temps, ax=ax)
    ax.set_title(f'Average Temperature by Latitude Band ({recent_year})', fontsize=16)
    ax.set_xlabel('Latitude', fontsize=14)
    ax.set_ylabel('Average Temperature (°C)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_fig(fig, "temperature_by_latitude.png")
    
    # 3.3 Urban heat island effect (compare major cities with country averages)
    # Join city and country datasets
    major_cities_df = dfs['major_cities']
    countries_df = dfs['countries']
    
    # Add year column to both dataframes
    major_cities_df['Year'] = major_cities_df['dt'].dt.year
    countries_df['Year'] = countries_df['dt'].dt.year
    
    # Calculate annual averages
    city_annual = major_cities_df.groupby(['City', 'Country', 'Year'])['AverageTemperature'].mean().reset_index()
    country_annual = countries_df.groupby(['Country', 'Year'])['AverageTemperature'].mean().reset_index()
    
    # Merge city and country data
    city_country_comp = pd.merge(city_annual, country_annual, 
                                on=['Country', 'Year'], 
                                suffixes=('_city', '_country'))
    
    # Calculate temperature difference (city minus country)
    city_country_comp['Temp_Difference'] = city_country_comp['AverageTemperature_city'] - city_country_comp['AverageTemperature_country']
    
    # Calculate average temperature difference by city
    city_temp_diff = city_country_comp.groupby('City')['Temp_Difference'].mean().reset_index()
    city_temp_diff = city_temp_diff.sort_values('Temp_Difference', ascending=False)
    
    # Plot top 20 cities with highest temperature difference (urban heat island effect)
    top_cities = city_temp_diff.head(20)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(x='Temp_Difference', y='City', data=top_cities, ax=ax)
    ax.set_title('Top 20 Cities with Strongest Urban Heat Island Effect', fontsize=16)
    ax.set_xlabel('Average Temperature Difference (City - Country) in °C', fontsize=14)
    ax.set_ylabel('City', fontsize=14)
    plt.tight_layout()
    save_fig(fig, "urban_heat_island_effect.png")
    
    return recent_cities, city_country_comp

# =========================================================
# Part 4: Data Quality and Uncertainty Analysis
# =========================================================

def uncertainty_analysis(dfs):
    """Analyze uncertainty in temperature measurements"""
    print("\nPerforming uncertainty analysis...")
    
    # 4.1 Global uncertainty trends over time
    global_df = dfs['global']
    
    # Annual average uncertainty
    global_df['Year'] = global_df['dt'].dt.year
    uncertainty_by_year = global_df.groupby('Year')['LandAverageTemperatureUncertainty'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(uncertainty_by_year['Year'], uncertainty_by_year['LandAverageTemperatureUncertainty'], 'o-', linewidth=2)
    ax.set_title('Global Temperature Measurement Uncertainty Over Time', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Average Uncertainty (°C)', fontsize=14)
    plt.grid(True)
    save_fig(fig, "global_uncertainty_trend.png")
    
    # 4.2 Geographic distribution of uncertainty
    countries_df = dfs['countries']
    countries_df['Year'] = countries_df['dt'].dt.year
    
    # Calculate average uncertainty by country for a recent period
    recent_period = (countries_df['Year'] >= 1950) & (countries_df['Year'] <= 2010)
    uncertainty_by_country = countries_df[recent_period].groupby('Country')['AverageTemperatureUncertainty'].mean().reset_index()
    uncertainty_by_country = uncertainty_by_country.sort_values('AverageTemperatureUncertainty', ascending=False)
    
    # Plot top 20 countries with highest uncertainty
    top_uncertain_countries = uncertainty_by_country.head(20)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(x='AverageTemperatureUncertainty', y='Country', data=top_uncertain_countries, ax=ax)
    ax.set_title('Countries with Highest Temperature Measurement Uncertainty (1950-2010)', fontsize=16)
    ax.set_xlabel('Average Uncertainty (°C)', fontsize=14)
    ax.set_ylabel('Country', fontsize=14)
    plt.tight_layout()
    save_fig(fig, "uncertainty_by_country.png")
    
    # 4.3 Relationship between temperature and uncertainty
    global_scatter = global_df[recent_period]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(global_scatter['LandAverageTemperature'], 
               global_scatter['LandAverageTemperatureUncertainty'],
               alpha=0.5)
    ax.set_title('Relationship Between Temperature and Uncertainty', fontsize=16)
    ax.set_xlabel('Average Temperature (°C)', fontsize=14)
    ax.set_ylabel('Temperature Uncertainty (°C)', fontsize=14)
    
    # Add trend line
    z = np.polyfit(global_scatter['LandAverageTemperature'], 
                   global_scatter['LandAverageTemperatureUncertainty'], 1)
    p = np.poly1d(z)
    ax.plot(global_scatter['LandAverageTemperature'], 
            p(global_scatter['LandAverageTemperature']), "r--", linewidth=2)
    
    # Calculate and display correlation
    corr = global_scatter['LandAverageTemperature'].corr(global_scatter['LandAverageTemperatureUncertainty'])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, fontsize=14, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.grid(True)
    save_fig(fig, "temperature_uncertainty_correlation.png")
    
    return uncertainty_by_year, uncertainty_by_country

# =========================================================
# Part 5: Extreme Event Analysis
# =========================================================

def extreme_event_analysis(dfs):
    """Analyze extreme temperature events"""
    print("\nPerforming extreme event analysis...")
    
    # 5.1 Identify extreme temperature events in global record
    global_df = dfs['global']
    
    # Calculate rolling averages to smooth data
    global_df = global_df.sort_values('dt')
    global_df['RollingAvgTemp'] = global_df['LandAverageTemperature'].rolling(window=12).mean()
    
    # Calculate statistical thresholds for extremes (e.g., 2 standard deviations)
    global_mean = global_df['LandAverageTemperature'].mean()
    global_std = global_df['LandAverageTemperature'].std()
    
    extreme_high_threshold = global_mean + 2 * global_std
    extreme_low_threshold = global_mean - 2 * global_std
    
    # Identify extreme months
    global_df['ExtremeHigh'] = global_df['LandAverageTemperature'] > extreme_high_threshold
    global_df['ExtremeLow'] = global_df['LandAverageTemperature'] < extreme_low_threshold
    
    # Plot global temperature with extreme events highlighted
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot the full temperature series
    ax.plot(global_df['dt'], global_df['LandAverageTemperature'], 'b-', alpha=0.5, label='Monthly Temp')
    ax.plot(global_df['dt'], global_df['RollingAvgTemp'], 'k-', label='12-month Moving Avg')
    
    # Highlight extreme events
    extreme_high = global_df[global_df['ExtremeHigh']]
    extreme_low = global_df[global_df['ExtremeLow']]
    
    ax.scatter(extreme_high['dt'], extreme_high['LandAverageTemperature'], 
               color='red', s=30, label='Extreme High')
    ax.scatter(extreme_low['dt'], extreme_low['LandAverageTemperature'], 
               color='blue', s=30, label='Extreme Low')
    
    # Add threshold lines
    ax.axhline(y=extreme_high_threshold, color='r', linestyle='--', alpha=0.7, 
               label=f'High Threshold: {extreme_high_threshold:.2f}°C')
    ax.axhline(y=extreme_low_threshold, color='b', linestyle='--', alpha=0.7,
               label=f'Low Threshold: {extreme_low_threshold:.2f}°C')
    
    ax.set_title('Global Land Temperature with Extreme Events Highlighted', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Temperature (°C)', fontsize=14)
    ax.legend(loc='upper left')
    plt.grid(True)
    save_fig(fig, "global_extreme_events.png")
    
    # 5.2 Analyze frequency of extreme events over time
    global_df['Year'] = global_df['dt'].dt.year
    global_df['Decade'] = (global_df['Year'] // 10) * 10
    
    # Count extreme events by decade
    extreme_counts = global_df.groupby('Decade')[['ExtremeHigh', 'ExtremeLow']].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(extreme_counts['Decade'], extreme_counts['ExtremeHigh'], color='red', alpha=0.7, label='Extreme High')
    ax.bar(extreme_counts['Decade'], extreme_counts['ExtremeLow'], color='blue', alpha=0.7, label='Extreme Low')
    
    ax.set_title('Frequency of Extreme Temperature Events by Decade', fontsize=16)
    ax.set_xlabel('Decade', fontsize=14)
    ax.set_ylabel('Number of Extreme Months', fontsize=14)
    ax.legend()
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    save_fig(fig, "extreme_event_frequency.png")
    
    # 5.3 Analyze regional extreme events
    countries_df = dfs['countries']
    
    # Select a few major countries for analysis
    major_countries = ['United States', 'China', 'Russia', 'Brazil', 'Australia', 'India']
    selected_countries = countries_df[countries_df['Country'].isin(major_countries)]
    
    # Calculate country-specific thresholds
    country_stats = selected_countries.groupby('Country')['AverageTemperature'].agg(['mean', 'std']).reset_index()
    country_stats['high_threshold'] = country_stats['mean'] + 2 * country_stats['std']
    country_stats['low_threshold'] = country_stats['mean'] - 2 * country_stats['std']
    
    # Create a dictionary of extreme events by country
    country_extremes = {}
    
    for country in major_countries:
        country_data = selected_countries[selected_countries['Country'] == country]
        country_mean = country_stats[country_stats['Country'] == country]['mean'].values[0]
        country_high = country_stats[country_stats['Country'] == country]['high_threshold'].values[0]
        country_low = country_stats[country_stats['Country'] == country]['low_threshold'].values[0]
        
        country_data['ExtremeHigh'] = country_data['AverageTemperature'] > country_high
        country_data['ExtremeLow'] = country_data['AverageTemperature'] < country_low
        country_data['Year'] = country_data['dt'].dt.year
        
        country_extremes[country] = country_data
    
    # Plot extreme events for one example country (you can modify to create plots for all)
    example_country = 'United States'
    country_data = country_extremes[example_country]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot the full temperature series
    ax.plot(country_data['dt'], country_data['AverageTemperature'], 'b-', alpha=0.5, label='Monthly Temp')
    
    # Calculate and plot rolling average
    country_data = country_data.sort_values('dt')
    country_data['RollingAvgTemp'] = country_data['AverageTemperature'].rolling(window=12).mean()
    ax.plot(country_data['dt'], country_data['RollingAvgTemp'], 'k-', label='12-month Moving Avg')
    
    # Highlight extreme events
    extreme_high = country_data[country_data['ExtremeHigh']]
    extreme_low = country_data[country_data['ExtremeLow']]
    
    ax.scatter(extreme_high['dt'], extreme_high['AverageTemperature'], 
               color='red', s=30, label='Extreme High')
    ax.scatter(extreme_low['dt'], extreme_low['AverageTemperature'], 
               color='blue', s=30, label='Extreme Low')
    
    # Add threshold lines
    country_high = country_stats[country_stats['Country'] == example_country]['high_threshold'].values[0]
    country_low = country_stats[country_stats['Country'] == example_country]['low_threshold'].values[0]
    
    ax.axhline(y=country_high, color='r', linestyle='--', alpha=0.7, 
               label=f'High Threshold: {country_high:.2f}°C')
    ax.axhline(y=country_low, color='b', linestyle='--', alpha=0.7,
               label=f'Low Threshold: {country_low:.2f}°C')
    
    ax.set_title(f'Temperature and Extreme Events in {example_country}', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Temperature (°C)', fontsize=14)
    ax.legend(loc='upper left')
    plt.grid(True)
    save_fig(fig, f"extreme_events_{example_country.replace(' ', '_')}.png")
    
    return global_df, country_extremes

# =========================================================
# Part 6: Pattern Analysis and Clustering
# =========================================================

def pattern_analysis(dfs):
    """Analyze temperature patterns and cluster regions with similar behavior"""
    print("\nPerforming pattern analysis...")
    
    # 6.1 Temperature seasonality patterns by latitude
    cities_df = dfs['major_cities']
    
    # Extract numeric latitude values
    cities_df['Latitude_num'] = cities_df['Latitude'].str.replace('[NS]', '', regex=True).astype(float)
    cities_df['Latitude_num'] = cities_df['Latitude_num'] * cities_df['Latitude'].str.contains('S').map({True: -1, False: 1})
    
    # Create latitude bands (30-degree increments for simplicity)
    cities_df['Latitude_band'] = pd.cut(cities_df['Latitude_num'], 
                                      bins=[-90, -60, -30, 0, 30, 60, 90],
                                      labels=['South Polar', 'South Temperate', 'South Tropical', 
                                              'North Tropical', 'North Temperate', 'North Polar'])
    
    # Filter to a recent period (e.g., last 10 years)
    recent_years = cities_df['dt'].dt.year.max() - 10
    recent_data = cities_df[cities_df['dt'].dt.year >= recent_years]
    
    # Add month column
    recent_data['Month'] = recent_data['dt'].dt.month
    
    # Calculate average temperature by latitude band and month
    seasonal_patterns = recent_data.groupby(['Latitude_band', 'Month'])['AverageTemperature'].mean().reset_index()
    
    # Plot seasonal patterns by latitude band
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for band in seasonal_patterns['Latitude_band'].unique():
        band_data = seasonal_patterns[seasonal_patterns['Latitude_band'] == band]
        ax.plot(band_data['Month'], band_data['AverageTemperature'], 'o-', linewidth=2, label=band)
    
    ax.set_title('Seasonal Temperature Patterns by Latitude Band', fontsize=16)
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Average Temperature (°C)', fontsize=14)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    save_fig(fig, "seasonal_patterns_by_latitude.png")
    
    # 6.2 Cluster countries based on temperature patterns
    countries_df = dfs['countries']
    
    # Create year and month columns
    countries_df['Year'] = countries_df['dt'].dt.year
    countries_df['Month'] = countries_df['dt'].dt.month
    
    # Focus on recent 30 years of data for clustering
    recent_cutoff = countries_df['Year'].max() - 30
    clustering_data = countries_df[countries_df['Year'] >= recent_cutoff]
    
    # Create a pivot table of monthly temperature by country
    temp_pivot = clustering_data.pivot_table(
        index='Country',
        columns='Month',
        values='AverageTemperature',
        aggfunc='mean'
    ).reset_index()
    
    # Drop countries with missing values
    temp_pivot_clean = temp_pivot.dropna()
    
    # Prepare data for clustering (remove Country column)
    cluster_data = temp_pivot_clean.iloc[:, 1:].values
    countries = temp_pivot_clean['Country'].values
    
    # Normalize the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)
    
    # Determine optimal number of clusters using silhouette score
    from sklearn.metrics import silhouette_score
    
    silhouette_scores = []
    k_range = range(2, 11)  # Try 2 to 10 clusters
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cluster_data_scaled)
        silhouette_avg = silhouette_score(cluster_data_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.3f}")
    
    # Plot silhouette scores
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, silhouette_scores, 'bo-')
    ax.set_xlabel('Number of Clusters', fontsize=14)
    ax.set_ylabel('Silhouette Score', fontsize=14)
    ax.set_title('Silhouette Score for Different Numbers of Clusters', fontsize=16)
    ax.grid(True)
    save_fig(fig, "cluster_silhouette_scores.png")
    
    # Choose the best number of clusters based on highest silhouette score
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"Best number of clusters: {best_k}")
    
    # Perform clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_data_scaled)
    
    # Add cluster labels to the original data
    temp_pivot_clean['Cluster'] = cluster_labels
    
    # Calculate cluster centers and convert back to original scale
    cluster_centers = kmeans.cluster_centers_
    cluster_centers_original = scaler.inverse_transform(cluster_centers)
    
    # Plot the seasonal patterns of each cluster
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i in range(best_k):
        ax.plot(range(1, 13), cluster_centers_original[i], 'o-', linewidth=2, 
                label=f'Cluster {i+1} (n={sum(cluster_labels == i)})')
    
    ax.set_title('Temperature Patterns by Country Cluster', fontsize=16)
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Average Temperature (°C)', fontsize=14)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    save_fig(fig, "country_clusters_patterns.png")
    
    # Create a dataframe with countries and their assigned clusters
    country_clusters = temp_pivot_clean[['Country', 'Cluster']]
    
    # Print sample of countries in each cluster
    for i in range(best_k):
        cluster_countries = country_clusters[country_clusters['Cluster'] == i]['Country'].values
        sample_size = min(5, len(cluster_countries))  # Show at most 5 countries per cluster
        print(f"\nCluster {i+1} ({len(cluster_countries)} countries), Sample countries: {', '.join(cluster_countries[:sample_size])}")
    
    return seasonal_patterns, country_clusters

# =========================================================
# Part 7: Regional Comparison Analysis
# =========================================================

def regional_comparison(dfs):
    """Compare temperature patterns across different regions"""
    print("\nPerforming regional comparison analysis...")
    
    # 7.1 Northern vs Southern Hemisphere comparison
    cities_df = dfs['major_cities']
    
    # Extract hemisphere information
    cities_df['Hemisphere'] = cities_df['Latitude'].str.contains('N').map({True: 'Northern', False: 'Southern'})
    
    # Add year and month
    cities_df['Year'] = cities_df['dt'].dt.year
    cities_df['Month'] = cities_df['dt'].dt.month
    
    # Calculate average temperature by hemisphere, year, and month
    hemisphere_data = cities_df.groupby(['Hemisphere', 'Year', 'Month'])['AverageTemperature'].mean().reset_index()
    
    # Create a plot showing temperature trends for both hemispheres
    recent_years = hemisphere_data['Year'] >= 1950
    recent_hemisphere = hemisphere_data[recent_years]
    
    # Calculate average monthly temperature across all years by hemisphere
    avg_monthly = recent_hemisphere.groupby(['Hemisphere', 'Month'])['AverageTemperature'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for hemisphere in ['Northern', 'Southern']:
        hemi_data = avg_monthly[avg_monthly['Hemisphere'] == hemisphere]
        ax.plot(hemi_data['Month'], hemi_data['AverageTemperature'], 'o-', linewidth=2, label=hemisphere)
    
    ax.set_title('Average Monthly Temperature: Northern vs Southern Hemisphere (1950-Present)', fontsize=16)
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Average Temperature (°C)', fontsize=14)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend()
    plt.grid(True)
    save_fig(fig, "hemisphere_comparison.png")
    
    # 7.2 Continental temperature trends
    countries_df = dfs['countries']
    
    # Create a mapping of countries to continents (simplified)
    # This is a simplified mapping and will not cover all countries
    continent_mapping = {
        'North America': ['United States', 'Canada', 'Mexico', 'Cuba', 'Jamaica', 'Haiti', 'Dominican Republic'],
        'South America': ['Brazil', 'Argentina', 'Chile', 'Peru', 'Colombia', 'Venezuela', 'Ecuador', 'Bolivia'],
        'Europe': ['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland', 'Romania', 'Netherlands', 
                  'Belgium', 'Sweden', 'Austria', 'Switzerland', 'Portugal', 'Denmark', 'Finland', 'Norway'],
        'Asia': ['China', 'India', 'Japan', 'South Korea', 'Indonesia', 'Philippines', 'Vietnam', 'Thailand', 
                'Malaysia', 'Myanmar', 'Bangladesh', 'Pakistan', 'Turkey', 'Iran', 'Iraq', 'Saudi Arabia'],
        'Africa': ['Nigeria', 'Ethiopia', 'Egypt', 'South Africa', 'Algeria', 'Morocco', 'Kenya', 'Tanzania', 
                  'Ghana', 'Cameroon', 'Ivory Coast', 'Madagascar', 'Angola', 'Mozambique', 'Uganda'],
        'Oceania': ['Australia', 'New Zealand', 'Papua New Guinea', 'Fiji', 'Solomon Islands']
    }
    
    # Flatten the mapping to create a dictionary of country -> continent
    country_to_continent = {}
    for continent, countries_list in continent_mapping.items():
        for country in countries_list:
            country_to_continent[country] = continent
    
    # Add continent column to the dataframe
    countries_df['Continent'] = countries_df['Country'].map(country_to_continent)
    
    # Add year column
    countries_df['Year'] = countries_df['dt'].dt.year
    
    # Calculate average temperature by continent and year
    # Filter to only include rows with a mapped continent
    continent_temp = countries_df.dropna(subset=['Continent'])
    
    # Calculate average by continent and year
    continent_yearly = continent_temp.groupby(['Continent', 'Year'])['AverageTemperature'].mean().reset_index()
    
    # Plot temperature trends by continent
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for continent in continent_mapping.keys():
        cont_data = continent_yearly[continent_yearly['Continent'] == continent]
        if not cont_data.empty:
            ax.plot(cont_data['Year'], cont_data['AverageTemperature'], 'o-', linewidth=2, 
                    markersize=2, label=continent)
    
    ax.set_title('Temperature Trends by Continent', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Average Temperature (°C)', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    save_fig(fig, "continental_temperature_trends.png")
    
    # 7.3 Temperature volatility by region
    # Calculate temperature volatility (standard deviation) by continent and decade
    countries_df['Decade'] = (countries_df['Year'] // 10) * 10
    
    volatility = countries_df.dropna(subset=['Continent']).groupby(['Continent', 'Decade'])['AverageTemperature'].std().reset_index()
    volatility = volatility.rename(columns={'AverageTemperature': 'Temperature_Volatility'})
    
    # Create a pivot table for easier visualization
    volatility_pivot = volatility.pivot(index='Decade', columns='Continent', values='Temperature_Volatility')
    
    # Plot volatility trends
    fig, ax = plt.subplots(figsize=(14, 8))
    volatility_pivot.plot(ax=ax, marker='o')
    
    ax.set_title('Temperature Volatility by Continent and Decade', fontsize=16)
    ax.set_xlabel('Decade', fontsize=14)
    ax.set_ylabel('Temperature Volatility (°C)', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    save_fig(fig, "temperature_volatility_by_region.png")
    
    return hemisphere_data, continent_yearly, volatility

# =========================================================
# Part 8: Main Function to Run All Analyses
# =========================================================

def main():
    """Main function to run all analyses"""
    print("Starting Berkeley Earth Temperature Dataset EDA...")
    
    # Load data
    dfs = load_and_inspect_data()
    
    # Perform analyses
    annual_global, annual_country_temps = temporal_analysis(dfs)
    recent_cities, city_country_comp = spatial_analysis(dfs)
    uncertainty_by_year, uncertainty_by_country = uncertainty_analysis(dfs)
    global_df_extremes, country_extremes = extreme_event_analysis(dfs)
    seasonal_patterns, country_clusters = pattern_analysis(dfs)
    hemisphere_data, continent_yearly, volatility = regional_comparison(dfs)
    
    print("\nEDA completed! Results saved to:", output_dir)
    
    # Return all analysis results
    results = {
        'annual_global': annual_global,
        'annual_country_temps': annual_country_temps,
        'recent_cities': recent_cities,
        'city_country_comp': city_country_comp,
        'uncertainty_by_year': uncertainty_by_year,
        'uncertainty_by_country': uncertainty_by_country,
        'global_df_extremes': global_df_extremes,
        'country_extremes': country_extremes,
        'seasonal_patterns': seasonal_patterns,
        'country_clusters': country_clusters,
        'hemisphere_data': hemisphere_data,
        'continent_yearly': continent_yearly,
        'volatility': volatility
    }
    
    return results

# Run the analysis if this script is executed directly
if __name__ == "__main__":
    results = main()
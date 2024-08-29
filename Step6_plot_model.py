import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Path to the directory containing .csv files
directory_path = r'D:\Disertasi\plot\data'

# Function to load and filter data by percentiles
def load_and_filter_data(file_path, percentiles):
    df = pd.read_csv(file_path, usecols=['year', 'longitude', 'latitude', 'percentile', 'sum_rcp26', 'sum_rcp45', 'sum_rcp85'])
    return df[df['percentile'].isin(percentiles)]

# Function to average data by year and percentile
def average_by_year_and_percentile(df):
    return df.groupby(['year','longitude', 'latitude', 'percentile']).mean().reset_index()

# Define the percentiles we are interested in
percentiles_of_interest = [50]

# Load, filter, and average ensemble data by year and percentile
ensemble_data = load_and_filter_data(os.path.join(directory_path, 'all_combined_data.csv'), percentiles_of_interest)
ensemble_data_avg = average_by_year_and_percentile(ensemble_data)

# Initialize dictionary to store averaged data for each model
model_percentile_data_avg = {}

# Function to average model data by year and percentile
def process_and_average_model_data(file_path, model_name):
    model_data = load_and_filter_data(file_path, percentiles_of_interest)
    model_data_avg = average_by_year_and_percentile(model_data)
    return model_data_avg

# Process, filter, and average each model's data by year and percentile
for filename in os.listdir(directory_path):
    if filename.startswith("model_combined_data") and filename.endswith(".csv"):
        # Determine model name from the file name
        model_name = filename.split('_')[-1].replace('.csv', '')
        # Load, filter, and average model data by year and percentile
        model_data_avg = process_and_average_model_data(os.path.join(directory_path, filename), model_name)
        # Store the averaged data in the dictionary with the model name as the key
        model_percentile_data_avg[model_name] = model_data_avg
        
# Function to calculate the difference between model data and ensemble data
def calculate_difference(model_data_avg, ensemble_data_avg):
    # Merge model data with ensemble data on year, longitude, latitude, and percentile
    merged_df = pd.merge(model_data_avg, ensemble_data_avg, on=['year', 'longitude', 'latitude', 'percentile'], suffixes=('_model', '_ensemble'))
    
    # Calculate the difference for each sum_rcp variable (percentage difference)
    merged_df['sum_rcp26_diff'] = 100 * ((merged_df['sum_rcp26_model'] - merged_df['sum_rcp26_ensemble'])/merged_df['sum_rcp26_model'])
    merged_df['sum_rcp45_diff'] =  100 * ((merged_df['sum_rcp45_model'] - merged_df['sum_rcp45_ensemble'])/merged_df['sum_rcp45_ensemble'])
    merged_df['sum_rcp85_diff'] = 100 * ((merged_df['sum_rcp85_model'] - merged_df['sum_rcp85_ensemble'])/merged_df['sum_rcp85_ensemble'])
    
    # Select the relevant columns for the final DataFrame
    diff_df = merged_df[['year', 'longitude', 'latitude', 'percentile', 'sum_rcp26_diff', 'sum_rcp45_diff', 'sum_rcp85_diff']]
    
    return diff_df

# Dictionary to store the difference data for each model
model_diff_data = {}

# Calculate the difference for each model and store it in the dictionary
for model_name, model_data_avg in model_percentile_data_avg.items():
    model_diff_df = calculate_difference(model_data_avg, ensemble_data_avg)
    model_diff_data[model_name] = model_diff_df

# Function to create a map
def create_map(ax, title):
    ax.set_extent([104.0, 110.5, 8, 22.5], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE)
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title)

# Function to plot the spatial map for each model and scenario
def plot_spatial_maps(scenario_diff_column, title):
    fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(20, 20), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    cmap = plt.get_cmap('viridis')  # Use plt.get_cmap to get the colormap
    
    min_value = float('inf')
    max_value = float('-inf')
    
    # Determine the global min and max for color scaling across all models
    for model_name, diff_df in model_diff_data.items():
        diff_2100 = diff_df[diff_df['year'] == 2100]
        min_value = min(min_value, diff_2100[scenario_diff_column].min())
        max_value = max(max_value, diff_2100[scenario_diff_column].max())
    
    for ax, (model_name, diff_df) in zip(axes, model_diff_data.items()):
        # Filter data for the year 2100
        diff_2100 = diff_df[diff_df['year'] == 2100]
        lon = diff_2100['longitude']
        lat = diff_2100['latitude']
        rcp_data = diff_2100[scenario_diff_column]
        
        create_map(ax, f'{model_name}')
        sc = ax.scatter(lon, lat, c=rcp_data, cmap=cmap, marker='s', vmin=min_value, vmax=max_value)
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label=f'{scenario_diff_column} (%)')
    plt.suptitle(title, fontsize=16)
    plt.show()

# Plot for RCP2.6
plot_spatial_maps('sum_rcp26_diff', 'Difference in RCP2.6 Scenario in 2100')

# Plot for RCP4.5
plot_spatial_maps('sum_rcp45_diff', 'Difference in RCP4.5 Scenario in 2100')

# Plot for RCP8.5
plot_spatial_maps('sum_rcp85_diff', 'Difference in RCP8.5 Scenario in 2100')

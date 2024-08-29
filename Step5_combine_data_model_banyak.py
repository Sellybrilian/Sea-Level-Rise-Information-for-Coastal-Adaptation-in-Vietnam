############# COMBINE DATA RCP MODEL #####################
import pandas as pd
import os

# Direktori tempat file .csv berada
directory = r'D:\Disertasi\coba_banyak\fix_models\MRI-CGCM3\data\sea_level_projections' #adjust the directory of the model
model_name = 'MRI-CGCM3' #adjust the model used

# Dictionary untuk menyimpan DataFrame sementara
data_dict = {}

# Baca file site data
df_site_data = pd.read_csv(r'D:\Disertasi\Bayu_try\fix\points.csv', sep=';')
df_site_data['latitude'] = pd.to_numeric(df_site_data['latitude'], errors='coerce')
df_site_data['longitude'] = pd.to_numeric(df_site_data['longitude'], errors='coerce')

# Loop untuk membaca setiap file .csv
for filename in os.listdir(directory):
    if filename.endswith(".csv") and 'regional' in filename:
        # Ekstrak latitude dan longitude dari nama file
        parts = filename.split('_')
        index = int(parts[1])
        rcp = parts[2]  # rcp26, rcp45, rcp85

        # Baca file .csv
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        
        # Pastikan kolom yang diharapkan ada dalam DataFrame
        required_columns = {'year', 'exp', 'antdyn', 'antsmb',	'greendyn',	'greensmb',	'glacier',	'landwater','GIA', 'antnet', 'greennet', 'percentile', 'sum'}
        if not required_columns.issubset(df.columns):
            print(f"Skipping file {filename} due to missing columns")
            continue

        
        # Ekstrak kolom yang diperlukan dan ubah nama kolom berdasarkan rcp
        df = df[['year', 'exp', 'antdyn', 'antsmb', 'greendyn', 'greensmb', 'glacier', 'landwater', 'GIA', 'antnet', 'greennet', 'percentile', 'sum']].copy()

        # Ubah nama kolom dengan menambahkan nilai rcp
        df.rename(columns={
        'exp': f'exp_{rcp}',
        'antdyn': f'antdyn_{rcp}',
        'antsmb': f'antsmb_{rcp}',
        'greendyn': f'greendyn_{rcp}',
        'greensmb': f'greensmb_{rcp}',
        'glacier': f'glacier_{rcp}',
        'landwater': f'landwater_{rcp}',
        'GIA': f'GIA_{rcp}',
        'antnet': f'antnet_{rcp}',
        'greennet': f'greennet_{rcp}',
        'sum': f'sum_{rcp}'
        }, inplace=True)

        # Tambahkan kolom index, nama model, latitude, dan longitude
        df['index'] = index
        df['model'] = model_name

        # Tambahkan latitude dan longitude berdasarkan index
        if index < len(df_site_data):
            df['latitude'] = df_site_data.loc[index, 'latitude']
            df['longitude'] = df_site_data.loc[index, 'longitude']
        else:
            print(f"Index {index} out of bounds for site data")

        # Simpan DataFrame ke dalam dictionary
        if index not in data_dict:
            data_dict[index] = df
        else:
            data_dict[index] = pd.merge(data_dict[index], df, on=['year', 'percentile', 'index', 'model', 'latitude', 'longitude'], how='outer')

# Gabungkan semua DataFrame menjadi satu
final_df = pd.concat(data_dict.values(), ignore_index=True)

# Hapus kolom 'index'
final_df.drop(columns=['index'], inplace=True)

# Urutkan kolom sesuai dengan urutan yang diinginkan
final_df = final_df[['model', 'latitude', 'longitude', 'year', 'percentile'] + 
                    [col for col in final_df.columns if col.startswith('exp_')] +
                    [col for col in final_df.columns if col.startswith('antdyn_')] +
                    [col for col in final_df.columns if col.startswith('antsmb_')] +
                    [col for col in final_df.columns if col.startswith('greendyn_')] +
                    [col for col in final_df.columns if col.startswith('greensmb_')] +
                    [col for col in final_df.columns if col.startswith('glacier_')] +
                    [col for col in final_df.columns if col.startswith('landwater_')] +
                    [col for col in final_df.columns if col.startswith('GIA_')] +
                    [col for col in final_df.columns if col.startswith('antnet_')] +
                    [col for col in final_df.columns if col.startswith('greennet_')] +
                    [col for col in final_df.columns if col.startswith('percentile_')] +
                    [col for col in final_df.columns if col.startswith('sum_')]]

# Tentukan nama file berdasarkan nama model
output = r'D:\Disertasi\plot\data'
output_file = os.path.join(output, f'model_combined_data_{model_name}.csv')

# Simpan hasil ke file baru
final_df.to_csv(output_file, index=False)

print(f"Data combined and saved successfully to {output_file}.")














############# COMBINE DATA EXP #####################

import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.colors as mcolors
from collections import defaultdict

# Path to the directories containing .csv files
ensemble_directory = r'D:\Disertasi\plot\all'
model_directory = r'D:\Disertasi\plot\data'

# Dictionary to store data from ensemble models
ensemble_data_dict = {}

# Loop to read each .csv file in the ensemble directory
for filename in os.listdir(ensemble_directory):
    if filename.endswith(".csv"):
        # Extract latitude, longitude, and RCP from the filename
        parts = filename.split('_')
        latitude = float(parts[0])  # Ensure these are float
        longitude = float(parts[1])  # Ensure these are float
        rcp = parts[2].split('.')[0]

        # Read the .csv file
        file_path = os.path.join(ensemble_directory, filename)
        ensemble_df = pd.read_csv(file_path)

        # Ensure required columns are present
        required_columns = {'year', 'percentile', 'exp'}
        if not required_columns.issubset(ensemble_df.columns):
            print(f"Skipping file {filename} due to missing columns")
            continue

        # Select necessary columns and rename 'exp' column based on RCP
        ensemble_df = ensemble_df[['year', 'percentile', 'exp']].copy()
        ensemble_df.rename(columns={'exp': f'exp_{rcp}'}, inplace=True)

        # Add latitude and longitude columns
        ensemble_df['latitude'] = latitude
        ensemble_df['longitude'] = longitude

        # Use (latitude, longitude) as the key for the dictionary
        key = (latitude, longitude)
        if key not in ensemble_data_dict:
            ensemble_data_dict[key] = ensemble_df
        else:
            # Merge data frames on 'year', 'percentile', 'latitude', and 'longitude'
            ensemble_data_dict[key] = pd.merge(
                ensemble_data_dict[key], ensemble_df,
                on=['latitude', 'longitude', 'year', 'percentile'],
                how='outer'
            )

# Combine all data into a single DataFrame
ensemble_combined_df = pd.concat(ensemble_data_dict.values(), ignore_index=True)

# Define the desired column order
ensemble_ordered_columns = ['latitude', 'longitude', 'year', 'percentile', 'exp_rcp26', 'exp_rcp45', 'exp_rcp85']

# Ensure all necessary columns are present
for col in ensemble_ordered_columns:
    if col not in ensemble_combined_df.columns:
        ensemble_combined_df[col] = pd.NA

# Arrange columns in the specified order
ensemble_combined_df = ensemble_combined_df[ensemble_ordered_columns]

# Filter data for the period 2081-2100 and calculate the mean for RCP4.5
ensemble_filtered_data = ensemble_combined_df[(ensemble_combined_df['year'] >= 2081) & (ensemble_combined_df['year'] <= 2100)]
ensemble_mean_projection = ensemble_filtered_data.groupby(['latitude', 'longitude'])['exp_rcp45'].mean().reset_index()
ensemble_mean_projection.rename(columns={'exp_rcp45': 'mean_rcp45'}, inplace=True)






# Repeat similar steps for model data
model_data = defaultdict(list)

for filename in os.listdir(model_directory):
    if filename.startswith("model_combined_data") and filename.endswith(".csv"):
        model_name = filename.split('_')[-1].replace('.csv', '')
        file_path = os.path.join(model_directory, filename)
        model_df = pd.read_csv(file_path)
        model_df['model'] = model_name  # Add model name column for clarity
        model_data[model_name].append(model_df)

model_combined_df = {model: pd.concat(model_dfs, ignore_index=True) for model, model_dfs in model_data.items()}

# Filter and select relevant columns
model_selected_columns_data = {}
model_columns_to_select = ['model', 'longitude', 'latitude', 'year', 'percentile', 'exp_rcp45']

for model, model_df in model_combined_df.items():
    if all(column in model_df.columns for column in model_columns_to_select):
        model_selected_columns_data[model] = model_df[model_columns_to_select]

model_mean_projection_data = []

for model, model_df in model_selected_columns_data.items():
    model_filtered_df = model_df[(model_df['year'] >= 2081) & (model_df['year'] <= 2100)]
    model_mean_projection = model_filtered_df.groupby(['latitude', 'longitude'])['exp_rcp45'].mean().reset_index()
    model_mean_projection.rename(columns={'exp_rcp45': 'mean_rcp45'}, inplace=True)
    model_mean_projection['model'] = model
    model_mean_projection_data.append(model_mean_projection)

model_combined_mean_projection = pd.concat(model_mean_projection_data, ignore_index=True)




#calculate difference between ensemble and model

model_differences_data = []

for model_projection in model_mean_projection_data:
    merged_df = pd.merge(ensemble_mean_projection, model_projection, on=['latitude', 'longitude'], suffixes=('_ensemble', '_model'))
    merged_df['difference'] = merged_df['mean_rcp45_model'] - merged_df['mean_rcp45_ensemble']
    merged_df['difference'] = merged_df['difference'].abs()  # Make the difference absolute
    model_differences_data.append(merged_df)

# Combine all model differences into a single DataFrame
model_differences_combined = pd.concat(model_differences_data, ignore_index=True)

# Drop the 'mean_rcp45_ensemble' and 'mean_rcp45_model' columns
model_differences_combined.drop(columns=['mean_rcp45_ensemble', 'mean_rcp45_model'], inplace=True)





#plot Model exp
shapefile_path = r'D:\Disertasi\outgoing_20240529\outgoing\gadm41_VNM_shp\gadm41_VNM_0.shp'
gdf = gpd.read_file(shapefile_path)

# Define custom colormap
colors = ["#f9f8d1", "#f3f098", "#f7d58e", "#f2bf69", "#eba054", "#e37841", "#d33e33", "#942a26", "#6f1c1a", "#5c1120"]
cmap = mcolors.ListedColormap(colors)

# Define the boundaries and normalizer
boundaries = [0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30]
norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

# Set up the figure and subplots
num_models = len(model_combined_mean_projection['model'].unique())
fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(15, 10), constrained_layout=True)

# Ensure axes is a 1D array even if there's only one subplot
axes = axes.flatten() if num_models > 1 else [axes]

# Plot data for each model
for i, (model, ax) in enumerate(zip(model_combined_mean_projection['model'].unique(), axes)):
    # Filter the data for the current model
    model_data = model_combined_mean_projection[model_combined_mean_projection['model'] == model]
    
    # Plot the shapefile (base map)
    gdf.plot(ax=ax, color='lightgrey', edgecolor='black')
    
    # Scatter plot of the mean_rcp45 data
    scatter = ax.scatter(model_data['longitude'], model_data['latitude'],
                         c=model_data['mean_rcp45'], cmap=cmap, norm=norm, s=10)
    
    # Set plot title
    ax.set_title(f'{model}')
    
    # Set axis labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Customize ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(axis='x', rotation=45)

fig.suptitle("Sub-model mean projection of the time-averaged dynamic\nand steric sea level changes for the period 2081–2100", fontsize=20)

# Add a colorbar
cbar = fig.colorbar(scatter, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('(m)')

# Show the plot
plt.show()

















######## RMS #########

# Hitung rata-rata waktu untuk RCP4.5
mean_projection = filtered_data.groupby(['latitude', 'longitude'])['exp_rcp45'].mean().reset_index()

# Ganti nama kolom untuk kejelasan
mean_projection.rename(columns={'exp_rcp45': 'mean_rcp45'}, inplace=True)

# Gabungkan mean_projection dengan filtered_data untuk perhitungan RMS spread
merged_data = filtered_data.merge(mean_projection, on=['latitude', 'longitude'], how='left')

# Hitung RMS spread untuk RCP4.5
merged_data['squared_diff'] = (merged_data['exp_rcp45'] - merged_data['mean_rcp45'])**2
rms_spread = merged_data.groupby(['latitude', 'longitude'])['squared_diff'].mean().reset_index()
rms_spread['rms_spread_rcp45'] = np.sqrt(rms_spread['squared_diff'])

# Gabungkan hasil RMS spread dengan mean_projection
final_projection = mean_projection.merge(rms_spread[['latitude', 'longitude', 'rms_spread_rcp45']], on=['latitude', 'longitude'])

# Baca shapefile dengan Geopandas
shapefile_path = r'D:\Disertasi\outgoing_20240529\outgoing\gadm41_VNM_shp\gadm41_VNM_0.shp'
gdf = gpd.read_file(shapefile_path)

# Define custom colormap
colors = ["#f9f8d1", "#f3f098", "#f7d58e", "#f2bf69", "#eba054", "#e37841", "#d33e33", "#942a26", "#6f1c1a", "#5c1120"]
cmap = mcolors.ListedColormap(colors)

# Define the boundaries and normalizer
boundaries = [0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30]
norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

# Create a GeoDataFrame from mean_projection
gdf_points = gpd.GeoDataFrame(final_projection, geometry=gpd.points_from_xy(final_projection.longitude, final_projection.latitude))

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color='white', edgecolor='black')

# Plot the points with the custom colormap and normalization
sc = gdf_points.plot(ax=ax, column='rms_spread_rcp45', cmap=cmap, norm=norm, marker='s', legend=False, markersize=60, edgecolor='none', alpha=0.80)

# Menambahkan judul dan label sumbu
plt.title(" Root-mean square (RMS) spread (deviation) of the individual model result around \nthe ensemble mean (metres) for period 2081–2100")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Customize the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []  # Dummy array for the ScalarMappable
cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
cbar.set_ticks([0, 0.06, 0.12, 0.18, 0.24, 0.30])
cbar.set_ticklabels(['0.00', '0.06', '0.12', '0.18', '0.24', '0.30'])

# Menampilkan plot
plt.show()






# Create plot difference
fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(15, 10), constrained_layout=True)
axes = axes.flatten()  # Flatten to easily iterate over

# Unique models for plotting
unique_models = model_differences_combined['model'].unique()

# Plot each model's differences
for i, model in enumerate(unique_models):
    # Subset the data for the current model
    model_data = model_differences_combined[model_differences_combined['model'] == model]
    
    # Plotting
    ax = axes[i]
    gdf.plot(ax=ax, color='lightgrey', edgecolor='black') 
    sc = ax.scatter(model_data['longitude'], model_data['latitude'], c=model_data['difference'], cmap='seismic', s=10, marker='s')
    ax.set_title(f'{model}')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # Customize ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(axis='x', rotation=45)

fig.suptitle("Differences Between Model and Ensemble Projections in Dynamic \nand Steric Sea Level Changes (2081–2100)", fontsize=20)


cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('(m)')

# Adjust layout to prevent overlap
plt.show()



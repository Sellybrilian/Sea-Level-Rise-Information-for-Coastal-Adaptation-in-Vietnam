import pandas as pd
from config import settings
from slr_pkg import extract_dyn_steric_regression  # found in __init__.py__
from slr_pkg import models

def extract_cmip5_steric_dyn_regression(df, idx):
    """
    Finds all CMIP model names, then calculates the regression parameters
    between global and local sea level projections for the given locations.
    :param df: DataFrame of site location's metadata
    :param index: Index of the site for output file naming
    """
    print(f'Running function extract_cmip5_steric_dyn_regression for site index {idx}')
    print(df)
    
    # Specify the emission scenarios
    scenarios = ['rcp26', 'rcp45', 'rcp85']
    
    # Helper function to check if a model exists in the lists
    def model_exists(model_name):
        return model_name in models.cmip5_names() or model_name in models.cmip5_names_marginal()

    # Retrieve the cmip_sea setting
    cmip_sea = settings["cmipinfo"]["cmip_sea"]

    # Select CMIP5 models to use
    if cmip_sea == 'all':
        model_names = models.cmip5_names()
    elif cmip_sea == 'marginal':
        model_names = models.cmip5_names_marginal()
    elif model_exists(cmip_sea):
        model_names = [cmip_sea]
    else:
        raise UnboundLocalError(
            'The selected CMIP5 models to use - cmip_sea = ' +
            f'{cmip_sea} - ' +
            'is not recognised')
    
    # Calculate the regression parameters and plot the results
    extract_dyn_steric_regression(model_names, df, scenarios, idx)
    

def main():
    """
    Calculate the regression between local sea level change and global mean
    sea level rise from thermal expansion for multiple sites specified in a CSV file.
    """
    # Load site data from a CSV file
    try:
        df_site_data = pd.read_csv(r'D:\Disertasi\Bayu_try\fix\points.csv', sep=';')
        df_site_data['latitude'] = pd.to_numeric(df_site_data['latitude'], errors='coerce')
        df_site_data['longitude'] = pd.to_numeric(df_site_data['longitude'], errors='coerce')
    except Exception as e:
        print(f"Error reading or processing CSV file: {e}")
        return

    # Format latitude and longitude to two decimal places for display and further processing
    df_site_data['latitude'] = df_site_data['latitude'].round(2)
    df_site_data['longitude'] = df_site_data['longitude'].round(2)

    # Display loaded site data with two decimals
    print(f'Loaded site names and coordinates:\n{df_site_data}')

    # Iterate over each site and perform the regression analysis
    for idx, row in df_site_data.iterrows():
        site_name = row['site_name']
        latitude = row['latitude']
        longitude = row['longitude']
        print(f"Processing site {site_name} at Latitude {latitude}, Longitude {longitude}")

        # Construct the site metadata dataframe for the current site
        df_current_site = pd.DataFrame({
            'site_name': [site_name],
            'latitude': [latitude],
            'longitude': [longitude]
        })

        # Log the current site data
        print(f"Current site DataFrame:\n{df_current_site}")
        

        # Perform the regression analysis for the current site
        extract_cmip5_steric_dyn_regression(df_current_site, idx)

if __name__ == '__main__':
    main()



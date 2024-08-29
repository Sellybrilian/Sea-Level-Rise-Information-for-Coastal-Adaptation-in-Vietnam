import os
import numpy as np
import pandas as pd

from config import settings
from directories import read_dir, makefolder
from slr_pkg import abbreviate_location_name, plot_ij  # found in __init.py__
from slr_pkg import cmip, cubeutils, models, whichbox
from tide_gauge_locations import extract_site_info, distance


def accept_reject_cmip(cube, model, site_loc, cmip_i, cmip_j, site_lat,
                       site_lon, unit_test=False):
    """
    Accept or reject selected CMIP grid box based on a user input.
    If CMIP grid box is rejected, search neighbouring grid boxes until a
    suitable one is found.
    :param cube: cube containing zos field from CMIP models
    :param model: CMIP model name
    :param site_loc: name of the site location
    :param cmip_i: CMIP coord of site location's latitude
    :param cmip_j: CMIP coord of site location's longitude
    :param site_lat: latitude of the site location
    :param site_lon: longitude of the site location
    :param unit_test: flag to disable plotting for unit testing purposes
    :return: Selected CMIP coords or None if user doesn't confirm grid box
    selection
    """
    decision = "Y"
    if decision == 'Y' or decision == 'y':
        if not unit_test:
            plot_ij(cube, model, site_loc, [cmip_i, cmip_j],
                    site_lat, site_lon)
        return cmip_i, cmip_j
    elif decision == 'N' or decision == 'n':
        print('Selecting another CMIP grid box')
        return None, None
    else:
        raise TypeError('Response needs to be Y or N')


def calc_radius_range(radius):
    """
    Calculate the maximum distance to search for ocean grid point.
    :param radius: Maximum range to search for ocean point
    :return: x_radius_range, y_radius_range
    """
    x_radius_range = list(range(-radius, radius + 1))
    y_radius_range = list(range(-radius, radius + 1))
    return x_radius_range, y_radius_range


def check_cube_mask(cube):
    """
    Check if the cube has a scalar mask. If so, then re-mask the cube to
    remove grid points that are exactly equal to 0.
    **NOTES:
    - Land points have a value of 0, ocean points have non-zero (both
    positive and negative) values.
    - Because the CMIP models were interpolated onto a 1x1 grid before
    the mask was checked some interpolation issues may still arise.
    :param cube: cube containing zos field from CMIP models
    :return: original cube, or re-masked cube
    """
    apply_mask = False
    try:
        cube.data.mask
        apply_mask = not isinstance(cube.data.mask, (list, tuple, np.ndarray))
    except AttributeError:
        apply_mask = True

    if apply_mask:
        new_mask = (cube.data == 0.0)
        print(f'Scalar mask: Re-masking the cube to mask cells = 0')
        cube = cube.copy(data=np.ma.array(cube.data, mask=new_mask))

    return cube


def extract_lat_lon(df):
    """
    Extracts site location's metadata, calls wraplongitude() to convert
    longitude.
    :param df: DataFrame of site location's metadata
    :return: latitude, longitude, converted longitude
    """
    site = df.name
    lat = df['latitude']
    lon_orig = df['longitude']
    print(f'Site location: {site}, Lat: {lat:.2f}, Lon: {lon_orig:.2f}')

    lon = whichbox.wraplongitude(lon_orig)

    return lat, lon_orig, lon


def extract_ssh_data(cmip_sea):
    """
    Get a list of appropriate CMIP models to use. Load SSH data for each
    model on a re-gridded 1x1 grid.
    :param cmip_sea: variable to distinguish between which CMIP models to use
    :return: CMIP model names, and associated SSH data cubes
    """
    cmip_dir = settings["cmipinfo"]["sealevelbasedir"]
    
    def model_exists(model_name):
        return model_name in models.cmip5_names() or model_name in models.cmip5_names_marginal()

    if cmip_sea == 'all':
        model_names = models.cmip5_names()
    elif cmip_sea == 'marginal':
        model_names = models.cmip5_names_marginal()
    elif model_exists(cmip_sea):
        model_names = [cmip_sea]
    else:
        raise UnboundLocalError(f'The selected CMIP5 models to use - ' +
                                f'cmip_sea = {cmip_sea} - is not recognised')

    cubes = []
    cmip_dict = cmip.model_dictionary()
    for model in model_names:
        print(f'Getting data for {model} model')
        cmip_date = cmip_dict[model]['historical']
        cmip_file = f'{cmip_dir}zos_Omon_{model}_historical_{cmip_date}.nc'
        cube = cubeutils.loadcube(cmip_file, ncvar='zos')[0]
        cubes.append(cube.slices(['latitude', 'longitude']).next())

    return model_names, cubes

def find_ocean_pt(zos_cube_in, model, site_loc, site_lat, site_lon):
    """
    Searches for the nearest appropriate ocean point(s) in the CMIP model
    adjacent to the site location. Initially, finds the model grid box
    indices of the given location. Then, searches surrounding boxes until an
    appropriate ocean point is found - needs to be accepted by the user.
    **NOTES:
    - The GCM data have been interpolated to a common 1 x 1 degree grid
    :param zos_cube_in: cube containing zos field from CMIP models
    :param model: CMIP model name
    :param site_loc: name of the site location
    :param site_lat: latitude of the site location
    :param site_lon: longitude of the site location
    :return: model grid box indices
    """
    (i, j), = whichbox.find_gridbox_indicies(zos_cube_in, [(site_lon, site_lat)])
    grid_lons = zos_cube_in.coord('longitude').points
    grid_lats = zos_cube_in.coord('latitude').points
    close_glon = grid_lons[i]
    close_glat = grid_lats[j]
    
    grid_lons_out = []
    grid_lats_out = []
    grid_i_out = []
    grid_j_out = []
    dist_out = []
    for ilon in np.arange(-7, 8, 1):
        clon = close_glon + ilon
        ci = i + ilon
        for ilat in np.arange(-7, 8, 1):
            clat = close_glat + ilat
            cj = j + ilat
            grid_lons_out.append(clon)
            grid_lats_out.append(clat)
            grid_i_out.append(ci)
            grid_j_out.append(cj)
            dist_out.append(distance(clat, clon, site_lat, site_lon))
    
    grid_dist_df = pd.DataFrame({'glat': grid_lats_out,
                                 'glon': grid_lons_out,
                                 'gj': grid_j_out,
                                 'gi': grid_i_out,
                                 'dist': dist_out})
    grid_dist_df = grid_dist_df.sort_values(by='dist').reset_index(drop=True)

    zos_cube = check_cube_mask(zos_cube_in)
    for index, rows in grid_dist_df.iterrows():
        if not zos_cube.data.mask[int(rows['gj']), int(rows['gi'])]:
            plot_ij(zos_cube, model, site_loc, [int(rows['gi']), int(rows['gj'])],
                    site_lat, site_lon)
            return int(rows['gi']), int(rows['gj']), rows['glon'], rows['glat']

def ocean_point_wrapper(df, model_names, cubes):
    """
    Wrapper script to extract relevant metadata for site location, needed to
    select the nearest CMIP grid box. Collates the CMIP model name, i and j
    coords selected and lat and lon value of the site.
    Writes the results to a single .csv file assuming write=True.
    :param df: DataFrame of site location's metadata
    :param model_names: CMIP model names
    :param cubes: cube containing zos field from CMIP models
    """
    for site_loc in df.index.values:
        df_site = df.loc[site_loc]
        lat, lon_orig, lon = extract_lat_lon(df_site)

        result = []
        for n, zos_cube in enumerate(cubes):
            model = model_names[n]
            i, j, pt_lon, pt_lat = find_ocean_pt(zos_cube, model, site_loc, lat, lon)
            result.append([model, i, j, pt_lon, pt_lat])

        write_i_j(site_loc, result, lat, lon_orig)

def write_i_j(site_loc, result, site_lat, lon_orig):
    """
    Convert the grid indices to a data frame and writes to file.
    :param site_loc: name of site location
    :param result: grid indices and coordinates of nearest ocean point
    :param site_lat: latitude of the site location
    :param lon_orig: longitude of site
    """
    df_out = pd.DataFrame(result, columns=['Model', 'i', 'j', 'box_lon', 'box_lat'])
    out_cmipdir = read_dir()[0]
    makefolder(out_cmipdir)
    loc_abbrev = abbreviate_location_name(site_loc)
    outfile = os.path.join(out_cmipdir, f'{loc_abbrev}_ij_1x1_coords.csv')

    with open(outfile, 'w') as ofp:
        ofp.write(f'Location: {site_loc}\n')
        ofp.write(f'Latitude: {site_lat:8.3f}\n')
        ofp.write(f'Longitude: {lon_orig:8.3f}\n')
        df_out.to_csv(ofp, index=False)

def main():
    try:
        df_site_data = pd.read_csv(r'D:\Disertasi\Bayu_try\fix\points.csv', sep=';')
        df_site_data['latitude'] = pd.to_numeric(df_site_data['latitude'], errors='coerce')
        df_site_data['longitude'] = pd.to_numeric(df_site_data['longitude'], errors='coerce')
    except Exception as e:
        print(f"Error reading or processing CSV file: {e}")
        return

    df_site_data['latitude'] = df_site_data['latitude'].round(2)
    df_site_data['longitude'] = df_site_data['longitude'].round(2)
    print(f'Loaded site names and coordinates:\n{df_site_data}')

    try:
        cmip_models, ssh_cubes = extract_ssh_data(settings["cmipinfo"]["cmip_sea"])
        for index, row in df_site_data.iterrows():
            sl = row['site_name']
            lati = row['latitude']
            long = row['longitude']
            print(f"Processing site {sl} at Latitude {lati}, Longitude {long}")
        
        ocean_point_wrapper(df_site_data, cmip_models, ssh_cubes)
    except KeyError as e:
        print(f"Error in processing data: {e}")

if __name__ == '__main__':
    main()


 



# List of script files to execute
scripts = [
    'step2_extract_steric_dyn_regression_model_banyak.py',
    'step3_process_regional_sealevel_projections_model_banyak.py',
    'step4_plot_regional_sealevel_model_banyak.py'
]

# Execute each script
for script in scripts:
    with open(script) as f:
        exec(f.read())

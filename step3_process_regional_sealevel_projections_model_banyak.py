import os
import iris
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from netCDF4 import Dataset



from config import settings
from slr_pkg import abbreviate_location_name  # found in __init.py__
from directories import read_dir, makefolder

def calc_baseline_period(sci_method, yrs):
    if sci_method == 'global':
        byr1 = 1986.
        byr2 = 2005.
        G_offset = 0.0
        print("Baseline period = ", byr1, "to", byr2)
    elif sci_method == 'UK':
        byr1 = 1981.
        byr2 = 2000.
        G_offset = 0.011
        print("Baseline period = ", byr1, "to", byr2)

    midyr = (byr2 - byr1 + 1) * 0.5 + byr1

    return yrs[0] - midyr, G_offset

def calc_future_sea_level_at_site(df, site_loc, scenario):
    print('running function calc_future_sea_level_at_site')
    loc_coords = [df.at[site_loc, 'latitude'], df.at[site_loc, 'longitude']]

    np.random.seed(18)

    mcdir = settings["montecarlodir"]

    components = ['exp', 'antdyn', 'antsmb', 'greendyn', 'greensmb',
                  'glacier', 'landwater']
    nesm, nyrs, yrs = get_projection_info(mcdir, scenario)

    nsmps = 200000
    array_dims = [nesm, nsmps, nyrs]

    montecarlo_G, montecarlo_R = calculate_sl_components(mcdir, components,
                                                         scenario, site_loc,
                                                         loc_coords, yrs,
                                                         array_dims)

    G_df, R_df = calculate_summary_timeseries(components, yrs, montecarlo_G,
                                              montecarlo_R)

    sealev_ddir = read_dir()[4]
    districtname = f'loc_{site_loc}'
    file_header = '_'.join([districtname, scenario, "projection", "2100"])
    G_file = '_'.join([file_header, 'global']) + '.csv'
    R_file = '_'.join([file_header, 'regional']) + '.csv'

    makefolder(sealev_ddir)
    G_df.to_csv(os.path.join(sealev_ddir, G_file))
    R_df.to_csv(os.path.join(sealev_ddir, R_file))

def calc_gia_contribution(sci_method, yrs, nyrs, nsmps, coords):
    print('running function calc_gia_contribution')

    nGIA, GIA_vals = read_gia_estimates(sci_method, coords)

    Tdelta, G_offset = calc_baseline_period(sci_method, yrs)

    unit_series = (np.arange(nyrs) + Tdelta) * 0.001
    GIA_unit_series = np.ones([nsmps, nyrs]) * unit_series

    rgiai = np.random.randint(nGIA, size=nsmps)

    GIA_series = GIA_unit_series.transpose() * GIA_vals[rgiai]

    print('GIA_vals (mm/yr) = ', GIA_series)

    return GIA_series, G_offset

def calculate_sl_components(mcdir, components, scenario, site_loc, loc_coords,
                            yrs, array_dims):
    print('running function calculate_sl_components')
    ncomps = len(components)

    nesm, nsmps, nyrs = array_dims

    tlat, tlon = loc_coords

    sci_method = settings["sciencemethod"]

    nFPs, FPlist = setup_FP_interpolators(components, sci_method)

    resamples = np.random.choice(nesm, nsmps)
    rfpi = np.random.randint(nFPs, size=nsmps)

    offset_slopes = {'exp': 0.405,
                     'antnet': 0.095,
                     'antsmb': -0.025,
                     'antdyn': 0.120,
                     'greennet': 0.125,
                     'greensmb': 0.040,
                     'greendyn': 0.085,
                     'glacier': 0.270,
                     'landwater': 0.105}

    montecarlo_R = np.zeros((ncomps + 1, nyrs, nsmps))
    montecarlo_G = np.zeros((ncomps, nyrs, nsmps))

    GIA_series, G_offset = calc_gia_contribution(sci_method, yrs, nyrs, nsmps,
                                                 loc_coords)
    montecarlo_R[-1, :, :] = GIA_series[:, :]

    for cc, comp in enumerate(components):
        print(f'cc = {cc:d}, comp = {comp}')

        offset = G_offset * offset_slopes[comp]

        cube = iris.load_cube(os.path.join(mcdir, f'{scenario}_{comp}.nc'))
        montecarlo_G[cc, :, :] = cube.data[:, resamples] + offset

        if comp == 'exp':
            if sci_method == 'global':
                coeffs = load_CMIP5_slope_coeffs(site_loc, scenario)
                rand_coeffs = np.random.choice(coeffs, size=nsmps,
                                               replace=True)
            elif sci_method == 'UK':
                coeffs, weights = load_CMIP5_slope_coeffs_UK(scenario)
                rand_coeffs = np.random.choice(coeffs, size=nsmps,
                                               replace=True, p=weights)
            montecarlo_R[cc, :, :] = montecarlo_G[cc, :, :] * rand_coeffs
        elif comp == 'landwater':
            landwater_FP_interpolator = FPlist[0]['landwater']
            val = landwater_FP_interpolator([tlat, tlon])[0]
            montecarlo_R[cc, :, :] = montecarlo_G[cc, :, :] * val
        else:
            FPvals = []
            for FP_dict in FPlist:
                val = FP_dict[comp]([tlat, tlon])[0]
                FPvals.append(val)
            FPvals = np.array(FPvals)
            montecarlo_R[cc, :, :] = montecarlo_G[cc, :, :] * FPvals[rfpi]

    return montecarlo_G, montecarlo_R

def calculate_summary_timeseries(components, years, montecarlo_G,
                                 montecarlo_R):
    print('running function calculate_summary_timeseries')
    percentiles = [5, 10, 30, 33, 50, 67, 70, 90, 95]

    R_list = []
    G_list = []

    for cc, _ in enumerate(components):
        cgout = np.percentile(montecarlo_G[cc, :, :], percentiles, axis=1)
        G_list.append(cgout.flatten(order='F'))

        crout = np.percentile(montecarlo_R[cc, :, :], percentiles, axis=1)
        R_list.append(crout.flatten(order='F'))

    iterables = [years, percentiles]
    idx = pd.MultiIndex.from_product(iterables, names=['year', 'percentile'])
    G_df = pd.DataFrame(np.asarray(G_list).T, columns=components, index=idx)
    R_df = pd.DataFrame(np.asarray(R_list).T, columns=components, index=idx)
    R_df.rename(columns={"exp": "ocean"})

    ncomp = len(components)
    crout = np.percentile(montecarlo_R[ncomp, :, :], percentiles, axis=1)
    R_df['GIA'] = crout.flatten(order='F')

    antnet_tmpg = np.percentile(
        montecarlo_G[1, :, :] + montecarlo_G[2, :, :],
        percentiles, axis=1).flatten(order='F')
    G_df['antnet'] = pd.DataFrame(antnet_tmpg, columns=['antnet'], index=idx)
    greennet_tmpg = np.percentile(
        montecarlo_G[3, :, :] + montecarlo_G[4, :, :],
        percentiles, axis=1).flatten(order='F')
    G_df['greennet'] = pd.DataFrame(
        greennet_tmpg, columns=['greennet'], index=idx)
    antnet_tmpr = np.percentile(
        montecarlo_R[1, :, :] + montecarlo_R[2, :, :],
        percentiles, axis=1).flatten(order='F')
    R_df['antnet'] = pd.DataFrame(antnet_tmpr, columns=['antnet'], index=idx)
    greennet_tmpr = np.percentile(
        montecarlo_R[3, :, :] + montecarlo_R[4, :, :],
        percentiles, axis=1).flatten(order='F')
    R_df['greennet'] = pd.DataFrame(
        greennet_tmpr, columns=['greennet'], index=idx)

    montecarlo_Gsum = np.sum(montecarlo_G, axis=0)
    montecarlo_Rsum = np.sum(montecarlo_R, axis=0)
    cgout = np.percentile(montecarlo_Gsum, percentiles, axis=1)
    crout = np.percentile(montecarlo_Rsum, percentiles, axis=1)
    G_df['sum'] = cgout.flatten(order='F')
    R_df['sum'] = crout.flatten(order='F')

    return G_df, R_df

def create_FP_interpolator(datadir, dfile, method='linear'):
    cube = iris.load_cube(os.path.join(datadir, dfile))
    lon = cube.coord('longitude').points
    lat = cube.coord('latitude').points

    interp_object = RegularGridInterpolator((lat, lon), cube.data,
                                            method=method, bounds_error=True,
                                            fill_value=None)

    return interp_object

def get_projection_info(indir, scenario):
    sample_file = f'{scenario}_exp.nc'
    f = Dataset(f'{indir}{sample_file}', 'r')
    nesm = f.dimensions['realization'].size
    t = f.variables['time']
    nyrs = t.size
    unit_str = t.units
    first_year = int(unit_str.split(' ')[2][:4])
    f.close()

    yrs = first_year + np.arange(nyrs)

    return nesm, nyrs, yrs


def load_CMIP5_slope_coeffs(loc_name, scenario):
    print('running function load_CMIP5_slope_coeffs')

    in_zosddir = read_dir()[2]
    filename = os.path.join(in_zosddir, f'loc_{loc_name}_zos_regression.csv')
    print(f'Trying to load file: {filename}')
    if not os.path.exists(filename):
        raise FileNotFoundError(f'File not found: {filename}')
    df = pd.read_csv(filename, header=0)

    coeffs = df.loc[(df['Scenario'] == scenario)]['slope_05_00'].values

    if scenario == 'rcp26':
        rcp45_coeffs = df.loc[(df['Scenario'] == 'rcp45')]['slope_05_00'].values
        msgi = np.where(np.isnan(coeffs))[0]
        coeffs[msgi] = rcp45_coeffs[msgi]

    return coeffs

def load_CMIP5_slope_coeffs_UK(scenario):
    print('running function load_CMIP5_slope_coeffs_UK')
    in_zosdir_uk = settings["cmipinfo"]["slopecoeffsuk"]
    filename_uk = f'{scenario}_CMIP5_regress_coeffs_uk_mask_1.pickle'

    try:
        with open(os.path.join(in_zosdir_uk, filename_uk), 'rb') as f:
            data = pickle.load(f, encoding='latin1')['uk_mask_1']
    except FileNotFoundError:
        raise FileNotFoundError(filename_uk,
                                '- scenario selected does not exist')

    coeffs = data['coeffs']
    weights = data['weights']

    return coeffs, weights

def read_gia_estimates(sci_method, coords):
    print('running function read_gia_estimates')
    if sci_method == 'global':
        gia_file = settings["giaestimates"]["global"]
    elif sci_method == 'UK':
        gia_file = settings["giaestimates"]["uk"]
    else:
        raise UnboundLocalError('The selected GIA estimate - ' +
                                f'{sci_method} - is not available')

    with open(gia_file, "rb") as ifp:
        GIA_dict = pickle.load(ifp, encoding='latin1')

    GIA_vals = []
    lat, lon = coords
    for key in list(GIA_dict.keys()):
        val = GIA_dict[key]([lat, lon])[0]
        GIA_vals.append(val)

    nGIA = len(GIA_vals)
    GIA_vals = np.array(GIA_vals)

    return nGIA, GIA_vals

def setup_FP_interpolators(components, sci_method):
    print('running function setup_FP_interpolators')

    slangendir = settings["fingerprints"]["slangendir"]
    spadadir = settings["fingerprints"]["spadadir"]
    klemanndir = settings["fingerprints"]["klemanndir"]

    slangen_FPs = {}
    spada_FPs = {}
    klemann_FPs = {}

    comp = 'landwater'
    slangen_FPs[comp] = create_FP_interpolator(slangendir,
                                               comp + '_slangen_nomask.nc')

    components_todo = [c for c in components if c not in ['exp', 'landwater']]
    for comp in components_todo:
        slangen_FPs[comp] = create_FP_interpolator(slangendir,
                                                   comp + '_slangen_nomask.nc')
        spada_FPs[comp] = create_FP_interpolator(spadadir,
                                                 comp + '_spada_nomask.nc')
        klemann_FPs[comp] = create_FP_interpolator(klemanndir,
                                                   comp + '_klemann_nomask.nc')

    if sci_method == 'UK':
        FPlist = [slangen_FPs, spada_FPs]
    elif sci_method == 'global':
        FPlist = [slangen_FPs, spada_FPs, klemann_FPs]
    else:
        raise UnboundLocalError('The selected GRD fingerprint method - ' +
                                f'{sci_method} - is not available')

    nFPs = len(FPlist)

    return nFPs, FPlist

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

    scenarios = ['rcp26', 'rcp45', 'rcp85']
    for scenario in scenarios:
        for loc_name in df_site_data.index.values:
            calc_future_sea_level_at_site(df_site_data, loc_name, scenario)

if __name__ == '__main__':
    main()

import winsound
winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

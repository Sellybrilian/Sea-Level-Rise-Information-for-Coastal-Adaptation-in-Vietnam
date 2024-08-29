"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import os
import iris
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from config import settings
from tide_gauge_locations import extract_site_info, find_nearest_station_id
from slr_pkg import abbreviate_location_name  # found in __init__.py__
from main_surge import tide_gauge_library as tgl
from directories import read_dir, makefolder
from plotting_libraries import location_string, scenario_string, \
    ukcp18_colours, ukcp18_labels, calc_xlim, calc_ylim, plot_zeroline


def compute_uncertainties(df_r_list, scenarios, tg_years, tg_amsl):
    """
    Function to estimate uncertainties following Hawkins and Sutton (2009).
    :param df_r_list: regional sea level projections
    :param scenarios: emissions scenario
    :param tg_years: tide gauge years
    :param tg_amsl: annual mean sea level data
    :return: years, scenario uncertainty, model uncertainty, internal
    variability
    """
    allpmid = np.zeros([len(scenarios), len(df_r_list[0])])
    allunc = np.zeros([len(scenarios), len(df_r_list[0])])

    # Estimate internal variability from de-trended gauge data
    tg_years_arr = np.array(tg_years, dtype='int')
    vals = np.ma.masked_values(tg_amsl, -99999.)
    IntV = compute_variability(tg_years_arr / 1000., vals / 1.)

    # Get the values of the multi-index: years and percentile values
    years, percentiles = multi_index_values(df_r_list)

    for rcp_count, _ in enumerate(scenarios):
        # Get the sums of local sea level projections
        df_R = df_r_list[rcp_count]
        rlow, rmid, rupp = extract_comp_sl(df_R, percentiles, 'sum')

        allpmid[rcp_count, :] = rmid
        allunc[rcp_count, :] = (rupp - rlow) * 0.5

    # Scenario Uncertainty (expressed as 90% confidence interval)
    UncS = np.std(allpmid, axis=0) * 1.645
    # Model Uncertainty (already expressed as 90% confidence interval)
    UncM = np.mean(allunc, axis=0)

    return years, UncS, UncM, IntV


def compute_variability(x, y, factor=1.645):
    """
    Estimate internal variability from de-trended gauge data.
    :param x: tide gauge temporal data
    :param y: tide gauge data
    :param factor: multiplication factor
    :return: internal variability component of uncertainty
    """
    mask = np.ma.getmask(y)
    index = np.where(mask is True)
    new_x = np.delete(x, index)
    new_y = np.delete(y, index)
    fit = np.polyfit(new_x, new_y, 1)
    fit_data = new_x * fit[0] + fit[1]
    stdev = np.std(new_y - fit_data)

    return stdev * factor


def extract_comp_sl(df, percentiles, comp):
    """
    Get the sums of all components of local sea level projections.
    :param df: global or regional DataFrame of sea level projections
    :param percentiles: specified percentiles
    :param comp: components of sea level
    :return: sum of sea level components at lower, middle and upper percentile
    """
    # 5th percentile - based on UKCP18 percentile levels
    rlow = df.xs(percentiles[0], level='percentile')[comp].to_numpy(copy=True)
    # 50th percentile
    rmid = df.xs(percentiles[4], level='percentile')[comp].to_numpy(copy=True)
    # 95th percentile
    rupp = df.xs(percentiles[8], level='percentile')[comp].to_numpy(copy=True)

    return rlow, rmid, rupp


def multi_index_values(df_list):
    """
    Get the values of the multi-index: years and percentile values.
    :param df_list: DataFrame list of regional sea level projections
    :return: years and percentile values
    """
    df = df_list[0]
    proj_years = np.sort(list(set(list(df.index.get_level_values('year')))))
    percentiles_all = np.sort(
        [float(v) for v in list(set(list(
            df.index.get_level_values('percentile'))))])

    return proj_years, percentiles_all


def plot_figure_one(r_df_list, model_name, district_name, scenarios, sealev_fdir):
    years, percentiles = multi_index_values(r_df_list)
    rcp_colours = ukcp18_colours()[0]
    xlim = calc_xlim('proj', [], years)
    ylim = calc_ylim('proj', [], r_df_list)

    fig = plt.figure(figsize=(7.68, 3.5))
    matplotlib.rcParams['font.size'] = 7.5

    ax = fig.add_subplot(1, 2, 1)
    for rcp_count, rcp_str in enumerate(scenarios):
        r_df_rcp = r_df_list[rcp_count]
        rlow, rmid, rupp = extract_comp_sl(r_df_rcp, percentiles, 'sum')
        label = scenario_string(scenarios, rcp_count)
        ax.fill_between(years, rlow, rupp, alpha=0.3, color=rcp_colours[rcp_str], linewidth=0)
        ax.plot(years, rmid, color=rcp_colours[rcp_str], label=label)

    plot_zeroline(ax, xlim)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('left')

    ax.set_title(f'Local sea level - {model_name} loc {district_name}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Sea Level Change (m)')
    ax.legend(loc='upper left', frameon=False)

    bx = fig.add_subplot(1, 2, 2)
    comp_colours = ukcp18_colours()[1]
    comp_labels = ukcp18_labels()
    for rcp_count, rcp_str in enumerate(scenarios):
        if scenarios[rcp_count] == 'rcp85':
            break
    else:
        raise ValueError('RCP8.5 scenario not in list')

    r_df_rcp = r_df_list[rcp_count]
    for comp in ['sum', 'ocean', 'greennet', 'antnet', 'glacier', 'landwater', 'gia']:
        clow_85, cmid_85, cupp_85 = extract_comp_sl(r_df_rcp, percentiles, comp)
        colour = comp_colours[comp]
        label = comp_labels[comp]
        if comp in ['sum', 'ocean']:
            bx.fill_between(years, cupp_85, clow_85, facecolor=colour, alpha=0.3, edgecolor='None')
            bx.plot(years, cmid_85, colour, linewidth=1.5, label=label)
        else:
            bx.plot(years, cmid_85, colour, linewidth=1.5, label=label)

    plot_zeroline(bx, xlim)
    bx.set_xlim(xlim)
    bx.set_ylim(ylim)
    bx.yaxis.set_ticks_position('both')
    bx.yaxis.set_label_position('left')
    bx.legend(loc='upper left', frameon=False)
    bx.set_title(f'Sea level components - {scenario_string(rcp_str, -999)}')
    bx.set_xlabel('Year')

    fig.tight_layout()
    ffile = f'{sealev_fdir}01_{model_name}_loc{district_name}.png'
    plt.savefig(ffile, dpi=300, format='png')
    plt.close()


def plot_figure_two(r_df_list, model_name, tg_name, nflag, flag, tg_years, non_missing, tg_amsl, district_name, scenarios, sealev_fdir):
    years, percentiles = multi_index_values(r_df_list)
    rcp_colours = ukcp18_colours()[0]
    fig = plt.figure(figsize=(5, 4.5))
    matplotlib.rcParams['font.size'] = 10

    ax = fig.add_subplot(1, 1, 1)
    for rcp_count, rcp_str in enumerate(scenarios):
        r_df_rcp = r_df_list[rcp_count]
        rlow, rmid, rupp = extract_comp_sl(r_df_rcp, percentiles, 'sum')
        label = scenario_string(scenarios, rcp_count)
        ax.fill_between(years, rlow, rupp, alpha=0.3, color=rcp_colours[rcp_str], linewidth=0)
        ax.plot(years, rmid, color=rcp_colours[rcp_str], label=label)

    xlim = calc_xlim('tide', tg_years, years)
    df_tmp = pd.DataFrame([rlow, rmid, rupp])
    ylim = calc_ylim('tide', tg_amsl, df_tmp)
    plot_zeroline(ax, xlim)
    plot_tg_data(ax, nflag, flag, tg_years, non_missing, tg_amsl, tg_name)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('left')

    ax.set_title(f'Local sea level - {model_name} loc {district_name}')
    ax.set_ylabel('Sea Level Change (m)')
    ax.set_xlabel('Year')
    ax.legend(loc='upper left', frameon=False)

    fig.tight_layout()
    ffile = f'{sealev_fdir}02_{model_name}_loc{district_name}.png'
    plt.savefig(ffile, dpi=300, format='png')
    plt.close()


def plot_figure_three(g_df_list, model_name, r_df_list, global_low, global_mid, global_upp, district_name, scenarios, sealev_fdir):
    years, percentiles = multi_index_values(r_df_list)
    rcp_colours = ukcp18_colours()[0]
    xlim = calc_xlim('proj', [], years)

    for rcp_count, rcp_str in enumerate(scenarios):
        g_df_rcp = g_df_list[rcp_count]
        glow, gmid, gupp = extract_comp_sl(g_df_rcp, percentiles, 'sum')
        r_df_rcp = r_df_list[rcp_count]
        rlow, rmid, rupp = extract_comp_sl(r_df_rcp, percentiles, 'sum')
        r_ylim = calc_ylim('proj', [], r_df_rcp)
        g_ylim = calc_ylim('proj', [], g_df_rcp)
        ylim = [min(r_ylim[0], g_ylim[0]), max(r_ylim[1], g_ylim[1])]

        fig = plt.figure(figsize=(7.68, 3.5))
        matplotlib.rcParams['font.size'] = 7.5

        ax = fig.add_subplot(1, 2, 1)
        ax.plot(years, global_mid[rcp_count], color='k', linewidth=1.5, linestyle='-', label='IPCC AR5 + Levermann')
        ax.plot(years, global_low[rcp_count], color='k', linewidth=1.5, linestyle=':')
        ax.plot(years, global_upp[rcp_count], color='k', linewidth=1.5, linestyle=':')
        ax.plot(years, gmid, color='orange', linewidth=1.5, linestyle='--', label='Global Total')
        ax.fill_between(years, glow, gupp, color='orange', alpha=0.3, linewidth=0, edgecolor='None')

        plot_zeroline(ax, xlim)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.set_label_position('left')

        ax.set_title(f'Global sea level - {scenario_string(rcp_str, -999)}')
        ax.set_ylabel('Sea Level Change (m)')
        ax.set_xlabel('Year')
        ax.legend(loc='upper left', frameon=False)

        bx = fig.add_subplot(1, 2, 2)
        bx.plot(years, global_mid[rcp_count], color='k', linewidth=1.5, linestyle='-', label='Global Total')
        bx.plot(years, global_low[rcp_count], color='k', linewidth=1.5, linestyle=':')
        bx.plot(years, global_upp[rcp_count], color='k', linewidth=1.5, linestyle=':')
        bx.plot(years, rmid, color=rcp_colours[rcp_str], linewidth=1.5, linestyle='--', label=district_name)
        bx.fill_between(years, rlow, rupp, color=rcp_colours[rcp_str], alpha=0.3, linewidth=0, edgecolor='None')

        plot_zeroline(bx, xlim)
        bx.set_xlim(xlim)
        bx.set_ylim(ylim)
        bx.yaxis.set_ticks_position('both')
        bx.yaxis.set_label_position('left')
        bx.set_title(f'Local sea level - {model_name} loc {district_name} - {scenario_string(rcp_str, -999)}')
        bx.set_xlabel('Year')
        bx.legend(loc='upper left', frameon=False)

        fig.tight_layout()
        outfile = f'{sealev_fdir}03_{model_name}_loc{district_name}_{rcp_str}.png'
        plt.savefig(outfile, dpi=300, format='png')
        plt.close()


def plot_figure_four(r_df_list, model_name, district_name, scenarios, sealev_fdir):
    years, percentiles = multi_index_values(r_df_list)
    rcp_colours = ukcp18_colours()[0]

    fig = plt.figure(figsize=(5, 4.5))
    matplotlib.rcParams['font.size'] = 10

    ax = fig.add_subplot(1, 1, 1)
    for rcp_count, rcp_str in enumerate(scenarios):
        r_df_rcp = r_df_list[rcp_count]
        rlow, rmid, rupp = extract_comp_sl(r_df_rcp, percentiles, 'sum')
        label = scenario_string(scenarios, rcp_count)
        plt.plot(years, rmid, color=rcp_colours[rcp_str], linewidth=4.0, label=label)
        plt.plot(years, rupp, color=rcp_colours[rcp_str], linewidth=1.0, linestyle='--')
        plt.plot(years, rlow, color=rcp_colours[rcp_str], linewidth=1.0, linestyle='--')

    xlim = calc_xlim('proj', [], years)
    df_tmp = pd.DataFrame([rlow, rmid, rupp])
    ylim = calc_ylim('proj', [], df_tmp)
    plot_zeroline(ax, xlim)
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('left')

    plt.title(f'Local sea level - {model_name} loc {district_name}')
    plt.ylabel('Sea Level Change (m)')
    plt.xlabel('Year')
    plt.legend(loc='upper left', frameon=False)

    fig.tight_layout()
    outfile = f'{sealev_fdir}04_{model_name}_loc{district_name}.png'
    plt.savefig(outfile, dpi=300, format='png')
    plt.close()


def plot_figure_five(g_df_list, r_df_list, model_name, district_name, scenarios, sealev_fdir):
    years, percentiles = multi_index_values(r_df_list)
    comp_colours = ukcp18_colours()[1]
    comp_labels = ukcp18_labels()
    xlim = calc_xlim('proj', [], years)
    r_ylim = calc_ylim('proj', [], r_df_list)
    g_ylim = calc_ylim('proj', [], g_df_list)
    ylim = [min(r_ylim[0], g_ylim[0]), max(r_ylim[1], g_ylim[1])]

    fig = plt.figure(figsize=(7.68, 3.5))
    matplotlib.rcParams['font.size'] = 7.5

    for rcp_count, _ in enumerate(scenarios):
        ax = fig.add_subplot(1, 3, rcp_count + 1)
        g_df_rcp = g_df_list[rcp_count]
        glow, gmid, gupp = extract_comp_sl(g_df_rcp, percentiles, 'sum')
        ax.plot(years, gmid, color='k', linewidth=1.5, linestyle='--', label='Global Total')
        ax.plot(years, glow, color='k', linewidth=1.5, linestyle=':')
        ax.plot(years, gupp, color='k', linewidth=1.5, linestyle=':')

        r_df_rcp = r_df_list[rcp_count]
        for comp in ['sum', 'ocean', 'greennet', 'antnet', 'glacier', 'landwater', 'gia']:
            clow, cmid, cupp = extract_comp_sl(r_df_rcp, percentiles, comp)
            colour = comp_colours[comp]
            label = comp_labels[comp]
            if comp in ['sum', 'ocean']:
                ax.plot(years, cmid, colour, linewidth=1.5, label=label)
                ax.fill_between(years, cupp, clow, facecolor=colour, alpha=0.3, edgecolor='None')
            else:
                ax.plot(years, cmid, colour, linewidth=1.5, label=label)

        plot_zeroline(ax, xlim)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.set_label_position('left')
        ax.set_xlabel('Year')
        ax.set_title(f'{model_name} loc {district_name} - {scenario_string(scenarios[rcp_count], -999)}')
        if rcp_count == 0:
            ax.set_ylabel('Sea Level Change (m)')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    outfile = f'{sealev_fdir}05_{model_name}_loc{district_name}.png'
    plt.savefig(outfile, dpi=300, format='png')
    plt.close()


def plot_figure_six(r_df_list, model_name, global_low, global_mid, global_upp, district_name, scenarios, sealev_fdir):
    years, percentiles = multi_index_values(r_df_list)
    comp_colours = ukcp18_colours()[1]
    comp_labels = ukcp18_labels()
    xlim = calc_xlim('proj', [], years)
    r_ylim = calc_ylim('proj', [], r_df_list)
    g_ylim = [min(global_low[0]-0.1), max(global_upp[2])+0.1]
    ylim = [min(r_ylim[0], g_ylim[0]), max(r_ylim[1], g_ylim[1])]

    fig = plt.figure(figsize=(7.68, 3.5))
    matplotlib.rcParams['font.size'] = 7.5

    for rcp_count, _ in enumerate(scenarios):
        ax = fig.add_subplot(1, 3, rcp_count + 1)
        cmid_g = global_mid[rcp_count]
        clow_g = global_low[rcp_count]
        cupp_g = global_upp[rcp_count]
        ax.plot(years, cmid_g, color='k', linewidth=1.5, linestyle='--', label='IPCC AR5 + Levermann')
        ax.plot(years, clow_g, color='k', linewidth=1.5, linestyle=':')
        ax.plot(years, cupp_g, color='k', linewidth=1.5, linestyle=':')

        r_df_rcp = r_df_list[rcp_count]
        for comp in ['sum', 'ocean', 'greennet', 'antnet', 'glacier', 'landwater', 'gia']:
            clow, cmid, cupp = extract_comp_sl(r_df_rcp, percentiles, comp)
            colour = comp_colours[comp]
            label = comp_labels[comp]
            if comp in ['sum', 'ocean']:
                ax.plot(years, cmid, colour, linewidth=1.5, label=label)
                ax.fill_between(years, cupp, clow, facecolor=colour, alpha=0.3, edgecolor='None')
            else:
                ax.plot(years, cmid, colour, linewidth=1.5, label=label)

        plot_zeroline(ax, xlim)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.set_label_position('left')

        ax.set_xlabel('Year')
        ax.set_title(f'{model_name} loc {district_name} - {scenario_string(scenarios[rcp_count], -999)}')
        if rcp_count == 0:
            ax.set_ylabel('Sea Level Change (m)')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    outfile = f'{sealev_fdir}06_{model_name}_loc{district_name}.png'
    plt.savefig(outfile, dpi=300, format='png')
    plt.close()


def plot_figure_seven(g_df_list, r_df_list, model_name, district_name, scenarios, sealev_fdir):
    _, percentiles = multi_index_values(r_df_list)
    comp_colours = ukcp18_colours()[1]
    comp_labels = ukcp18_labels()
    xlim = ([-0.4, 7.4])
    r_ylim = calc_ylim('proj', [], r_df_list)
    g_ylim = calc_ylim('proj', [], g_df_list)
    ylim = [min(r_ylim[0], g_ylim[0]), max(r_ylim[1], g_ylim[1])]

    for rcp_count, rcp_str in enumerate(scenarios):
        if scenarios[rcp_count] == 'rcp85':
            break
    else:
        raise ValueError('RCP8.5 scenario not in list')
    r_df_rcp = r_df_list[rcp_count]
    g_df_rcp = g_df_list[rcp_count]

    fig = plt.figure(figsize=(7.68, 4.5))
    matplotlib.rcParams['font.size'] = 9

    ax_labels = []
    ax = fig.add_subplot(1, 2, 1)
    for cc, comp in enumerate(['sum', 'ocean', 'antnet', 'greennet', 'glacier', 'landwater']):
        clow_85_G, cmid_85_G, cupp_85_G = extract_comp_sl(g_df_rcp, percentiles, comp)
        colour = comp_colours[comp]
        label = comp_labels[comp]
        ax_labels.append(label)
        xpts = [cc - 0.4, cc + 0.4]
        ax.fill_between(xpts, clow_85_G[-1], cupp_85_G[-1], facecolor=colour, alpha=0.35, label=label)
        ax.plot(xpts, [cmid_85_G[-1], cmid_85_G[-1]], linewidth=2.0, color=colour)

    plot_zeroline(ax, xlim)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(ax_labels, rotation=90)
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('left')
    plt.ylim(ylim)
    plt.ylabel('Sea Level Change (m)')
    plt.title(f'Global sea level - {scenario_string(rcp_str, -999)}')

    bx = fig.add_subplot(1, 2, 2)
    bx_labels = []
    for cc, comp in enumerate(['sum', 'ocean', 'antnet', 'greennet', 'glacier', 'landwater', 'gia']):
        clow_85_R, cmid_85_R, cupp_85_R = extract_comp_sl(r_df_rcp, percentiles, comp)
        colour = comp_colours[comp]
        label = comp_labels[comp]
        bx_labels.append(label)
        xpts = [cc - 0.4, cc + 0.4]
        bx.fill_between(xpts, clow_85_R[-1], cupp_85_R[-1], facecolor=colour, alpha=0.35, label=label)
        bx.plot(xpts, [cmid_85_R[-1], cmid_85_R[-1]], linewidth=2.0, color=colour)

    plot_zeroline(bx, xlim)
    bx.set_xticks([0, 1, 2, 3, 4, 5, 6])
    bx.set_xticklabels(bx_labels, rotation=90)
    bx.yaxis.set_ticks_position('both')
    bx.yaxis.set_label_position('left')
    plt.ylim(ylim)

    plt.title(f'Local sea level - {model_name} loc {district_name} - {scenario_string(scenarios[rcp_count], -999)}')

    fig.tight_layout()
    outfile = f'{sealev_fdir}07_{model_name}_loc{district_name}_rcp85.png'
    plt.savefig(outfile, dpi=200, format='png')
    plt.close()

def plot_tg_data(ax, nflag, flag, tg_years, non_missing, tg_amsl, tg_name):
    """
    Plot the annual mean sea levels from the tide gauge data.
    :param ax: subplot number
    :param nflag: number of flagged years
    :param flag: flagged data
    :param tg_years: tide gauge years
    :param non_missing: boolean to indicate NaN values
    :param tg_amsl: annual mean sea level data
    :param tg_name: tide gauge name
    """
    if nflag > 0:
        # There are some years with less than min_valid_fraction of flag data;
        # plot these annual means as open symbols.
        print(f'Tide gauge data has been flagged for attention - ' +
              f'{tg_years[(flag & non_missing)]}')
        ax.plot(tg_years[(flag & non_missing)], tg_amsl[(flag & non_missing)],
                marker='o', mec='black', mfc='None',
                markersize=3, linestyle='None', label='TG flagged')
    if nflag < len(flag):
        ax.plot(tg_years[(~flag & non_missing)],
                tg_amsl[(~flag & non_missing)], 'ko', markersize=3,
                label=f'{location_string(tg_name)} TG')


def read_G_R_sl_projections(site_name, scenarios):
    """
    Reads in the global and regional sea level projections calculated from
    the CMIP projections (in metres). These data are relative to a baseline
    of 1981-2010.
    :param site_name: site location
    :param scenarios: emission scenarios
    :return: DataFrame of global and regional sea level projections
    """
    print('running function read_regional_sea_level_projections')

    # Read in the global and regional sea level projections
    in_slddir = read_dir()[4]
    districtname = f'loc_{site_name}'

    G_df_list = []
    R_df_list = []
    for sce in scenarios:
        G_filename = '{}{}_{}_projection_2100_global.csv'.format(in_slddir, districtname, sce)
        R_filename = '{}{}_{}_projection_2100_regional.csv'.format(in_slddir, districtname, sce)
        try:
            print(f'Trying to load global projection file: {G_filename}')
            G_df = pd.read_csv(G_filename, header=0, index_col=['year', 'percentile'])
            G_df.rename(columns={'exp': 'ocean', 'GIA': 'gia'}, inplace=True)
            
            print(f'Trying to load regional projection file: {R_filename}')
            R_df = pd.read_csv(R_filename, header=0, index_col=['year', 'percentile'])
            R_df.rename(columns={'exp': 'ocean', 'GIA': 'gia'}, inplace=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"File {e.filename} not found. Scenario: {sce} - scenario selected does not currently exist")

        G_df_list.append(G_df)
        R_df_list.append(R_df)

    return G_df_list, R_df_list


def read_IPCC_AR5_Levermann_proj(scenarios, refname='sum'):
    """
    Read in the IPCC AR5 + Levermann global sea level projections.
    :param scenarios: emission scenario
    :param refname: name of the component to plot e.g. "expansion",
    "antsmb", "sum"
    :return: lower, middle and upper time series of IPCC AR5 + Levermann
    global sea level projections
    """
    print('running function read_IPCC_AR5_Levermann_proj')

    # Directory of Monte Carlo time series for new projections
    mcdir = settings["montecarlodir"]

    ar5_low = []
    ar5_mid = []
    ar5_upp = []

    for sce in scenarios:
        reflow = iris.load_cube(
            os.path.join(mcdir, sce + '_' + refname + 'lower.nc')).data
        refmid = iris.load_cube(
            os.path.join(mcdir, sce + '_' + refname + 'mid.nc')).data
        refupp = iris.load_cube(
            os.path.join(mcdir, sce + '_' + refname + 'upper.nc')).data

        ar5_low.append(reflow)
        ar5_mid.append(refmid)
        ar5_upp.append(refupp)

    return ar5_low, ar5_mid, ar5_upp


def read_PSMSL_tide_gauge_obs(root_dir, data_source, data_type, region,
                              df_site, index, base_fdir):
    """
    Read annual mean sea level observations from PSMSL.
    :param root_dir: base directory
    :param data_source: data source of tide gauge data
    :param data_type: data type of tide gauge data
    :param region: user specified region (for folder structure)
    :param df_site: DataFrame of site metadata
    :param index: site location
    :param base_fdir: figure directory
    :return: annual mean sea level data and relevant metadata on missing /
    flagged data
    """
    # Determine Station ID - indicator for .rlr file
    latitude = df_site.loc[index, 'latitude']
    longitude = df_site.loc[index, 'longitude']
    station_id, tg_name = find_nearest_station_id(
        root_dir, data_source, data_type, region, latitude, longitude)
    
    file_path = os.path.join(root_dir, 'rlr_annual', 'data', f'{station_id}.rlrdata')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please check the file path and ensure the file exists.")

    baseline_years = [1981, 2010]
    min_valid_fraction = 0.5

    # Read in the annual mean sea levels, downloaded from PSMSL
    tg_years, tg_amsl, _, tg_flag_ex = \
        tgl.read_rlr_annual_mean_sea_level(station_id)
    non_missing = ~np.isnan(tg_amsl)

    # Find the years with 'flagged' data, flagged for attention '001' or MTL
    # in MSL time series '010'
    flag = (tg_flag_ex >= min_valid_fraction)

    nflag = sum(flag)

    # Calculate the baseline sea level and difference between the observed
    # period and the baseline.
    # Latter will be zero if sufficent observations lie within the baseline
    loc_abbrev = abbreviate_location_name(str(index))
    baseline_sl, _ = tgl.calc_baseline_sl(root_dir, region, loc_abbrev,
                                          tg_years, tg_amsl, baseline_years)

    # Adjust the tide gauge data to be sea levels relative to the baseline,
    # to match the regional projections
    tg_amsl = tg_amsl - baseline_sl

    # Convert the tide gauge sea levels from mm to metres.
    tg_amsl /= 1000.0

    return tg_name, nflag, flag, tg_years, non_missing, tg_amsl


def main():
    sealev_fdir = read_dir()[5]
    makefolder(sealev_fdir)
    df_site_data = pd.read_csv(r'D:\Disertasi\coba_banyak\UK\UK.csv', sep=';')
    df_site_data['latitude'] = pd.to_numeric(df_site_data['latitude'], errors='coerce')
    df_site_data['longitude'] = pd.to_numeric(df_site_data['longitude'], errors='coerce')
    df_site_data['latitude'] = df_site_data['latitude'].round(2)
    df_site_data['longitude'] = df_site_data['longitude'].round(2)
    df_site_data['Station ID'] = df_site_data.index  # Assuming index can be used as 'Station ID'
    df_site_data['model_name']=df_site_data['site_name']
    
    # Exclude indices 19-27 and 33-36
    indices_to_exclude = list(range(4, 6)) 
    df_site_data = df_site_data.drop(indices_to_exclude)
    
    print(f'Loaded site names and coordinates:\n{df_site_data}')
    rcp_scenarios = ['rcp26', 'rcp45', 'rcp85']
    
    for df_loc in df_site_data.index.values:
        g_df_list, r_df_list = read_G_R_sl_projections(df_loc, rcp_scenarios)
        ar5_low, ar5_mid, ar5_upp = read_IPCC_AR5_Levermann_proj(rcp_scenarios)
        tg_name, nflag, flag, tg_years, non_missing, tg_amsl = \
            read_PSMSL_tide_gauge_obs(settings["baseoutdir"], settings[
                "tidegaugeinfo"]["source"], settings["tidegaugeinfo"][
                "datafq"], settings["siteinfo"]["region"], df_site_data,
                                      df_loc, sealev_fdir)
        plot_figure_one(r_df_list, df_site_data.loc[df_loc, 'model_name'], df_loc, rcp_scenarios, sealev_fdir)
        plot_figure_two(r_df_list, df_site_data.loc[df_loc, 'model_name'], tg_name, nflag, flag, tg_years, non_missing, tg_amsl, df_loc, rcp_scenarios, sealev_fdir)
        plot_figure_three(g_df_list, df_site_data.loc[df_loc, 'model_name'], r_df_list, ar5_low, ar5_mid, ar5_upp, df_loc, rcp_scenarios, sealev_fdir)
        plot_figure_four(r_df_list, df_site_data.loc[df_loc, 'model_name'], df_loc, rcp_scenarios, sealev_fdir)
        plot_figure_five(g_df_list, r_df_list, df_site_data.loc[df_loc, 'model_name'], df_loc, rcp_scenarios, sealev_fdir)
        plot_figure_six(r_df_list, df_site_data.loc[df_loc, 'model_name'], ar5_low, ar5_mid, ar5_upp, df_loc, rcp_scenarios, sealev_fdir)
        plot_figure_seven(g_df_list, r_df_list, df_site_data.loc[df_loc, 'model_name'], df_loc, rcp_scenarios, sealev_fdir)

if __name__ == '__main__':
    main()


import winsound
winsound.Beep(1000, 1000)  # Beep at 1000 Hz for 500 ms

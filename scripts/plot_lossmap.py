import pandas as pd
import numpy as np
import xtrack as xt
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from generic_parser import EntryPointParameters, entrypoint
from warm_regions import FCC_EE_WARM_REGIONS
from beam_parameters import FCC_EE_BEAM_PARAMETERS


# Change mathtext font to Nimbus Roman
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Roman'
plt.rcParams['mathtext.it'] = 'Nimbus Roman'
plt.rcParams['mathtext.bf'] = 'Nimbus Roman:bold'


# Constants --------------------------------------------------------------------

FONTSIZE_AX = 14
FONTSIZE_LEG = FONTSIZE_AX - 4
FONTSIZE_TICKS = FONTSIZE_AX - 4

EV_TO_JOULE = 1.60218e-19


# Script arguments -------------------------------------------------------------

def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="machine",
        type=str,
        required=True,
        help="Which machine (fccee, skekb, lhc, ...)."
    )
    params.add_parameter(
        name="file",
        type=str,
        required=True,
        help="Path to the part.hdf file."
    )
    params.add_parameter(
        name="lifetime",
        type=float,
        required=False,
        default=None,
        help="Lifetime to plot the loss map in terms of power."
    )
    params.add_parameter(
        name="show_plot",
        type=str,
        required=False,
        default="True",
        help="Whether to show the loss map or not."
    )
    params.add_parameter(
        name="plot_file",
        type=str,
        required=False,
        help="File name to which save the loss map."
    )

    return params


# Entrypoint -------------------------------------------------------------------

@entrypoint(get_params(), strict=True)
def main(inp):
    lossmap_data = {'lossmap_scalar': None, 'lossmap_aper': None, 'lossmap_coll': None}
    for lm_key in lossmap_data.keys():
        lossmap_data[lm_key] = pd.read_hdf(inp.file, key=lm_key)

    plot_lossmap(lossmap_data, inp.machine, inp.lifetime, inp.show_plot, inp.plot_file)


def check_warm_loss(s, warm_regions):
    return np.any((warm_regions.T[0] < s) & (warm_regions.T[1] > s))


def plot_lossmap(lossmap_data, machine, lifetime=None, show_plot=True, plot_file=None):
    lms = lossmap_data['lossmap_scalar']
    lma = lossmap_data['lossmap_aper']
    lmc = lossmap_data['lossmap_coll']

    s_min = lms['s_min'][0]
    s_max = lms['s_max'][0]
    nbins = lms['nbins'][0]
    binwidth = lms['binwidth'][0]
    s_range = (s_min, s_max)

    # Collimator losses
    coll_start = lmc['coll_start']
    coll_end = lmc['coll_end']
    coll_values = lmc['coll_loss']

    coll_lengths = coll_end - coll_start
    
    # Normalization 'total'
    norm_val = sum(coll_values) + sum(lma['aper_loss'])

    coll_values /= (norm_val * coll_lengths)

    # There can be an alternative way of plotting using a bar plot
    # Make the correct edges to get the correct width of step plot
    # The dstack and flatten merges the arrays one set of values at a time
    zeros = np.full_like(lmc.index, 0)  # Zeros to pad the bars
    coll_edges = np.dstack(
        [coll_start, coll_start, coll_end, coll_end]).flatten()
    coll_loss = np.dstack([zeros, coll_values, coll_values, zeros]).flatten()

    # Aperture losses
    aper_edges = np.linspace(s_min, s_max, nbins)
    aper_loss = lma['aper_loss'].reindex(range(0, nbins-1), fill_value=0)
    aper_loss /= (norm_val * binwidth)

    # Check if the start of the bin is in a warm region
    if machine == 'fccee':
        warm_regions = FCC_EE_WARM_REGIONS
    else:
        raise ValueError(f"Machine {machine} not supported.")
    
    mask_warm = np.array([check_warm_loss(s, warm_regions)
                        for s in aper_edges[:-1]])

    warm_loss = aper_loss * mask_warm
    cold_loss = aper_loss * ~mask_warm

    # The zorder determines the plotting order = collimator -> warm-> cold (on top)
    # The edge lines on the plots provide the dynamic scaling feature of the plots
    # e.g a 10 cm aperture loss spike is still resolvable for a full ring view
    lw = 1

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.stairs(warm_loss, aper_edges, color='r',
            lw=lw, ec='r', fill=True, zorder=20)
    ax.stairs(cold_loss, aper_edges, color='b',
            lw=lw, ec='b', fill=True, zorder=30)

    ax.fill_between(coll_edges, coll_loss, step='pre', color='k', zorder=9)
    ax.step(coll_edges, coll_loss, color='k', lw=lw, zorder=10)

    plot_margin = 500
    ax.set_xlim(s_range[0] - plot_margin, s_range[1] + plot_margin)
    ax.set_ylim(1e-6, 2)

    ax.yaxis.grid(visible=True, which='major', zorder=0, alpha=0.5)

    ax.set_yscale('log', nonpositive='clip')

    ax.set_xlabel(r'$\mathrm{s}~[\mathrm{m}]$', fontsize=FONTSIZE_AX)
    ax.set_ylabel(r'$\eta~[\mathrm{m}^{-1}]$', fontsize=FONTSIZE_AX)

    # Fake plot for a nice looking legend
    ax.plot([], [], c='k', lw=3, label='Collimator')
    ax.plot([], [], c='b', lw=3, label='Cold')
    ax.plot([], [], c='r', lw=3, label='Warm')

    # Load the Nimbus Roman font and set it as the legend font
    prop = font_manager.FontProperties(fname='/usr/share/fonts/type1/gsfonts/n021003l.pfb', size=FONTSIZE_LEG)
    ax.legend(prop=prop, labelspacing=0.2, handlelength=1.5, framealpha=0.65, borderpad=0.3)

    # Set the size of the xticks and yticks label
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICKS)

    # Add labels for the IPs to the top of the plot
    if machine == 'fccee':
        y_text = 8
        ax.text(s_min, y_text, 'IPA', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*1, y_text, 'PB', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*2, y_text, 'IPD', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*3, y_text, 'PF', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*4, y_text, 'IPG', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*5, y_text, 'PH', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*6, y_text, 'IPJ', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*7, y_text, 'PL', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text(s_max, y_text, 'IPA', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
    else:
        raise ValueError(f"Machine {machine} not supported.")
    
    fig.tight_layout()

    if plot_file is not None:
        plt.savefig(f'{plot_file}.png', dpi=300)

    if show_plot:  
        plt.show()

    if lifetime is not None:
        plot_lossmap_power(lossmap_data, machine, lifetime, show_plot, plot_file)


def plot_lossmap_power(lossmap_data, machine, lifetime=None, show_plot=True, plot_file=None):
    lms = lossmap_data['lossmap_scalar']
    lma = lossmap_data['lossmap_aper']
    lmc = lossmap_data['lossmap_coll']

    s_min = lms['s_min'][0]
    s_max = lms['s_max'][0]
    nbins = lms['nbins'][0]
    binwidth = lms['binwidth'][0]
    s_range = (s_min, s_max)

    # Collimator losses
    coll_start = lmc['coll_start']
    coll_end = lmc['coll_end']
    coll_values = lmc['coll_loss']

    coll_lengths = coll_end - coll_start
    
    if machine == 'fccee':
        beam_params = FCC_EE_BEAM_PARAMETERS
    else: 
        raise ValueError(f"Machine {machine} not supported.")
    
    total_beam_energy = beam_params['NB'] * beam_params['KB'] * beam_params['BEAM_ENERGY']
    total_sim_energy = lms['n_primaries'][0] * beam_params['BEAM_ENERGY']
    total_sim_lost_energy = sum(lmc['coll_loss']) + sum(lma['aper_loss'])
    lost_energy_fraction = total_sim_lost_energy / total_sim_energy
    total_lost_energy_joule = total_beam_energy * lost_energy_fraction * EV_TO_JOULE
    
    norm_val = total_sim_energy / total_lost_energy_joule

    coll_values /= (norm_val * lifetime)

    # There can be an alternative way of plotting using a bar plot
    # Make the correct edges to get the correct width of step plot
    # The dstack and flatten merges the arrays one set of values at a time
    zeros = np.full_like(lmc.index, 0)  # Zeros to pad the bars
    coll_edges = np.dstack(
        [coll_start, coll_start, coll_end, coll_end]).flatten()
    coll_loss = np.dstack([zeros, coll_values, coll_values, zeros]).flatten()

    # Aperture losses
    aper_edges = np.linspace(s_min, s_max, nbins)
    aper_loss = lma['aper_loss'].reindex(range(0, nbins-1), fill_value=0)
    aper_loss /= (norm_val * lifetime)

    # Check if the start of the bin is in a warm region
    if machine == 'fccee':
        warm_regions = FCC_EE_WARM_REGIONS
    else:
        raise ValueError(f"Machine {machine} not supported.")
    
    mask_warm = np.array([check_warm_loss(s, warm_regions)
                        for s in aper_edges[:-1]])

    warm_loss = aper_loss * mask_warm
    cold_loss = aper_loss * ~mask_warm

    # The zorder determines the plotting order = collimator -> warm-> cold (on top)
    # The edge lines on the plots provide the dynamic scaling feature of the plots
    # e.g a 10 cm aperture loss spike is still resolvable for a full ring view
    lw = 1

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.stairs(warm_loss, aper_edges, color='r',
            lw=lw, ec='r', fill=True, zorder=20)
    ax.stairs(cold_loss, aper_edges, color='b',
            lw=lw, ec='b', fill=True, zorder=30)

    ax.fill_between(coll_edges, coll_loss, step='pre', color='k', zorder=9)
    ax.step(coll_edges, coll_loss, color='k', lw=lw, zorder=10)

    plot_margin = 500
    ax.set_xlim(s_range[0] - plot_margin, s_range[1] + plot_margin)
    ax.set_ylim(1e-2, 1e5)

    ax.yaxis.grid(visible=True, which='major', zorder=0, alpha=0.5)

    ax.set_yscale('log', nonpositive='clip')

    ax.set_xlabel(r'$\mathrm{s}~[\mathrm{m}]$', fontsize=FONTSIZE_AX)
    ax.set_ylabel(r'$\mathrm{P}~[\mathrm{W}]$', fontsize=FONTSIZE_AX)

    # Fake plot for a nice looking legend
    ax.plot([], [], c='k', lw=3, label='Collimator')
    ax.plot([], [], c='b', lw=3, label='Cold')
    ax.plot([], [], c='r', lw=3, label='Warm')

    # Load the Nimbus Roman font and set it as the legend font
    prop = font_manager.FontProperties(fname='/usr/share/fonts/type1/gsfonts/n021003l.pfb', size=FONTSIZE_LEG)
    ax.legend(prop=prop, labelspacing=0.2, handlelength=1.5, framealpha=0.65, borderpad=0.3)

    # Set the size of the xticks and yticks label
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICKS)

    # Add labels for the IPs to the top of the plot
    if machine == 'fccee':
        y_text = 5e5
        ax.text(s_min, y_text, 'IPA', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*1, y_text, 'PB', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*2, y_text, 'IPD', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*3, y_text, 'PF', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*4, y_text, 'IPG', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*5, y_text, 'PH', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*6, y_text, 'IPJ', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text((s_max/8)*7, y_text, 'PL', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
        ax.text(s_max, y_text, 'IPA', fontsize=FONTSIZE_TICKS, fontname='Nimbus Roman', ha='center')
    else:
        raise ValueError(f"Machine {machine} not supported.")
    
    fig.tight_layout()

    if plot_file is not None:
        plt.savefig(f'{plot_file}_power.png', dpi=300)

    if show_plot:  
        plt.show()


# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    main()



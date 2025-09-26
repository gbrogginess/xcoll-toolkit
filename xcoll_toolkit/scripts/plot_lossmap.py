import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from generic_parser import EntryPointParameters, entrypoint

######################################################
# Change mathtext font to Nimbus Roman
######################################################
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Roman'
plt.rcParams['mathtext.it'] = 'Nimbus Roman'
plt.rcParams['mathtext.bf'] = 'Nimbus Roman:bold'

######################################################
# TODO: add warm regions for different machines
######################################################

######################################################
# Script arguments
######################################################
def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="file",
        type=str,
        required=True,
        help="Path to the particles file.",
    )
    params.add_parameter(
        name="figsize",
        type=float,
        nargs=2,
        help="x-y size of the figure.",
    )
    params.add_parameter(
        name="xlim",
        type=float,
        nargs=2,
        help="Limits on the x-axis.",
    )
    params.add_parameter(
        name="ylim",
        type=float,
        nargs=2,
        help="Limits on the y-axis.",
    )
    params.add_parameter(
        name="fs_ax",
        type=float,
        help=".",
    )
    params.add_parameter(
        name="fs_leg",
        type=float,
        help=".",
    )
    params.add_parameter(
        name="fs_ticks",
        type=float,
        help=".",
    )
# add xticks
# add yticks
    return params

######################################################
# Entrypoint
######################################################
@entrypoint(get_params(), strict=True)
def main(inp):
    lms = pd.read_hdf(inp.file, key='lossmap_scalar')
    lma = pd.read_hdf(inp.file, key='lossmap_aper')
    lmc = pd.read_hdf(inp.file, key='lossmap_coll')

    if inp.figsize is not None:
        figsize = (inp.figsize[0], inp.figsize[1])
    else:
        figsize = (12, 4)
    if inp.xlim is not None:
        xlim = (inp.xlim[0], inp.xlim[1])
    else:
        xlim = None
    if inp.ylim is not None:
        ylim = (inp.ylim[0], inp.ylim[1])
    else:
        ylim = None
    if inp.fs_ax is not None:
        fs_ax = inp.fs_ax
    else:
        fs_ax = 12
    if inp.fs_leg is not None:
        fs_leg = inp.fs_leg
    else:
        fs_leg = 10
    if inp.fs_ticks is not None:
        fs_ticks = inp.fs_ticks
    else:
        fs_ticks = 10

    lm_data = {
        'lossmap_scalar': lms,
        'lossmap_aper': lma,
        'lossmap_coll': lmc,
    }

    return plot_lossmap(lm_data, figsize=figsize, xlim=xlim, ylim=ylim, fontsize_ax=fs_ax,
                        fontsize_leg=fs_leg, fontsize_ticks=fs_ticks)

######################################################
# Auxiliary functions
######################################################
def check_warm_loss(s, warm_regions):
    return np.any((warm_regions.T[0] < s) & (warm_regions.T[1] > s))

def check_cold_loss(s, cold_regions):
    return np.any((cold_regions.T[0] >= s) & (cold_regions.T[1] <= s))

######################################################
# Plotting function
######################################################
def plot_lossmap(lossmap_data, figsize,
                 xlim=None, ylim=None,
                 fontsize_ax=12, fontsize_leg=10, fontsize_ticks=10):
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

    # coll_values /= (norm_val * coll_lengths)

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

    # aper_loss /= (norm_val * binwidth)

    # warm_regions = SKEKB_WARM_REGIONS
    # mask_warm = np.array([check_warm_loss(s, warm_regions)
    #                     for s in aper_edges[:-1]])
    # mask_cold = np.zeros_like(aper_edges, dtype=bool)
    # mask_cold = mask_cold[:-1]
    # for region in SKEKB_COLD_REGIONS:
    #     tolerance = 1e-3
    #     start, end = region
    #     mask_cold = mask_cold | ((aper_edges[:-1] >= start-tolerance) & (aper_edges[:-1] <= end+tolerance))

    # warm_loss = aper_loss * ~mask_cold
    # cold_loss = aper_loss * mask_cold

    warm_loss = aper_loss

    # The zorder determines the plotting order = collimator -> warm-> cold (on top)
    # The edge lines on the plots provide the dynamic scaling feature of the plots
    # e.g a 10 cm aperture loss spike is still resolvable for a full ring view
    lw = 1

    fig, ax = plt.subplots(figsize=figsize)
    ax.stairs(warm_loss, aper_edges, color='r',
            lw=lw, ec='r', fill=True, zorder=20)
    # ax.stairs(cold_loss, aper_edges, color='b',
    #         lw=lw, ec='b', fill=True, zorder=30)

    ax.fill_between(coll_edges, coll_loss, step='pre', color='k', zorder=9)
    ax.step(coll_edges, coll_loss, color='k', lw=lw, zorder=10)

    if xlim is None:
        xmargin = 10
        ax.set_xlim(s_range[0] - xmargin, s_range[1] + xmargin)
    else:
        ax.set_xlim(xlim)

    if ylim is None:
        ax.set_ylim(1e-6, 5e1)
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1,])
    else:
        ax.set_ylim(ylim)

    ax.yaxis.grid(visible=True, which='major', zorder=0, alpha=0.5)

    ax.set_yscale('log', nonpositive='clip')

    ax.set_xlabel(r'$\mathrm{s}~[\mathrm{m}]$', fontsize=fontsize_ax)
    ax.set_ylabel(r'$\eta~[\mathrm{m}^{-1}]$', fontsize=fontsize_ax)

    # Fake plot for a nice looking legend
    ax.plot([], [], c='k', lw=3, label='Collimator')
    # ax.plot([], [], c='b', lw=3, label='Cold')
    ax.plot([], [], c='r', lw=3, label='Warm')

    # Load the Nimbus Roman font and set it as the legend font
    prop = font_manager.FontProperties(fname='/usr/share/fonts/type1/gsfonts/n021003l.pfb', size=fontsize_leg)
    ax.legend(prop=prop, labelspacing=0.2, handlelength=1.5, framealpha=0.65, borderpad=0.3)

    # Increase the size of the xticks and yticks label
    ax.tick_params(axis='both', labelsize=fontsize_ticks)

    # # Add labels for the IPs to the top of the plot
    # ax.text(s_min, 60, 'Tsukuba', fontsize=fontsize_ax, fontname='Nimbus Roman', ha='center')
    # ax.text((s_max/8)*2, 60, 'Nikko', fontsize=fontsize_ax, fontname='Nimbus Roman', ha='center')
    # ax.text((s_max/8)*4, 60, 'Fuji', fontsize=fontsize_ax, fontname='Nimbus Roman', ha='center')
    # ax.text((s_max/8)*6, 60, 'Oho (NLC)', fontsize=fontsize_ax, fontname='Nimbus Roman', ha='center')
    # ax.text(s_max, 60, 'Tsukuba', fontsize=fontsize_ax, fontname='Nimbus Roman', ha='center')

    fig.tight_layout()
    # plt.savefig('Output-eBrem/skekb_eBrem_lossmap_full.png', dpi=300)
    plt.show()

##############################################################
# Script mode
##############################################################
if __name__ == "__main__":
    main()

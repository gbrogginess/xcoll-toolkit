"""
Module to prepare "collimasim" loss maps from collimation simulation outputs.
=============================================
Author(s): Giacomo Broggi, Andrey Abramov
Email:  giacomo.broggi@cern.ch
Date:   12-03-2025
"""
import numpy as np
import pandas as pd
import xpart as xp
import xcoll as xc
from warnings import warn


def prepare_lossmap(particles, line, s0, binwidth, weights):
    lossmap_weights = ['none', 'energy']
    if weights not in lossmap_weights:
        raise ValueError('weights must be in [{}]'.format(', '.join(lossmap_weights)))

    s_ele = np.array(line.get_s_elements(mode='downstream'))
    max_s = max(s_ele)
    s_range = (0, max_s)  # Get the end point as assume the start point is zero

    coll_idx = []
    for idx, elem in enumerate(line.elements):
        if isinstance(elem, xc.Geant4Collimator):
            coll_idx.append(idx)
    coll_idx = np.array(coll_idx)
    coll_names = np.take(line.element_names, coll_idx)

    # Ignore unallocated array slots
    mask_allocated = particles.state > -9999
    particles = particles.filter(mask_allocated)

    # Count the number of primary particles for information
    mask_prim = particles.parent_particle_id == particles.particle_id
    n_prim = len(particles.filter(mask_prim).x)

    mask_lost = particles.state <= 0

    # If no losses return empty loss maps, but preserve structures
    if not np.any(mask_lost):
        warn('No losses found, loss map will be empty')
        particles = xp.Particles(x=[], **{kk[1]:getattr(particles,kk[1]) 
                                          for kk in particles.scalar_vars})
    else:
        particles = particles.filter(mask_lost) #(mask_part_type & mask_lost)

    # Get a mask for the collimator losses
    mask_losses_coll = np.isin(particles.at_element, coll_idx)

    if weights == 'energy':
        part_mass_ratio = particles.charge_ratio / particles.chi
        part_mom = (particles.delta + 1) * particles.p0c
        part_mass = part_mass_ratio * particles.mass0
        part_tot_energy = np.sqrt(part_mom**2 + part_mass**2)
        histo_weights = part_tot_energy
    elif weights == 'none':
        histo_weights = np.full_like(particles.x, 1)
    else:
        raise ValueError('weights must be in [{}]'.format(', '.join(lossmap_weights)))

    # Collimator losses binned per element
    h_coll, edges_coll = np.histogram(particles.at_element[mask_losses_coll],
                                      bins=range(max(coll_idx)+2),
                                      weights=histo_weights[mask_losses_coll])

    # Process the collimator per element histogram for plotting
    coll_lengths = np.array([line.elements[ci].length for ci in coll_idx])
    # reduce the empty bars in the histogram
    coll_values = np.take(h_coll, coll_idx)

    coll_end = np.take(s_ele, coll_idx)
    coll_start = coll_end - coll_lengths

    # Aperture losses binned in S
    nbins_ap = int(np.ceil((s_range[1] - s_range[0])/binwidth))
    bins_ap = np.linspace(s_range[0], s_range[1], nbins_ap)

    aper_loss, _ = np.histogram(particles.s[~mask_losses_coll],
                                bins=bins_ap,
                                weights=histo_weights[~mask_losses_coll])

    # Prepare structures for optimal storage
    aper_loss_series = pd.Series(aper_loss)

    # Scalar variables go in their own DF to avoid replication
    # The bin edges can be re-generated with linspace, no need to store
    scalar_dict = {
        'binwidth': binwidth,
        'weights': weights,
        'nbins': nbins_ap,
        's_min': s_range[0],
        's_max': s_range[1],
        'n_primaries': n_prim,
    }

    # Drop the zeros while preserving the index
    aperloss_dict = {
        'aper_loss': aper_loss_series[aper_loss_series > 0],
    }

    coll_dict = {
        'coll_name': coll_names,
        'coll_element_index': coll_idx,
        'coll_start': coll_start,
        'coll_end': coll_end,
        'coll_loss': coll_values
    }

    scalar_df = pd.DataFrame(scalar_dict, index=[0])
    coll_df = pd.DataFrame(coll_dict)
    aper_df = pd.DataFrame(aperloss_dict)

    lm_dict = {'lossmap_scalar': scalar_df,
               'lossmap_aper': aper_df,
               'lossmap_coll': coll_df}

    return lm_dict
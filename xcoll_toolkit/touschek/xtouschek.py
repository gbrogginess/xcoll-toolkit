"""
Python module for Monte Carlo simulation of Touschek effect in Xsuite.
=============================================
Author(s): Giacomo Broggi
Email:  giacomo.broggi@cern.ch
Date:   13-03-2025
"""
# ===========================================
# 🔹 Required modules
# ===========================================
import xtrack as xt
import xcoll as xc
import numpy as np
import re
from scipy.integrate import quad
from scipy.special import i0
from scipy.constants import physical_constants

# ===========================================
# 🔹 Constants
# ===========================================
ELECTRON_MASS_EV = xt.ELECTRON_MASS_EV
C_LIGHT_VACUUM = physical_constants['speed of light in vacuum'][0]
CLASSICAL_ELECTRON_RADIUS = physical_constants['classical electron radius'][0]


class TouschekCalculator():
    def __init__(self, manager):
        # TODO: Should manager be a class attribute instead of an instance attribute?
        self.manager = manager
        self.p0c = self.manager.ref_particle.p0c[0]
        self.npart_over_two = int(self.manager.n_part_mc / 2)
        self.local_momentum_aperture = None
        self.PP1 = None
        self.mask_PP1 = None
        self.PP2 = None
        self.mask_PP2 = None
        self.element = None
        self.twiss = None
        self.theta_cm = None
        self.gamma_cm = None
        self.beta0 = None
        self.mask_particles = None
        self.integrated_piwinski_total_scattering_rates = {}

    def _get_fourmomenta_matrix(self, PP):
        pz = np.sqrt((1 + PP[:, 5])**2 - PP[:, 1]**2 - PP[:, 3]**2)
        VV = self.p0c * np.column_stack((PP[:, 1], PP[:, 3], pz))
        p_abs = np.linalg.norm(VV, axis=1)
        e = np.sqrt(p_abs**2 + ELECTRON_MASS_EV**2)
        VV = np.column_stack((VV, e))

        return VV
    
    def _get_boost_matrix(self, VV):
        PP = VV[:, :3]
        EE = VV[:, 3].reshape(-1, 1)
        BB = PP / EE

        return BB

    def _lab_to_cm(self, VV, BB):
        # Assuming BB is a matrix where each row is a bst vector [bx, by, bz]
        # and VV is a matrix where each row corresponds to an object with columns [px, py, pz, e]

        # Extracting components from BB and VV
        bx, by, bz = BB[:, 0], BB[:, 1], BB[:, 2]
        px, py, pz, e = VV[:, 0], VV[:, 1], VV[:, 2], VV[:, 3]
        # Compute b2 for each row in BB
        b2 = np.sum(BB**2, axis=1)
        # Compute gamma for each row in BB
        gamma = 1 / np.sqrt(1 - b2)
        # Compute bp for each row in BB and corresponding row in VV
        bp = bx * px + by * py + bz * pz

        factor = (gamma - 1) / b2

        # Compute qx, qy, qz for each row
        qx = px + factor * bp * bx - gamma * e * bx
        qy = py + factor * bp * by - gamma * e * by
        qz = pz + factor * bp * bz - gamma * e * bz
        # Compute q2 for each row
        q2 = qx**2 + qy**2 + qz**2
        # Compute e for each row
        e_new = np.sqrt(q2 + ELECTRON_MASS_EV**2)

        QQ = np.column_stack((qx, qy, qz, e_new))

        return QQ
    

    def _rotate(self, QQ, theta, phi):
        px, py, pz = QQ[:, 0], QQ[:, 1], QQ[:, 2]
        p_abs = np.sqrt(px**2 + py**2 + pz**2)
        # Compute spherical angles
        th = np.arccos(pz / p_abs)
        ph = np.arctan2(py, px)
        # Trigonometric terms
        s1 = np.sin(th)
        s2 = np.sin(ph)
        c1 = np.cos(th)
        c2 = np.cos(ph)
        # Direction cosines of new axis
        x0 = np.cos(theta)
        y0 = np.sin(theta) * np.cos(phi)
        z0 = np.sin(theta) * np.sin(phi)

        # Transformed vector components
        QQ[:, 0] = p_abs * (s1 * c2 * x0 - s2 * y0 - c1 * c2 * z0)
        QQ[:, 1] = p_abs * (s1 * s2 * x0 + c2 * y0 - c1 * s2 * z0)
        QQ[:, 2] = p_abs * (c1 * x0 + s1 * z0)

        return QQ
    

    def _cm_to_lab(self, QQ, BB):
        # Assuming BB is a matrix where each row is a bst vector [bx, by, bz]
        # and QQ is a matrix where each row corresponds to an object with columns [px, py, pz, e]

        # Extracting components from BB and QQ
        bx, by, bz = BB[:, 0], BB[:, 1], BB[:, 2]
        qx, qy, qz, e = QQ[:, 0], QQ[:, 1], QQ[:, 2], QQ[:, 3]
        # Compute b2 for each row in BB
        b2 = np.sum(BB**2, axis=1)
        # Compute gamma for each row in BB
        gamma = 1 / np.sqrt(1 - b2)
        # Compute bq for each row in BB and corresponding row in QQ
        bq = bx * qx + by * qy + bz * qz

        factor = (gamma - 1) / b2

        px1 = qx + gamma * bx * e + factor * bq * bx
        py1 = qy + gamma * by * e + factor * bq * by
        pz1 = qz + gamma * bz * e + factor * bq * bz
        e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + ELECTRON_MASS_EV**2)

        px2 = -qx + gamma * bx * e - factor * bq * bx
        py2 = -qy + gamma * by * e - factor * bq * by
        pz2 = -qz + gamma * bz * e - factor * bq * bz
        e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + ELECTRON_MASS_EV**2)

        VV1 = np.column_stack((px1, py1, pz1, e1))
        VV2 = np.column_stack((px2, py2, pz2, e2))

        return VV1, VV2 
    

    def _compute_moller_shape(self, theta_cm):
        st2   = np.sin(theta_cm)**2
        b2    = self.beta0**2

        return (1.0 - b2) * ( ((1.0 + 1.0/b2)**2) * (4.0/st2**2 - 3.0/st2) + 1.0 + 4.0/st2 )


    def scatter(self, PP1, PP2):
        p0c = self.p0c

        # Compute 4-momenta for the scattering particles
        VV1 = self._get_fourmomenta_matrix(PP1)
        VV2 = self._get_fourmomenta_matrix(PP2)

        # Boost to the cm frame
        VV = VV1 + VV2
        BB = self._get_boost_matrix(VV)
        b2 = np.sum(BB**2, axis=1)

        # Compute gamma of the cm frame w.r.t. the lab frame
        gamma_cm = 1.0 / np.sqrt(1.0 - b2)
        self.gamma_cm = gamma_cm

        # Transform to the cm frame
        QQ = self._lab_to_cm(VV1, BB)

        # Sample theta and phi in the cm frame
        # Avoid sampling exactly 0 or π to prevent singularities 
        # in the Møller differential cross section (which diverges at θ = 0 or π)
        # The chosen range for θ (from ELEGANT) should maintain physical accuracy while improving numerical stability
        self.theta_cm = (np.random.uniform(0, 1, self.npart_over_two) * 0.9999 + 0.00005) * np.pi
        phi_cm        = np.random.uniform(0, np.pi, self.npart_over_two)

        # Apply the scattering angle
        QQ = self._rotate(QQ, self.theta_cm, phi_cm)

        # beta0 = |q'| / E* (after scattering)
        qabs   = np.linalg.norm(QQ[:, :3], axis=1)
        E_star = QQ[:, 3]
        beta0  = qabs / E_star
        self.beta0 = beta0
        # Compute Møller DCS
        self.moller_dcs = CLASSICAL_ELECTRON_RADIUS**2 / 4 * self._compute_moller_shape(self.theta_cm)

        # Transform back to the lab frame
        VV1, VV2 = self._cm_to_lab(QQ, BB)

        PP1[:, 1] = VV1[:, 0] / p0c
        PP1[:, 3] = VV1[:, 1] / p0c
        PP1[:, 5] = (np.sqrt(VV1[:, 0]**2 + VV1[:, 1]**2 + VV1[:, 2]**2) - p0c) / p0c

        PP2[:, 1] = VV2[:, 0] / p0c
        PP2[:, 3] = VV2[:, 1] / p0c
        PP2[:, 5] = (np.sqrt(VV2[:, 0]**2 + VV2[:, 1]**2 + VV2[:, 2]**2) - p0c) / p0c

        self.PP1 = PP1
        self.PP2 = PP2
        return PP1, PP2


    def _compute_local_scattering_rate(self, phase_space_volume, dens1, dens2):
        n_events = self.manager.n_part_mc // 2
        v0_star  = self.beta0 * C_LIGHT_VACUUM # Relative velocity in the cm frame after scattering
        inv_gcm2 = self.gamma_cm**-2 
        w        = phase_space_volume / n_events
        s        = np.sin(self.theta_cm)

        local_scattering_rate1 = w * v0_star * inv_gcm2 * self.moller_dcs * s * dens1 * dens2
        local_scattering_rate2 = w * v0_star * inv_gcm2 * self.moller_dcs * s * dens1 * dens2

        return local_scattering_rate1, local_scattering_rate2


    def _compute_piwinski_integral(self, tm, b1, b2):
        km = np.arctan(np.sqrt(tm))

        def int_piwinski(k, km, B1, B2):
            """
            Integrand of the piwinski formula
            In case the Bessel function has too large value
            (more than :math:`10^251`) it
            is substituted by its exponential approximation:
            :math:`I_0(x)~\frac{\exp(x)}{\sqrt{2 \pi x}}`
            """
            t = np.tan(k) ** 2
            tm = np.tan(km) ** 2
            fact = (
                (2 * t + 1) ** 2 * (t / tm / (1 + t) - 1) / t
                + t
                - np.sqrt(t * tm * (1 + t))
                - (2 + 1 / (2 * t)) * np.log(t / tm / (1 + t))
            )
            if B2 * t < 500:
                intp = fact * np.exp(-B1 * t) * i0(B2 * t) * np.sqrt(1 + t)
            else:
                intp = (
                    fact
                    * np.exp(B2 * t - B1 * t)
                    / np.sqrt(2 * np.pi * B2 * t)
                    * np.sqrt(1 + t)
                )
            return intp

        args = (km, b1, b2)
        val, _ =  quad(
            int_piwinski,
            km,
            np.pi / 2,
            args=args,
            epsabs=1e-16,
            epsrel=1e-12
        )

        return val


    def _compute_piwinski_total_scattering_rate(self, element):
        # TODO: if element is thick, use the average optical functions
        p0c = self.manager.ref_particle.p0c[0]
        beta0 = self.manager.ref_particle.beta0[0]
        gamma0 = self.manager.ref_particle.gamma0[0]
        kb = self.manager.kb
        fdelta = self.manager.fdelta
        local_momentum_aperture = self.local_momentum_aperture
        gemitt_x = self.manager.nemitt_x / beta0 / gamma0
        alfx = self.twiss['alfx', element]
        betx = self.twiss['betx', element]
        gemitt_y = self.manager.nemitt_y / beta0 / gamma0
        alfy = self.twiss['alfy', element]
        bety = self.twiss['bety', element]
        sigma_z = self.manager.sigma_z
        sigma_delta = self.manager.sigma_delta
        delta = self.twiss['delta', element]
        dx = self.twiss['dx', element]
        dpx = self.twiss['dpx', element]
        dxt = alfx * dx + betx * dpx # dxt: dx tilde
        dy = self.twiss['dy', element]
        dpy = self.twiss['dpy', element]
        dyt = alfy * dy + bety * dpy # dyt: dy tilde

        deltaN = local_momentum_aperture[element][0] * fdelta
        deltaP = local_momentum_aperture[element][1] * fdelta

        sigmab_x = np.sqrt(gemitt_x * betx) # Horizontal betatron beam size
        sigma_x = np.sqrt(gemitt_x * betx + dx**2 * sigma_delta**2) # Horizontal beam size

        sigmab_y = np.sqrt(gemitt_y * bety) # Vertical betatron beam size
        sigma_y = np.sqrt(gemitt_y * bety + dy**2 * sigma_delta**2) # Vertical beam size

        sigma_h = (sigma_delta**-2 + (dx**2 + dxt**2)/sigmab_x**2 + (dy**2 + dyt**2)/sigmab_y**2)**(-0.5)

        p = p0c * (1 + delta)
        gamma = np.sqrt(1 + p**2 / ELECTRON_MASS_EV**2)
        beta = np.sqrt(1 - gamma**-2)

        b1 = betx**2 / (2 * beta**2 * gamma**2 * sigmab_x**2) * (1 - sigma_h**2 * dxt**2 / sigmab_x**2) \
             + bety**2 / (2 * beta**2 * gamma**2 * sigmab_y**2) * (1 - sigma_h**2 * dyt**2 / sigmab_y**2)
        
        b2 = np.sqrt(b1**2 - betx**2 * bety**2 * sigma_h**2 / (beta**4 * gamma**4 * sigmab_x**4 * sigmab_y**4 * sigma_delta**2) \
                             * (sigma_x**2 * sigma_y**2 - sigma_delta**4 * dx**2 * dy**2))

        tmN = beta**2 * (abs(deltaN))**2 
        tmP = beta**2 * (deltaP)**2

        piwinski_integralN = self._compute_piwinski_integral(tmN, b1, b2)
        piwinski_integralP = self._compute_piwinski_integral(tmP, b1, b2)

        piwinski_integral = piwinski_integralN + piwinski_integralP

        rate = CLASSICAL_ELECTRON_RADIUS**2 * C_LIGHT_VACUUM * betx * bety * sigma_h * kb**2 \
                        / (8*np.sqrt(np.pi) * beta**2 * gamma**4 * sigmab_x**2 * sigmab_y**2 * sigma_z * sigma_delta) \
                        * piwinski_integral

        return rate


    def _assign_piwinski_total_scattering_rates(self):
        line = self.manager.line
        T_rev0 = self.twiss.T_rev0
        tab = line.get_table()

        # Indices of Touschek markers in the line
        ii_tmarker = []
        for ii, nn in enumerate(line.element_names):
            if nn.startswith('TMarker_'):
                ii_tmarker.append(ii)

        ii_current_tmarker = 0
        integrated_rate = 0.0  # ∫ r(s) ds over the current interval (between two TMarkers)

        # Tail pre-pass (wrap-around): integrate from last TMarker to end of line
        # This ensures the first stored value corresponds to the interval (last -> first).
        last_tm_idx = ii_tmarker[-1]
        last_tm_name = line.element_names[last_tm_idx]
        s_last_tm = tab.rows[tab.name == last_tm_name].s[0]

        # Left endpoint at the last TMarker itself
        rate_before = self._compute_piwinski_total_scattering_rate(last_tm_name)
        s_before = s_last_tm

        # Walk elements *after* the last TMarker up to the end of the line
        for jj in range(last_tm_idx + 1, len(line.elements)):
            nn_tail = line.element_names[jj]
            s = tab.rows[tab.name == nn_tail].s[0]
            if s > s_before:
                ds = s - s_before
                rate = self._compute_piwinski_total_scattering_rate(nn_tail)
                integrated_rate += 0.5 * (rate_before + rate) * ds
                s_before = s
                rate_before = rate

        # Main pass from s=0 up to each TMarker
        s_before = 0.0

        for (ii, ee), nn in zip(enumerate(line.elements), line.element_names):
            s = tab.rows[tab.name == nn].s[0]

            if ii < ii_tmarker[ii_current_tmarker]:
                if s > s_before:
                    ds = s - s_before
                    rate = self._compute_piwinski_total_scattering_rate(nn)
                    integrated_rate += 0.5 * (rate_before + rate) * ds
                    s_before = s
                    rate_before = rate

            elif ii == ii_tmarker[ii_current_tmarker]:
                if s > s_before:
                    ds = s - s_before
                    rate = self._compute_piwinski_total_scattering_rate(nn)
                    integrated_rate += 0.5 * (rate_before + rate) * ds
                    s_before = s
                    rate_before = rate

                # Store ∫ r(s) ds between the previous TMarker and this one
                self.integrated_piwinski_total_scattering_rates[f'TMarker_{ii_current_tmarker}'] = (
                    integrated_rate / C_LIGHT_VACUUM / T_rev0
                )

                # Reset accumulator for the next interval
                integrated_rate = 0.0
                ii_current_tmarker += 1
                if ii_current_tmarker >= len(ii_tmarker):
                    break


    def compute_total_scattering_rate(self, phase_space_volume, dens1, dens2):
        fdelta = self.manager.fdelta
        deltaN = self.local_momentum_aperture[self.element][0] * fdelta
        deltaP = self.local_momentum_aperture[self.element][1] * fdelta
        mask_PP1 = (self.PP1[:, 5] > deltaP) | (self.PP1[:, 5] < deltaN)
        mask_PP2 = (self.PP2[:, 5] > deltaP) | (self.PP2[:, 5] < deltaN)
        self.mask_PP1 = mask_PP1
        self.mask_PP2 = mask_PP2

        local_scattering_rate1, local_scattering_rate2 = self._compute_local_scattering_rate(phase_space_volume,
                                                                                             dens1, dens2)
        
        # print(f'\nM = {sum(mask_PP1) + sum(mask_PP2)}\n')
        total_scattering_rate_mc = np.sum(local_scattering_rate1[mask_PP1]) \
                                   + np.sum(local_scattering_rate2[mask_PP2])

        # This is OK (benchmarked with APS-ERL)
        piwinski_total_scattering_rate = self._compute_piwinski_total_scattering_rate(self.element)

        # mc_to_piwinski_ratio = total_scattering_rate_mc / piwinski_total_scattering_rate
        # print(f'\nMC to Piwinski ratio: {mc_to_piwinski_ratio}\n')

        integrated_piwinski_total_scattering_rate = self.integrated_piwinski_total_scattering_rates[self.element]

        # TODO: check if better to normalize over total_scattering_rate_mc or piwinski_total_scattering_rate
        total_scattering_rate1 = local_scattering_rate1[mask_PP1] / total_scattering_rate_mc * integrated_piwinski_total_scattering_rate
        total_scattering_rate2 = local_scattering_rate2[mask_PP2] / total_scattering_rate_mc * integrated_piwinski_total_scattering_rate

        total_scattering_rate = np.concatenate((total_scattering_rate1, total_scattering_rate2))

        return total_scattering_rate, total_scattering_rate_mc, piwinski_total_scattering_rate


class TouschekManager:
    def __init__(self, line, local_momaper, n_elems, nemitt_x, nemitt_y, sigma_z, sigma_delta, kb, n_part_mc, fdelta=0.85, nx=3, ny=3, nz=3):
        self.line = line
        self.local_momaper = local_momaper
        self.ref_particle = line.particle_ref
        self.n_elems = n_elems
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.sigma_z = sigma_z
        self.sigma_delta = sigma_delta
        self.kb = kb
        self.n_part_mc = n_part_mc
        self.fdelta = fdelta
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.s = []
        self.touschek_dict = {}
        self.touschek = TouschekCalculator(self)


    def _get_s_elements_to_insert(self):
        ds = self.line.get_length() / self.n_elems
        s = np.linspace(ds, self.line.get_length(), self.n_elems)
        s = s.tolist()

        coll_idx = []
        for idx, elem in enumerate(self.line.elements):
            if isinstance(elem, xc.Geant4Collimator):
                coll_idx.append(idx)
        coll_idx = np.array(coll_idx, dtype=int)

        s_ele_us = np.array(self.line.get_s_elements(mode='upstream'))
        s_ele_ds = np.array(self.line.get_s_elements(mode='downstream'))
        s_coll_us = np.take(s_ele_us, coll_idx)
        s_coll_ds = np.take(s_ele_ds, coll_idx)

        coll_regions = np.column_stack((s_coll_us, s_coll_ds))

        # Initialize a mask with False values
        mask_in_coll_region = np.zeros_like(s, dtype=bool)

        # Iterate over each collimator region and update the mask
        for us, ds in coll_regions:
            # Check if elements in s_array fall between the upstream and downstream of any collimator
            mask_in_coll_region |= (s >= us) & (s <= ds)

        tolerance = 1e-3 # m
        for idx, is_in_coll_region in enumerate(mask_in_coll_region):
            if is_in_coll_region:
                s_coll = coll_regions[np.any(np.isclose(coll_regions, s[idx]), axis=1)]
                argmin = np.argmin(np.abs(s[idx] - s_coll))
                s_closest = s_coll[0][argmin]
                if argmin == 0: 
                    s[idx] = s_closest - tolerance
                elif argmin == 1:
                    s[idx] = s_closest + tolerance

        return s


    def _install_touschek_markers(self, s):
        print('\nInstalling Touschek markers...')
        touschek_markers_dict = {}
        for ii in range(self.n_elems):
            touschek_markers_dict[f'TMarker_{ii}'] = xt.Marker()

        self.line._insert_thin_elements_at_s([(s_elem, [(key, touschek_markers_dict[key])]) for s_elem, key in zip(s, touschek_markers_dict.keys())])

        print('Done.\n')


    def _generate_coord_and_compute_density(self, twiss, element):
        n_part_over_two = int(self.n_part_mc / 2)
        beta0 = self.ref_particle.beta0[0]
        gamma0 = self.ref_particle.gamma0[0]

        kb = self.kb

        nx = self.nx
        ny = self.ny
        nz = self.nz

        sigma_z = self.sigma_z
        sigma_delta = self.sigma_delta

        gemitt_x = self.nemitt_x / beta0 / gamma0
        gemitt_y = self.nemitt_y / beta0 / gamma0   
        gemitt_z = sigma_z * sigma_delta

        bets = sigma_z / sigma_delta

        alfx = twiss['alfx', element]
        betx = twiss['betx', element]

        alfy = twiss['alfy', element]
        bety = twiss['bety', element]   

        dx = twiss['dx', element]
        dpx = twiss['dpx', element]

        dy = twiss['dy', element]
        dpy = twiss['dpy', element]     

        sigmab_x = np.sqrt(gemitt_x * betx) # Horizontal betatron beam size
        sigmab_y = np.sqrt(gemitt_y * bety) # Vertical betatron beam size

        x_norm1 = np.random.uniform(-nx*np.sqrt(gemitt_x), nx*np.sqrt(gemitt_x), n_part_over_two)
        px_norm1 = np.random.uniform(-nx*np.sqrt(gemitt_x), nx*np.sqrt(gemitt_x), n_part_over_two)
        px_norm2 = np.random.uniform(-nx*np.sqrt(gemitt_x), nx*np.sqrt(gemitt_x), n_part_over_two)
        y_norm1 = np.random.uniform(-ny*np.sqrt(gemitt_y), ny*np.sqrt(gemitt_y), n_part_over_two)
        py_norm1 = np.random.uniform(-ny*np.sqrt(gemitt_y), ny*np.sqrt(gemitt_y), n_part_over_two)
        py_norm2 = np.random.uniform(-ny*np.sqrt(gemitt_y), ny*np.sqrt(gemitt_y), n_part_over_two)
        zeta_norm1 = np.random.uniform(-nz*np.sqrt(gemitt_z), nz*np.sqrt(gemitt_z), n_part_over_two)
        delta_norm1 = np.random.uniform(-nz*np.sqrt(gemitt_z), nz*np.sqrt(gemitt_z), n_part_over_two)
        delta_norm2 = np.random.uniform(-nz*np.sqrt(gemitt_z), nz*np.sqrt(gemitt_z), n_part_over_two)

        xb1 = np.sqrt(betx) * x_norm1 
        pxb1 = (px_norm1 - alfx*x_norm1) / np.sqrt(betx)

        yb1 = np.sqrt(bety) * y_norm1
        pyb1 = (py_norm1 - alfy*y_norm1) / np.sqrt(bety)

        zeta1 = np.sqrt(bets) * zeta_norm1
        zeta2 = zeta1
        delta1 = (delta_norm1) / np.sqrt(bets)
        delta2 = (delta_norm2) / np.sqrt(bets)

        # Dispersion correction
        x1 = xb1 + dx * delta1
        px1 = pxb1 + dpx * delta1
        y1 = yb1 + dy * delta1 
        py1 = pyb1 + dpy * delta1

        x2b = x1 - dx * delta2
        y2b = y1 - dy * delta2

        x_norm2 = x2b / np.sqrt(betx)
        y_norm2 = y2b / np.sqrt(bety)

        pxb2 = (px_norm2 - alfx*x_norm2) / np.sqrt(betx)
        pyb2 = (py_norm2 - alfy*y_norm2) / np.sqrt(bety)

        x2 = x1
        px2 = pxb2 + dpx * delta2
        y2 = y1
        py2 = pyb2 + dpy * delta2

        PP1 = np.column_stack((x1, px1, y1, py1, zeta1, delta1))

        dens1 = kb / (8 * np.pi**3 * gemitt_x * gemitt_y * gemitt_z) \
                                * np.exp(-zeta1**2 / (2 * sigma_z**2) - delta1**2 / (2 * sigma_delta**2)) \
                                * np.exp(-(xb1**2 + (alfx*xb1 + betx*pxb1)**2) / (2 * sigmab_x**2) \
                                         -(yb1**2 + (alfy*yb1 + bety*pyb1)**2) / (2 * sigmab_y**2))
        
        PP2 = np.column_stack((x2, px2, y2, py2, zeta2, delta2))

        dens2 = kb / (8 * np.pi**3 * gemitt_x * gemitt_y * gemitt_z) \
                                * np.exp(-zeta2**2 / (2 * sigma_z**2) - delta2**2 / (2 * sigma_delta**2)) \
                                * np.exp(-(x2b**2 + (alfx*x2b + betx*pxb2)**2) / (2 * sigmab_x**2) \
                                         -(y2b**2 + (alfy*y2b + bety*pyb2)**2) / (2 * sigmab_y**2))

        return PP1, dens1, PP2, dens2
    

    def _compute_phase_space_volume(self, twiss, element):
        nx = self.nx
        ny = self.ny
        nz = self.nz

        sigma_z = self.sigma_z
        sigma_delta = self.sigma_delta

        gemitt_x = self.nemitt_x / self.ref_particle.beta0[0] / self.ref_particle.gamma0[0] 
        betx = twiss['betx', element]

        gemitt_y = self.nemitt_y / self.ref_particle.beta0[0] / self.ref_particle.gamma0[0]
        bety = twiss['bety', element]

        bets = sigma_z / sigma_delta
        gemitt_z = sigma_z**2 / bets

        phase_space_volume = np.pi**2 / np.sqrt(betx * bety * bets) * (2*nx*np.sqrt(gemitt_x))**3 * (2*ny*np.sqrt(gemitt_y))**3 * (2*nz*np.sqrt(gemitt_z))**3
        
        return phase_space_volume


    def initialise_touschek(self, element=None):
        self.s = self._get_s_elements_to_insert()
        self._install_touschek_markers(self.s)

        self.line.build_tracker()
        twiss = self.line.twiss()

        # Pass the twiss table to the TouschekCalculator
        self.touschek.twiss = twiss

        # Pass the local momentum aperture to the TouschekCalculator
        self.touschek.local_momentum_aperture = self.local_momaper
            
        # Assign the Piwikinski total scattering rate to each element with a length 
        # and to each Touschek marker
        print('\nComputing Piwinski total scattering rates...')
        self.touschek._assign_piwinski_total_scattering_rates()
        print('Done.\n')

        if element is not None:
            # Get the index of the Touschek scattering center being considered
            # element need to be of the form 'TMarker_{ii_touschek_element}'
            ii_touschek_element = int(re.search(r'\d+', element).group())
            self.s = [self.s[ii_touschek_element]]

        for ii in range(len(self.s)):
            # Pass the name of the Touschek marker to the TouschekCalculator
            self.touschek.element = f'TMarker_{ii if element is None else ii_touschek_element}'

            print(f'Generating 6D coordinates and computing phase space density at element {self.touschek.element}...')
            PP1, dens1, PP2, dens2 = self._generate_coord_and_compute_density(twiss, self.touschek.element)

            print(f'Scattering particles at element {self.touschek.element}...')
            PP1, PP2 = self.touschek.scatter(PP1, PP2)

            print(f'Compute phase space volume at element {self.touschek.element}...')
            phase_space_volume = self._compute_phase_space_volume(twiss, self.touschek.element)

            print(f'Computing scattering rates at element {self.touschek.element}...')
            total_scattering_rate, mc_rate, piwinski_rate = self.touschek.compute_total_scattering_rate(phase_space_volume,
                                                                                                        dens1, dens2)
            
            print('\n')

            # Filter out the particles that have delta less than delta_min
            PP1 = PP1[self.touschek.mask_PP1, :]
            PP2 = PP2[self.touschek.mask_PP2, :]

            PP = np.vstack((PP1, PP2))

            # Target coverage of total scattering rate (elegant default = 0.99)
            keep_portion = getattr(self, "keep_portion", 0.99)
            # # Safety floor to avoid tiny samples when a few weights dominate
            # min_keep = getattr(self, "min_keep", 20000)

            # Sort by descending weight
            idx_desc = np.argsort(total_scattering_rate)[::-1]
            w_sorted = total_scattering_rate[idx_desc]
            cum = np.cumsum(w_sorted)
            total_w = cum[-1]

            # Minimal n that reaches the desired coverage
            cutoff = keep_portion * total_w
            n_keep = np.searchsorted(cum, cutoff, side="left") + 1
            # Enforce minimum sample size
            # n_keep = max(n_keep, min_keep)

            keep_idx = np.sort(idx_desc[:n_keep])

            # Apply selection
            PP = PP[keep_idx,:]
            total_scattering_rate = total_scattering_rate[keep_idx]

            # This is the number of particles we will track
            n_part_to_track = n_keep

            # Prepare particle object
            # NOTE: Touschek scattering rates assigned as particle weights.
            #       This could affect collective simulations (e.g., beam-beam),
            #       but should be OK for single-particle tracking
            particles = xt.Particles(
                mass0=self.ref_particle.mass0,
                q0=self.ref_particle.q0,
                p0c=self.ref_particle.p0c[0],
                x=PP[:,0], px=PP[:,1],
                y=PP[:,2], py=PP[:,3],
                zeta=PP[:,4], delta=PP[:,5],
                weight=total_scattering_rate,
                _capacity=2*n_part_to_track
            )

            particles.start_tracking_at_element = -1

            self.touschek_dict[self.touschek.element] = {'particles': particles,
                                                         'mc_rate': mc_rate,
                                                         'piwinski_rate': piwinski_rate,
                                                         'mc_to_piwinski_ratio': mc_rate / piwinski_rate}

        self.line.discard_tracker()

        return
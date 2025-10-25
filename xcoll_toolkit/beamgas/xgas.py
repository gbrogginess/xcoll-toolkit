"""
Python module for Monte Carlo simulation of beam-gas interactions in Xsuite.
=============================================
Author(s): Giacomo Broggi, Andrey Abramov
Email:  giacomo.broggi@cern.ch
Date:   12-03-2025
"""
# ===========================================
# 🔹 Required modules
# ===========================================
import numpy as np
import pandas as pd
import xtrack as xt
import xcoll as xc
import periodictable as pt

from collections import Counter
from scipy.constants import physical_constants

# ===========================================
# 🔹 Constants
# ===========================================
ELECTRON_MASS_EV = xt.ELECTRON_MASS_EV
EV_TO_MEV = 1E-6
MEV_TO_EV = 1E6

C_LIGHT = physical_constants['speed of light in vacuum'][0]
ALPHA = physical_constants['fine-structure constant'][0]
HBAR = physical_constants['Planck constant over 2 pi in eV s'][0]
CLASSICAL_ELECTRON_RADIUS = physical_constants['classical electron radius'][0]
BOHR_RADIUS = physical_constants['Bohr radius'][0]
ELECTRON_REDUCED_COMPTON_WAVELENGTH = physical_constants['reduced Compton wavelength'][0]
ATOMIC_MASS_CONSTANT_EV = physical_constants['atomic mass constant energy equivalent in MeV'][0] * MEV_TO_EV

C_TF = 1/2 * (3*np.pi/4)**(2/3) # Thomas-Fermi constant


def _energy_from_momentum(p):
    return np.sqrt(p**2 + ELECTRON_MASS_EV**2)


def _atomic_number_from_symbol(element_symbol):
    element = getattr(pt, element_symbol)
    return element.number


class ElementData:
    def __init__(self, Z):
        self.Z = Z
        self.A = Z*2 if Z != 1 else 1
        self.mass = pt.elements[Z].mass * ATOMIC_MASS_CONSTANT_EV
        self.nuclear_radius = 1.27e-15 * self.A**0.27
        self.f_el, self.f_inel = self._compute_radiation_logarithms()
        self.f_c = self._compute_Coulomb_factor()
        self.f_Z_factor_1, self.f_Z_factor_2, self.f_Z = self._compute_Z_factors()
        self.gamma_factor, self.epsilon_factor = self._compute_gamma_epsilon_factors()


    def _compute_Coulomb_factor(self):
        K1, K2, K3, K4 = 0.0083, 0.20206, 0.0020, 0.0369
        a_Z = ALPHA * self.Z

        return (K1 * a_Z**4 + K2 + 1 / (1 + a_Z**2)) * a_Z**2 - (K3 * a_Z**4 + K4) * a_Z**4


    def _compute_radiation_logarithms(self):
        # Compute elastic and inelastic radiation logarithms

        # f_el and f_inel for low-Z elements (where Thomas-Fermi model is not accurate)
        # Computed using the Dirac-Fock atomic model
        F_EL_LOWZ = [0.0, 5.3104, 4.7935, 4.7402, 4.7112, 4.6694, 4.6134, 4.5520]
        F_INEL_LOWZ = [0.0, 5.9173, 5.6125, 5.5377, 5.4728, 5.4174, 5.3688, 5.3236]

        if self.Z < 5:
            f_el = F_EL_LOWZ[self.Z]
            f_inel = F_INEL_LOWZ[self.Z]
        else:
            f_el = np.log(184.15) - np.log(self.Z) / 3
            f_inel = np.log(1194) - 2 * np.log(self.Z) / 3

        return f_el, f_inel
    

    def _compute_Z_factors(self):
        f_Z_factor_1 = (self.f_el - self.f_c) + self.f_inel/self.Z
        f_Z_factor_2 = (1 + 1/self.Z) / 12

        f_Z = np.log(self.Z)/3 + self.f_c

        return f_Z_factor_1, f_Z_factor_2, f_Z


    def _compute_gamma_epsilon_factors(self):
        gamma_factor = 100 * ELECTRON_MASS_EV * EV_TO_MEV / np.cbrt(self.Z)
        epsilon_factor = 100 * ELECTRON_MASS_EV * EV_TO_MEV / (np.cbrt(self.Z) ** 2)

        return gamma_factor, epsilon_factor


class CoulombScatteringCalculator:
    def __init__(self, Z, q0, p0c, theta_lim=(1e-7, 50e-3)):
        self.Z = Z
        self.q0 = q0
        self.p0c = p0c
        self.theta_lim = theta_lim
        self.element_data = ElementData(Z)
        self.ekin = _energy_from_momentum(p0c) - ELECTRON_MASS_EV

        etot = self.ekin + ELECTRON_MASS_EV
        self.gamma = etot / ELECTRON_MASS_EV
        self.beta = np.sqrt(1.0 - 1.0 / (self.gamma**2))

        # Thomas–Fermi radius
        self.a_TF = C_TF * BOHR_RADIUS * self.Z**(-1/3)

        # small guard factor (Geant4-style) for rejection step
        self._gmax = 1.0 + 2e-4 * (self.Z**2)

        num = 2*self.element_data.mass * self.ekin * (self.ekin + 2*ELECTRON_MASS_EV)
        den = ELECTRON_MASS_EV**2 + self.element_data.mass**2 + 2*self.element_data.mass*etot
        self.tmax = num / den


    def _screening_As(self):
        # Screening parameter (Moliere, 1947)
        return (HBAR * C_LIGHT / (2.0 * self.p0c * self.a_TF))**2 * \
               (1.13 + 3.76 * (ALPHA * self.Z / self.beta)**2)


    def _mf_ratio(self, theta):
        # McKinley and Fesbach Mott-to-Rutherford ratio
        s = np.sin(theta * 0.5)

        return 1.0 - (self.beta**2) * (s**2) - self.q0 * self.Z * ALPHA * self.beta * np.pi * s * (1.0 - s)
    

    def _exp_form_factor(self, theta):
        # From G4ScreeningMottCrossSection::FormFactor2ExpHof
        s2 = np.sin(theta * 0.5) ** 2
        t = self.tmax * s2
        q2 = (t * (t + 2.0 * self.element_data.mass)) * (HBAR*C_LIGHT)**-2
        xN = self.element_data.nuclear_radius**2 * q2
        den = 1.0 + xN / 12.0
        FN = 1.0 / den

        return FN*FN


    def _compute_dxsec(self, theta):
        # Wenztel-Mott differential cross section (no nuclear form factor)
        # Rutherford prefactor (LAB, infinite target mass)
        s_half = np.sin(theta * 0.5)
        dxsec_rutherford = (self.Z**2) * CLASSICAL_ELECTRON_RADIUS**2 / 4.0 \
                           * (self.beta**-4) * (self.gamma**-2) * (s_half**-4)

        # McKinley and Fesbach Mott-to-Rutherford ratio
        R_McF = self._mf_ratio(theta)

        # Screening parameter (Moliere, 1947)
        As = self._screening_As()
        screening_factor = (s_half**2) / (As + s_half**2)

        # Nuclear form factor 
        F2 = self._exp_form_factor(theta)

        dxsec = dxsec_rutherford * R_McF * (screening_factor**2) * F2

        return dxsec * 2.0*np.pi*np.sin(theta)


    def _sample_theta(self, n):
        # screening parameter (constant over θ for fixed beam/target)
        As = self._screening_As()

        # limits in z = 1 - cosθ
        z1 = 1.0 - np.cos(self.theta_lim[0])
        z2 = 1.0 - np.cos(self.theta_lim[1])

        def sample_z(nleft):
            u = np.random.random(nleft)
            a_lo = 1.0 / (2.0 * As + z1)
            a_hi = 1.0 / (2.0 * As + z2)
            inv = a_lo - u * (a_lo - a_hi)  # = 1 / (2 As + z)
            return (1.0 / inv) - 2.0 * As

        out = []
        while len(out) < n:
            z = sample_z(n - len(out))
            theta = np.arccos(1.0 - z)
            # rejection on MF only
            acc = np.random.random(theta.size) < (self._mf_ratio(theta) / self._gmax)
            if np.any(acc):
                out.extend(theta[acc])

        return np.array(out[:n])


    def sample_deflections(self, particles, n):
        phi = np.random.uniform(0.0, 2.0*np.pi, n)
        theta = self._sample_theta(n)

        sintheta = np.sin(theta); costheta = np.cos(theta)
        sinphi = np.sin(phi);     cosphi = np.cos(phi)

        pz = np.sqrt((1.0 + particles.delta)**2 - particles.px**2 - particles.py**2)
        PP = np.column_stack((particles.px, particles.py, pz))
        norms = np.linalg.norm(PP, axis=1, keepdims=True)
        PP_HAT = PP / norms

        UU_HAT = np.zeros_like(PP_HAT)
        tol = 1e-12
        mask = (PP_HAT[:, 0]**2 + PP_HAT[:, 1]**2) < tol**2

        UU_HAT[~mask] = np.stack([-PP_HAT[~mask, 1], PP_HAT[~mask, 0], np.zeros_like(PP_HAT[~mask, 0])], axis=1)
        UU_HAT[mask] = np.array([1.0, 0.0, 0.0])
        UU_HAT /= np.linalg.norm(UU_HAT, axis=1, keepdims=True)

        VV_HAT = np.cross(PP_HAT, UU_HAT)

        scattered_dir = (
            sintheta[:, None] * cosphi[:, None] * UU_HAT +
            sintheta[:, None] * sinphi[:, None] * VV_HAT +
            costheta[:, None] * PP_HAT
        )
        PP_OUT = scattered_dir * norms
        return PP_OUT[:, 0].tolist(), PP_OUT[:, 1].tolist()


    def compute_xsec(self):
        from scipy.integrate import quad
        return quad(self._compute_dxsec, self.theta_lim[0], self.theta_lim[1])[0]


class BremsstrahlungCalculator:
    def __init__(self, Z, p0c, energy_cut=10e3):
        self.Z = Z
        self.p0c = p0c
        self.energy_cut = energy_cut # Default energy cut is 10 keV
        self.element_data = ElementData(Z)
        self.ekin = _energy_from_momentum(p0c) - ELECTRON_MASS_EV


    def _compute_dxsec(self, gamma_energy):
        etot = self.ekin + ELECTRON_MASS_EV
        y = gamma_energy / etot
        dum0 = (1 - y) + 0.75 * y**2
        dum1 = y / (etot - gamma_energy)
        gamma = dum1 * self.element_data.gamma_factor
        epsilon = dum1 * self.element_data.epsilon_factor

        phi1, phi1m2, psi1, psi1m2 = self._compute_screening_functions(gamma, epsilon)

        if self.Z < 5:
            dxsec = dum0*self.element_data.f_Z_factor_1 + (1-y)*self.element_data.f_Z_factor_2
        else:
            dxsec = dum0*((0.25*phi1 - self.element_data.f_Z) + \
                          (0.25*psi1 - 2*np.log(self.Z)/3) / self.Z) + \
                          (0.125*(1 - y)*(phi1m2 + psi1m2/self.Z))
            
        return dxsec


    def _sample_gamma_energies(self, n):
        # Set the density correction factor to 0
        # Back-of-the-envelope-calculation shows that it is not relevant for the purpose of high-energy e+/e- on low-density gas
        f_density_corr = 0
        max_energy = self.ekin

        func_max = self.element_data.f_Z_factor_1 + self.element_data.f_Z_factor_2

        # Define the min and the max of the transformed variable
        xmin = np.log(self.energy_cut**2 + f_density_corr)
        xrange = np.log(max_energy**2 + f_density_corr) - xmin

        gamma_energies = []
        while len(gamma_energies) < n:
            # Generate two random numbers between 0 and 1
            rndm = np.random.rand(2)
            gamma_energy = np.sqrt(max(np.exp(xmin + rndm[0] * xrange) - f_density_corr, 0))
            func_val = self._compute_dxsec(gamma_energy)
            # Check if the generated gamma energy meets the acceptance condition
            if func_val >= func_max * rndm[1]:
                gamma_energies.append(gamma_energy)

        return np.array(gamma_energies)
    

    def sample_deltas(self, n):
        gamma_energies = self._sample_gamma_energies(n)
        delta = ((self.p0c*np.ones(n) - gamma_energies) - self.p0c) / self.p0c

        return delta
        

    def _sample_costheta(self, n):
        # From G4ModfiedTsai.cc
        u_max = 2 * (1 + self.ekin / ELECTRON_MASS_EV)
        a1 = 1.6
        a2 = a1 / 3.0
        border = 0.25

        costheta = []
        while len(costheta) < n:
            uu = -np.log(np.random.rand() * np.random.rand())
            u = uu * a1 if np.random.rand() < border else uu * a2
            if u <= u_max:
                cos_theta = 1.0 - 2.0 * u * u / (u_max * u_max)
                costheta.append(cos_theta)
        
        return np.array(costheta)
    
    @staticmethod
    def _rotate_to_direction(vectors, directions):
        # Rodrigues rotation
        z = np.array([0, 0, 1])
        rotated = []
        for vec, target in zip(vectors, directions):
            v = np.cross(z, target)
            s = np.linalg.norm(v)
            c = np.dot(z, target)
            if s == 0:
                rotated.append(vec * c)
            else:
                vx = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
                rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
                rotated.append(rot @ vec)

        return np.array(rotated)
    

    def sample_deflections(self, particles, n):
        # Step 1: sample gamma energies
        gamma_energies = self._sample_gamma_energies(n)

        # Step 2: sample angles using Tsai model
        costheta = self._sample_costheta(n)
        sintheta = np.sqrt(1 - costheta**2)
        phi = np.random.uniform(0, 2*np.pi, n)

        # Step 3: build gamma directions (in local z frame)
        gammadirs_local = np.column_stack((
            sintheta * np.cos(phi),
            sintheta * np.sin(phi),
            costheta
        ))

        # Step 4: rotate gamma directions into particle frame
        # Get current momentum directions from particles
        pz = np.sqrt((1 + particles.delta)**2 - particles.px**2 - particles.py**2)
        PP = np.column_stack((particles.px, particles.py, pz))
        norms = np.linalg.norm(PP, axis=1, keepdims=True) # This is equivalent to 1 + particles.delta
        PP_HAT = PP / norms

        gammadirs_rotated = self._rotate_to_direction(gammadirs_local, PP_HAT)

        # Step 5: compute gamma momenta
        PP_GAMMAS = gamma_energies[:, None] * gammadirs_rotated

        # Step 6: compute final momentum (conservation)
        PP_OUT = (PP * self.p0c - PP_GAMMAS) / self.p0c

        delta = np.linalg.norm(PP_OUT, axis=1) - 1

        return PP_OUT[:, 0].tolist(), PP_OUT[:, 1].tolist(), delta.tolist()

    
    def _compute_screening_functions(self, gamma, epsilon):
        phi1 = 16.863 - 2 * np.log(1 + 0.311877 * gamma ** 2) + 2.4 * np.exp(-0.9 * gamma) + 1.6 * np.exp(-1.5*gamma)
        phi1m2 = 2 / (3 + 19.5 * gamma + 18 * gamma ** 2)
        psi1 = 24.34 - 2 * np.log(1 + 13.111641 * epsilon ** 2) + 2.8 * np.exp(-8 * epsilon) + 1.2 * np.exp(-29.2 * epsilon)
        psi1m2 = 2 / (3 + 120 * epsilon + 1200 * epsilon ** 2)

        return phi1, phi1m2, psi1, psi1m2


    def compute_xsec(self):
        etot = _energy_from_momentum(self.p0c)
        alpha_min = np.log(self.energy_cut / etot)
        alpha_max = np.log(self.ekin / self.energy_cut)
        n_sub = max(int(0.45 * alpha_max), 0) + 4
        delta = alpha_max / n_sub

        # abscissas and weights of an 8 point Gauss-Legendre quadrature
        # for numerical integration on [0,1]
        gXGL = np.array([1.98550718e-02, 1.01666761e-01, 2.37233795e-01, 4.08282679e-01,
                        5.91717321e-01, 7.62766205e-01, 8.98333239e-01, 9.80144928e-01])
        
        gWGL = np.array([5.06142681e-02, 1.11190517e-01, 1.56853323e-01, 1.81341892e-01,
                        1.81341892e-01, 1.56853323e-01, 1.11190517e-01, 5.06142681e-02])

        # Set minimum value of the first sub-interval
        alpha_i = alpha_min

        xsec = 0
        for _ in range(n_sub):
            for igl in range(8):
                # Compute the emitted photon energy k
                k = np.exp(alpha_i + gXGL[igl] * delta) * etot
                # Compute the DCS value at k
                dcs = self._compute_dxsec(k)
                xsec += gWGL[igl] * dcs
            # Update sub-interval minimum value
            alpha_i += delta

        # Apply corrections due to variable transformation
        xsec *= delta

        return 16 * ALPHA * (CLASSICAL_ELECTRON_RADIUS**2) * (self.Z**2) / 3 * xsec


class BeamGasManager():
    log_interacted_part_ids = []
    df_interactions_log = pd.DataFrame(columns=['name', 's', 'particle_id', 'interaction'])

    def __init__(self, density_df, q0, p0c, eBrem, CoulombScat, eBrem_energy_cut=10e3, theta_lim=(1e-7, 50e-3), interaction_length_is_n_turns=1):
        self.rng = np.random.default_rng()

        if eBrem and CoulombScat:
            raise ValueError("Only one of eBrem or CoulombScat can be True at a time.")

        self.q0 = q0
        self.p0c = p0c
        self.density_df = density_df
        self.interaction_length_is_n_turns = interaction_length_is_n_turns

        self.atomic_species = {
            element: _atomic_number_from_symbol(element)
            for element in density_df.columns[1:]
        }

        self.eBrem = None
        self.CoulombScat = None

        if eBrem:
            self.eBrem = {
                key: BremsstrahlungCalculator(self.atomic_species[key], p0c, energy_cut=eBrem_energy_cut)
                for key in self.atomic_species
            }
        
        if CoulombScat:
            self.CoulombScat = {
                key: CoulombScatteringCalculator(self.atomic_species[key], q0, p0c, theta_lim=theta_lim)
                for key in self.atomic_species
            }

        self.bg_element_names = None
        self.particles = None
        self.circumference = None
        self.interaction_dist = None
        self.part_initialised = False


    def cross_section_biasing(self, gas_parameters):
        avg_gas_parameters = {
            kk: {'n_avg': self.density_df[kk.split('_')[0]].mean(), 'xsec': gas_parameters[kk]['xsec']}
            for kk in gas_parameters
            }

        avg_mfp = [1 / (gg['n_avg'] * gg['xsec']) for gg in avg_gas_parameters.values()]

        avg_mfp_tot = 1 / sum([1/_mfp for _mfp in avg_mfp])

        biasing_factor = avg_mfp_tot / (self.interaction_length_is_n_turns * self.circumference)

        for key in gas_parameters:
            gas_parameters[key]['xsec'] *= biasing_factor

        return gas_parameters


    def initialise_particles(self, particles):
        self.particles = particles
        active_part_ids = particles.particle_id[particles.state > 0]
        self.interaction_dist = dict(zip(active_part_ids, -np.log(self.rng.random(len(active_part_ids)))))
        self.part_initialised = True

    def install_beam_gas_markers(self, line):
        self.circumference = line.get_length()
        dict_bg_markers = {}
        s = []

        for index, values in self.density_df.iterrows():
            dict_bg_markers[f'BGMarker_{index}'] = xt.Marker()
            s.append(values.s)

        coll_idx = []
        for idx, elem in enumerate(line.elements):
            if isinstance(elem, xc.Geant4Collimator):
                coll_idx.append(idx)
        coll_idx = np.array(coll_idx)

        s_ele_us = np.array(line.get_s_elements(mode='upstream'))
        s_ele_ds = np.array(line.get_s_elements(mode='downstream'))
        s_coll_us = np.take(s_ele_us, coll_idx)
        s_coll_ds = np.take(s_ele_ds, coll_idx)

        coll_regions = np.column_stack((s_coll_us, s_coll_ds))

        # Initialize a mask with False values
        mask_in_coll_region = np.zeros_like(s, dtype=bool)

        # Iterate over each collimator region and update the mask
        for us, ds in coll_regions:
            # Check if elements in s_array fall between the upstream and downstream of any collimator
            mask_in_coll_region |= (s >= us) & (s <= ds)

        tolerance = 1e-3  # m
        for idx, is_in_coll_region in enumerate(mask_in_coll_region):
            if is_in_coll_region:
                # Find the collimator region in which s[idx] is located
                in_region_idx = np.where((coll_regions[:, 0] <= s[idx]) & (s[idx] <= coll_regions[:, 1]))[0]

                if len(in_region_idx) == 0:
                    print(f"[WARNING] s[{idx}] = {s[idx]:.6f} flagged as in a collimator region but no matching region found")
                    continue

                us, ds = coll_regions[in_region_idx[0]]

                # Distances to the upstream and downstream edges
                dist_us = abs(s[idx] - us)
                dist_ds = abs(s[idx] - ds)

                # Shift marker just outside the closest boundary
                if dist_us < dist_ds:
                    s_new = us - tolerance
                else:
                    s_new = ds + tolerance

                print(f"[INFO] Moving BGMarker_{idx}: {s[idx]:.6f} → {s_new:.6f} (moved outside collimator region)")
                s[idx] = s_new

        BeamGasManager.df_interactions_log['s'] = s
        
        elements_to_insert = [(s_elem, [(key, dict_bg_markers[key])]) for s_elem, key in zip(s, dict_bg_markers.keys())]

        line._insert_thin_elements_at_s(elements_to_insert)

    def install_beam_gas_elements(self, line):
        if line.tracker is not None:
            line.discard_tracker()

        self.circumference = line.get_length()
        ds_list = [self.density_df.s.iloc[0] + self.circumference - self.density_df.s.iloc[-1]] + list(np.diff(self.density_df.s))
        dict_bg_elems = {}

        for index, values in self.density_df.iterrows():
            local_gas_params = {}

            # Add eBrem parameters if eBrem is initialized
            if self.eBrem is not None:
                local_gas_params.update({
                    f'{key}_eBrem': {'n': values[key], 'xsec': self.eBrem[key].compute_xsec()}
                    for key in self.density_df.columns[1:]
                })

            # Add CoulombScat parameters if CoulombScat is initialized
            if self.CoulombScat is not None:
                local_gas_params.update({
                    f'{key}_CoulombScat': {'n': values[key], 'xsec': self.CoulombScat[key].compute_xsec()}
                    for key in self.density_df.columns[1:]
                })

            # Bias the cross-section parameters
            local_gas_params = self.cross_section_biasing(local_gas_params)
            dict_bg_elems[f'beam_gas_{index}'] = BeamGasElement(ds_list[index], local_gas_params, f'beam_gas_{index}', self)
        
        self.bg_element_names = list(dict_bg_elems.keys())
        BeamGasManager.df_interactions_log['name'] = self.bg_element_names.copy()
        
        newLine = xt.Line(elements=[], element_names=[])
        for ee, nn in zip(line.elements, line.element_names):
            if nn.startswith('BGMarker_'):
                ii_bg = nn.split('_')[1]
                nn_bg = 'beam_gas_' + ii_bg
                newLine.append_element(dict_bg_elems[nn_bg], nn_bg)
            else:
                newLine.append_element(ee, nn)

        # Update the line in place
        line.element_names = newLine.element_names
        line.element_dict.update(newLine.element_dict)

    # def install_beam_gas_elements(self, line):
    #     self.circumference = line.get_length()
    #     ds_list = [self.density_df.s.iloc[0] + self.circumference - self.density_df.s.iloc[-1]] + list(np.diff(self.density_df.s))
    #     dict_bg_elems = {}
    #     s = []

    #     for index, values in self.density_df.iterrows():
    #         local_gas_params = {}

    #         # Add eBrem parameters if eBrem is initialized
    #         if self.eBrem is not None:
    #             local_gas_params.update({
    #                 f'{key}_eBrem': {'n': values[key], 'xsec': self.eBrem[key].compute_xsec()}
    #                 for key in self.density_df.columns[1:]
    #             })

    #         # Add CoulombScat parameters if CoulombScat is initialized
    #         if self.CoulombScat is not None:
    #             local_gas_params.update({
    #                 f'{key}_CoulombScat': {'n': values[key], 'xsec': self.CoulombScat[key].compute_xsec()}
    #                 for key in self.density_df.columns[1:]
    #             })

    #         # Bias the cross-section parameters
    #         local_gas_params = self.cross_section_biasing(local_gas_params)
    #         dict_bg_elems[f'beam_gas_{index}'] = BeamGasElement(ds_list[index], local_gas_params, self)
    #         s.append(values.s)
        
    #     self.bg_element_names = list(dict_bg_elems.keys())
    #     BeamGasManager.df_interactions_log['name'] = self.bg_element_names.copy()

    #     coll_idx = []
    #     for idx, elem in enumerate(line.elements):
    #         if isinstance(elem, xc.Geant4Collimator):
    #             coll_idx.append(idx)
    #     coll_idx = np.array(coll_idx)

    #     s_ele_us = np.array(line.get_s_elements(mode='upstream'))
    #     s_ele_ds = np.array(line.get_s_elements(mode='downstream'))
    #     s_coll_us = np.take(s_ele_us, coll_idx)
    #     s_coll_ds = np.take(s_ele_ds, coll_idx)

    #     coll_regions = np.column_stack((s_coll_us, s_coll_ds))

    #     # Initialize a mask with False values
    #     mask_in_coll_region = np.zeros_like(s, dtype=bool)

    #     # Iterate over each collimator region and update the mask
    #     for us, ds in coll_regions:
    #         # Check if elements in s_array fall between the upstream and downstream of any collimator
    #         mask_in_coll_region |= (s >= us) & (s <= ds)

    #     tolerance = 1e-3 # m
    #     for idx, is_in_coll_region in enumerate(mask_in_coll_region):
    #         if is_in_coll_region:
    #             s_coll = coll_regions[np.any(np.isclose(coll_regions, s[idx]), axis=1)]
    #             argmin = np.argmin(np.abs(s[idx] - s_coll))
    #             s_closest = s_coll[0][argmin]
    #             if argmin == 0: 
    #                 s[idx] = s_closest - tolerance
    #             elif argmin == 1:
    #                 s[idx] = s_closest + tolerance

    #     BeamGasManager.df_interactions_log['s'] = s
        
    #     elements_to_insert = [(s_elem, [(key, dict_bg_elems[key])]) for s_elem, key in zip(s, dict_bg_elems.keys())]

    #     line._insert_thin_elements_at_s(elements_to_insert)


    def update_interaction_dist(self, mfp_step):
        if not self.part_initialised:
            raise Exception('Need to initialise the BeamGas manager with particles'
                            'before tracking with BeamGasManager.initialise_particles()')
        
        new_values = np.array(list(self.interaction_dist.values())) - mfp_step
        mask_interacting = new_values < 0

        # the particles here is a reference so the survival is up to date
        candidate_interacting_ids = np.array(list(self.interaction_dist.keys()))[mask_interacting]

        interacting_part_mask = (
            (self.particles.state > 0) & 
            (self.particles.parent_particle_id == self.particles.particle_id) & 
            np.in1d(self.particles.particle_id, np.array(list(self.interaction_dist.keys()))[candidate_interacting_ids]) & 
            ~np.isin(self.particles.particle_id, BeamGasManager.log_interacted_part_ids)
        )

        interacting_part_ids = self.particles.particle_id[interacting_part_mask]
        BeamGasManager.log_interacted_part_ids.extend(interacting_part_ids)

        # idx_bg_elem = BeamGasManager.df_interactions_log['particle_id'].isna().idxmax()
        # BeamGasManager.df_interactions_log.at[idx_bg_elem, 'particle_id'] = interacting_part_ids.tolist()

        noninteracting_part_mask = ~interacting_part_mask & (self.particles.state >0) & (self.particles.parent_particle_id == self.particles.particle_id)
        noninteracting_part_ids = self.particles.particle_id[noninteracting_part_mask]
        
        # sample a new interaction distance for the interacted particles
        if len(interacting_part_ids) > 0:
            self.interaction_dist.update({pid: -np.log(self.rng.random()) for pid in interacting_part_ids})

        # Update the interaction distance for the non-interacting particles
        if len(noninteracting_part_ids) > 0:
            new_distances = new_values[noninteracting_part_ids]
            self.interaction_dist.update(dict(zip(noninteracting_part_ids, new_distances)))
            
        return interacting_part_mask


    def draw_angles_and_delta(self, particles, processes, n):
        gases = [p.split('_')[0] for p in processes]
        gas_counter = Counter(gases)

        px, py, delta = [], [], []
        if self.eBrem is not None:
            npp = 0
            # Bremsstrahlung
            for gas, ngas in gas_counter.items():   
                mask = np.zeros(n, dtype=bool)
                mask[npp:npp+ngas] = True
                pp = particles.filter(mask)
                _px, _py, _delta = self.eBrem[gas].sample_deflections(pp, ngas)

                px.append(_px)
                py.append(_py)
                delta.append(_delta)

                npp += ngas

        elif self.CoulombScat is not None:
            npp = 0
            # Coulomb Scattering
            for gas, ngas in gas_counter.items():   
                mask = np.zeros(n, dtype=bool)
                mask[npp:npp+ngas] = True
                pp = particles.filter(mask)
                _px, _py = self.CoulombScat[gas].sample_deflections(pp, ngas)
                _delta = pp.delta

                px.append(_px)
                py.append(_py)
                delta.append(_delta)

                npp += ngas

        px = np.concatenate(px)
        py = np.concatenate(py)
        delta = np.concatenate(delta)

        return px, py, delta
    

class BeamGasElement(xt.BeamElement):
    def __init__(self, ds, local_gas_parameters, name, manager, **kwargs):
        self.iscollective = True
        self.isthick = False
        self.name = name
        self.ds = ds
        self.interactions = list(local_gas_parameters.keys())
        self.mfp = [1 / (gg['n'] * gg['xsec']) if gg['n'] > 0 else 1 / ( 1 * gg['xsec']) for gg in local_gas_parameters.values()]
        # NOTE: if the atomic density n is zero, the mfp is computed with considering n = 1 at/m^3 to avoid division by zero
        self.mfp_tot = 1 / sum([1/_mfp for _mfp in self.mfp])
        self.int_prob = [self.mfp_tot / _mfp for _mfp in self.mfp]
        self.mfp_step = self.ds / self.mfp_tot
        self.manager = manager
        super().__init__(**kwargs)

    def track(self, particles):
        # Which particles are interacting
        interacting_mask = self.manager.update_interaction_dist(self.mfp_step)
        n_interactions = sum(interacting_mask)

        if n_interactions == 0:
            return

        # self.update_n_interactions(n_interactions)

        # Get the type of interactions and apply the effect
        interactions = self.manager.rng.choice(self.interactions, p=self.int_prob, size=n_interactions)

        interacting_ids = self.manager.particles.particle_id[interacting_mask].tolist()
        interactions_list = interactions.tolist()

        idx_log = self.manager.df_interactions_log.index[
            self.manager.df_interactions_log['particle_id'].isna() &
            self.manager.df_interactions_log['interaction'].isna()
        ][0]

        self.manager.df_interactions_log.loc[idx_log] = {
            'name': self.name,
            's': self.manager.density_df.s.iloc[int(self.name.split("_")[-1])],
            'particle_id': interacting_ids,
            'interaction': interactions_list
        }

        px, py, delta = self.manager.draw_angles_and_delta(particles.filter(interacting_mask),
                                                            interactions,
                                                            n_interactions)
        
        particles.px[interacting_mask] = px
        particles.py[interacting_mask] = py
        
        # Update the `delta` value of the particles object. `ptau` and `rvv` and
        # `rpp` are updated accordingly.
        # Ref: https://github.com/xsuite/xtrack/blob/f45c5720c246f34cd3db593829d83f3b0c61c3b8/xtrack/particles/particles.py#L1130
        delta_temp = particles.delta.copy()
        delta_temp[interacting_mask] = delta
        particles.update_delta(delta_temp)
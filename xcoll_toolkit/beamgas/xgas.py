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
        self.A = 2*Z
        self.mass = pt.elements[Z].mass * ATOMIC_MASS_CONSTANT_EV
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


    def _compute_dxsec(self, theta):
        # Wenztel-Mott differential cross section
        # Wentzel correction accounts for the screening of the nuclear charge by atomic electrons
        # Mott correction accounts for the magnetic moment interaction between e+/e- and target nucleus
        etot = self.ekin + ELECTRON_MASS_EV
        gamma = etot / ELECTRON_MASS_EV
        beta = np.sqrt(1 - gamma**-2)
        # Thomas-Fermi radius
        a_TF = C_TF * BOHR_RADIUS * self.Z**(-1/3)
        # Rutherford differential cross section
        dxsec_rutherford = self.Z**2 * CLASSICAL_ELECTRON_RADIUS**2 / 4 * beta**-4 * gamma**-2 * np.sin(theta/2)**-4
        # McKinley and Fesbach Mott-to_rutherford ratio
        R_McF = 1 - beta**2 * np.sin(theta/2)**2 - self.q0*ALPHA*beta*np.pi*np.sin(theta/2)*(1-np.sin(theta/2))
        # Screening parameter (Moliere, 1947)
        As = (HBAR*C_LIGHT / (2*self.p0c*a_TF))**2 * (1.13 + 3.76*(ALPHA*self.Z/beta)**2)
    
        screening_factor = np.sin(theta/2)**2 / (As + np.sin(theta/2)**2)

        dxsec = dxsec_rutherford * R_McF * screening_factor**2

        return dxsec * 2*np.pi*np.sin(theta)


    def _find_max_dxsec(self):
        from scipy.optimize import minimize_scalar
        # Define the function for optimization
        def neg_dxsec(theta):
            return -self._compute_dxsec(theta)  # Multiply by -1 to use minimization for maximization

        # Use scalar minimization within the bounds, passing theta as the variable to optimize
        result = minimize_scalar(neg_dxsec, bounds=self.theta_lim, method='bounded')
        
        # The maximum dxsec value is the negative of the minimized result
        max_dxsec = -result.fun

        return max_dxsec


    def _sample_theta(self, n):
        max_dxsec = self._find_max_dxsec()
        theta_proposed = np.random.uniform(self.theta_lim[0], self.theta_lim[1], size=n)
        dxsec_values = self._compute_dxsec(theta_proposed)
        rndm = np.random.uniform(0, max_dxsec, size=n)
        accepted_theta = theta_proposed[rndm < dxsec_values]

        while len(accepted_theta) < n:
            theta_proposed = np.random.uniform(self.theta_lim[0], self.theta_lim[1], size=n-len(accepted_theta))
            dxsec_values = self._compute_dxsec(theta_proposed)
            rndm = np.random.uniform(0, max_dxsec, size=n-len(accepted_theta))
            accepted_theta = np.concatenate((accepted_theta, theta_proposed[rndm < dxsec_values]))

        return accepted_theta[:n]


    def sample_deflections(self, n):
        phi = np.random.uniform(0, 2*np.pi, n)
        theta = self._sample_theta(n)

        dpx = np.sin(theta) * np.cos(phi)
        dpy = np.sin(theta) * np.sin(phi)

        return dpx, dpy


    def compute_xsec(self):
        from scipy.integrate import quad
        def dxsec(theta):
            return self._compute_dxsec(theta)

        # Perform the integration
        xsec, error = quad(dxsec, self.theta_lim[0], self.theta_lim[1])
        
        return xsec


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
    

    def sample_deflections(self, n):
        costheta = self._sample_costheta(n)
        sintheta = np.sqrt(1 - costheta**2)
        phi = np.random.uniform(0, 2*np.pi, n)
        gammadirs = np.array([[sintheta[i] * np.cos(phi[i]), sintheta[i] * np.sin(phi[i]), costheta[i]] for i in range(len(sintheta))])
        dpx = -gammadirs[:, 0]
        dpy = -gammadirs[:, 1]

        return dpx, dpy

    
    def _compute_screening_functions(self, gamma, epsilon):
        phi1 = 16.863 - 2 * np.log(1 + 0.311877 * gamma ** 2) + 2.4 * np.exp(-0.9 * gamma)
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

        self.q0 = q0
        self.p0c = p0c
        self.density_df = density_df
        self.interaction_length_is_n_turns = interaction_length_is_n_turns

        self.atomic_species = {
            element: _atomic_number_from_symbol(element)
            for element in density_df.columns[1:]
        }

        if eBrem:
            self.eBrem = {
                key: BremsstrahlungCalculator(self.atomic_species[key], p0c, energy_cut=eBrem_energy_cut)
                for key in self.atomic_species
            }
        else:
            self.eBrem = None
        
        if CoulombScat:
            self.CoulombScat = {
                key: CoulombScatteringCalculator(self.atomic_species[key], q0, p0c, theta_lim=theta_lim)
                for key in self.atomic_species
            }
        else:
            self.CoulombScat = None

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
                s_coll = coll_regions[np.any(np.isclose(coll_regions, s[idx]), axis=1)]
                argmin = np.argmin(np.abs(s[idx] - s_coll))
                s_closest = s_coll[0][argmin]
                if argmin == 0:
                    s[idx] = s_closest - tolerance
                elif argmin == 1:
                    s[idx] = s_closest + tolerance

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
            dict_bg_elems[f'beam_gas_{index}'] = BeamGasElement(ds_list[index], local_gas_params, self)
        
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

        idx_bg_elem = BeamGasManager.df_interactions_log['particle_id'].isna().idxmax()
        BeamGasManager.df_interactions_log.at[idx_bg_elem, 'particle_id'] = interacting_part_ids.tolist()

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


    def draw_angles_and_delta(self, processes, n):
        # Initialize dpx, dpy and delta array
        dpx = np.zeros(n)
        dpy = np.zeros(n)
        delta = np.zeros(n)

        unique_processes = np.unique(processes)
        if len(unique_processes) == 1:
            # Only one interaction process on a single gas species
            gas = unique_processes[0].split('_')[0]
            process = unique_processes[0].split('_')[1]

            if process == 'eBrem':
                # Only eBrem interactions on a single gas species
                dpx, dpy = self.eBrem[gas].sample_deflections(n)
                delta = self.eBrem[gas].sample_deltas(n)

            elif process == 'CoulombScat':
                # Only CoulombScat interactions on a single gas species
                dpx, dpy = self.CoulombScat[gas].sample_deflections(n)

        # TODO: implement the possibility to have all interactions at once
        # for ii, process in enumerate(processes):
        #     atomic_species = process.split('_')[0]
        #     process = process.split('_')[1]

        #     if process == 'eBrem':
        #         dpx[ii], dpy[ii] = self.eBrem[atomic_species].sample_deflections(n)
        #         delta[ii] = self.eBrem[atomic_species].sample_deltas(n)
        #     elif process == 'CoulombScat':
        #         dpx[ii], dpy[ii] = self.CoulombScat[atomic_species].sample_deflections(n)

        return dpx, dpy, delta


class BeamGasElement(xt.BeamElement):
    def __init__(self, ds, local_gas_parameters, manager, **kwargs):
        self.iscollective = True
        self.isthick = False
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
        # self.update_n_interactions(n_interactions)

        # Get the type of interactions and apply the effect
        interactions = self.manager.rng.choice(self.interactions, p=self.int_prob, size=n_interactions)

        idx_bg_elem = self.manager.df_interactions_log['interaction'].isna().idxmax()
        self.manager.df_interactions_log.at[idx_bg_elem, 'interaction'] = interactions.tolist()

        dpx, dpy, delta = self.manager.draw_angles_and_delta(interactions, n_interactions)
        particles.px[interacting_mask] += dpx
        particles.py[interacting_mask] += dpy
        particles.delta[interacting_mask] += delta
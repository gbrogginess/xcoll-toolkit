import xtrack as xt
import xpart as xp
import xcoll as xc
import numpy as np
from scipy.stats import gaussian_kde
from scipy.constants import physical_constants

ELECTRON_MASS_EV = xt.ELECTRON_MASS_EV
C_LIGHT_VACUUM = physical_constants['speed of light in vacuum'][0]
CLASSICAL_ELECTRON_RADIUS = physical_constants['classical electron radius'][0]


class LorentzVector:
    def __init__(self, px=0, py=0, pz=0, e=0):
        self.vec = np.array([px, py, pz, e])

    def momvec(self):
        return np.array([self.vec[0], self.vec[1], self.vec[2]])
    
    def etot(self):
        return self.vec[3]

    def boost_vector(self):
        """Compute the boost vector for this Lorentz vector (momentum/energy if energy != 0)."""
        if self.vec[3] != 0:
            return self.vec[:3] / self.vec[3]
        else:
            return np.array([0, 0, 0])

    def boost(self, bst):
        """Boost this Lorentz vector by the given boost vector."""
        bx, by, bz = bst
        b2 = bx**2 + by**2 + bz**2
        gamma = 1.0 / np.sqrt(1 - b2)
        bp = bx * self.vec[0] + by * self.vec[1] + bz * self.vec[2]
        gamma2 = (gamma - 1.0) / b2 if b2 > 0 else 0.0

        # Lorentz transformation equations
        self.vec[0] += gamma2 * bp * bx + gamma * bx * self.vec[3]
        self.vec[1] += gamma2 * bp * by + gamma * by * self.vec[3]
        self.vec[2] += gamma2 * bp * bz + gamma * bz * self.vec[3]
        self.vec[3] = gamma * (self.vec[3] + bp)

    def rotate(self, theta, phi):
        """Rotate the spatial components by angles theta (polar) and phi (azimuthal)."""
        # Rotation around the z-axis by phi
        R_z = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ])

        # Rotation around the y-axis by theta
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Combine rotations (R_y * R_z)
        rotation_matrix = R_y @ R_z

        # Apply the rotation to the spatial components (px, py, pz)
        self.vec[:3] = rotation_matrix @ self.vec[:3]

    @property
    def px(self):
        return self.vec[0]

    @property
    def py(self):
        return self.vec[1]

    @property
    def pz(self):
        return self.vec[2]

    @property
    def e(self):
        return self.vec[3]
    
    @property
    def gamma(self):
        return self.vec[3] / ELECTRON_MASS_EV
    
    @property
    def beta(self):
        return np.sqrt(1 - self.gamma**-2)

    def __add__(self, other):
        """Add two Lorentz vectors."""
        return LorentzVector(self.vec[0] + other.vec[0], 
                             self.vec[1] + other.vec[1],
                             self.vec[2] + other.vec[2],
                             self.vec[3] + other.vec[3])


class TouschekCalculator():
    def __init__(self, manager):
        # Should manager be a class attribute instead of an instance attribute?
        self.manager = manager
        self.npart_over_two = int(self.manager.n_part / 2)
        self.particles = None
        self.element = None
        self.twiss = None
        self.gamma = None
        self.theta_cm = None
        self.gamma_cm = None
        self.integrated_piwinski_total_scattering_rates = {}
    

    def _etot_from_momvec(self, p):
        p = np.linalg.norm(p)
        return np.sqrt(p**2 + ELECTRON_MASS_EV**2)


    def _compute_pz(self, particles):
        return np.sqrt((1 + particles.delta)**2 - particles.px**2 - particles.py**2)


    def _get_momenta(self, particles1, particles2):
        pz_1 = self._compute_pz(particles1)
        pz_2 = self._compute_pz(particles2)

        p1, p2 = [], []
        for ii in range(self.npart_over_two):
            p1.append(particles1.p0c[0] * np.array([particles1.px[ii], particles1.py[ii], pz_1[ii]]))
            p2.append(particles2.p0c[0] * np.array([particles2.px[ii], particles2.py[ii], pz_2[ii]]))

        return np.array(p1), np.array(p2)


    def _get_fourmomenta(self, p1, p2):
        v1, v2 = [], []
        for ii in range(len(p1)):
            v1.append(LorentzVector(p1[ii][0], p1[ii][1], p1[ii][2], self._etot_from_momvec(p1[ii])))
            v2.append(LorentzVector(p2[ii][0], p2[ii][1], p2[ii][2], self._etot_from_momvec(p2[ii])))

        return v1, v2


    def scatter(self, particles):
        p0c = self.manager.ref_particle.p0c

        p1, p2 = self._get_momenta(particles[0], particles[1])

        v1, v2 = self._get_fourmomenta(p1, p2)

        gamma1, gamma2 = [], []
        for ii in range(self.npart_over_two):
            gamma1.append(v1[ii].gamma)
            gamma2.append(v2[ii].gamma)
        self.gamma.append(np.array(gamma1))
        self.gamma.append(np.array(gamma2))

        # Step 1: Calculate the total four-vector in the lab frame
        vtot = []
        for ii in range(len(v1)):
            vtot.append(v1[ii] + v2[ii])

        # Step 2: Compute the boost vector to the center-of-mass frame
        boost_to_cm = []
        for ii in range(len(vtot)):
            boost_to_cm.append(vtot[ii].boost_vector())

        for ii in range(self.npart_over_two):
            v1[ii].boost(-boost_to_cm[ii])
            v2[ii].boost(-boost_to_cm[ii])

        gamma1_cm, gamma2_cm = [], []
        for ii in range(self.npart_over_two):
            gamma1_cm.append(v1[ii].gamma)
            gamma2_cm.append(v2[ii].gamma)
        self.gamma_cm.append(np.array(gamma1_cm))
        self.gamma_cm.append(np.array(gamma2_cm))

        # Sample theta and phi in the cm frame
        self.theta_cm = np.random.uniform(0, np.pi/2, self.npart_over_two)
        phi_cm = np.random.uniform(0, 2*np.pi, self.npart_over_two)

        self.moller_dcs = self._compute_moller_dcs(particles, self.theta_cm)

        # Apply the scattering angle
        for ii, theta, phi in zip(range(self.npart_over_two), self.theta_cm, phi_cm):
            v1[ii].rotate(theta, phi)
            v2[ii].rotate(theta, phi)

        # Boost back to the lab frame
        for ii in range(self.npart_over_two):
            v1[ii].boost(boost_to_cm[ii])
            v2[ii].boost(boost_to_cm[ii])

        particles[0].px = [v1[ii].px / p0c for ii in range(self.npart_over_two)]
        particles[0].py = [v1[ii].py / p0c for ii in range(self.npart_over_two)]
        particles[0].delta = [(np.sqrt(v1[ii].px**2 + v1[ii].py**2 + v1[ii].pz**2) - p0c) / p0c for ii in range(self.npart_over_two)]

        particles[1].px = [v2[ii].px / p0c for ii in range(self.npart_over_two)]
        particles[1].py = [v2[ii].py / p0c for ii in range(self.npart_over_two)]
        particles[1].delta = [(np.sqrt(v2[ii].px**2 + v2[ii].py**2 + v2[ii].pz**2) - p0c) / p0c for ii in range(self.npart_over_two)]

        self.particles = particles


    def _compute_moller_dcs(self, theta_cm):
        def _moller_dcs(gamma_cm, beta_cm, theta_cm):
            return CLASSICAL_ELECTRON_RADIUS**2  / 4 * gamma_cm**-2 * ((1 + beta_cm**-2)**2 * (4 - 3*np.sin(theta_cm)**2) * np.sin(theta_cm)**4 + 4*np.sin(theta_cm)**-2 + 1)

        moller_dcs1, moller_dcs2 = [], []
        for ii in range(self.npart_over_two):
            gamma_cm = self.gamma_cm[0][ii]
            beta_cm = np.sqrt(1 - gamma_cm**-2)
            moller_dcs1.append(_moller_dcs(gamma_cm, beta_cm, theta_cm[ii]))

            gamma_cm = self.gamma_cm[1][ii]
            beta_cm = np.sqrt(1 - gamma_cm**-2)
            moller_dcs2.append(_moller_dcs(gamma_cm, beta_cm, theta_cm[ii]))
        
        return [np.array(moller_dcs1), np.array(moller_dcs2)]


    def _compute_local_scattering_rate(self, phase_space_volume, phase_space_density):
        n_part = self.manager.n_part
        # v_prime_cm are the velocities of the particles after scattering in the cm frame
        v1_cm, v2_cm = [], []
        for ii in range(self.npart_over_two): 
            beta1_cm = np.sqrt(1 - self.gamma_cm[0][ii]**-2)
            v1_cm.append(beta1_cm * C_LIGHT_VACUUM)
            beta2_cm = np.sqrt(1 - self.gamma_cm[1][ii]**-2)
            v2_cm.append(beta2_cm * C_LIGHT_VACUUM)
        v_cm = [np.array(v1_cm), np.array(v2_cm)]
        # moller_dcs is the Moller cross section in the cm frame
        moller_dcs = self._compute_moller_dcs(self.theta_cm)

        local_scattering_rate1, local_scattering_rate2 = [], []
        for ii in range(self.npart_over_two):
             local_scattering_rate1.append(phase_space_volume / n_part * v_cm[0][ii] * self.gamma[0][ii]**-2 * moller_dcs[0][ii] * np.sin(self.theta_cm[ii]) * phase_space_density[0][ii] * phase_space_density[1][ii])
             local_scattering_rate2.append(phase_space_volume / n_part * v_cm[1][ii] * self.gamma[1][ii]**-2 * moller_dcs[1][ii] * np.sin(self.theta_cm[ii]) * phase_space_density[0][ii] * phase_space_density[1][ii])

        return [np.array(local_scattering_rate1), np.array(local_scattering_rate2)]


    def _compute_piwinski_total_scattering_rate(self, element):
        # TODO: if element is thick, use the average optical functions
        p0c = self.manager.ref_particle.p0c
        beta0 = self.manager.ref_particle.beta0[0]
        gamma0 = self.manager.ref_particle.gamma0[0]
        n_part = self.manager.n_part
        delta_min = self.manager.delta_min
        gemitt_x = self.manager.nemitt_x / beta0 / gamma0
        sigma_x = np.sqrt(self.twiss['betx', element] * gemitt_x)
        gemitt_y = self.manager.nemitt_y / beta0 / gamma0
        sigma_y = np.sqrt(self.twiss['bety', element] * gemitt_y)
        sigma_z = self.manager.sigma_z
        sigma_delta = self.manager.sigma_delta
        delta = self.twiss['delta', element]
        dx = self.twiss['dx', element]
        dy = self.twiss['dy', element]

        p = p0c * (1 + delta)
        gamma = np.sqrt(1 + p**2 / ELECTRON_MASS_EV**2)
        beta = np.sqrt(1 - gamma**-2)

        tau_m = beta**2 * delta_min**2

        f_factor_piwinski = self._compute_f_factor_piwinski()

        rate = CLASSICAL_ELECTRON_RADIUS**2 * C_LIGHT_VACUUM * n_part \
                / (8*np.pi * gamma**2 * sigma_z * np.sqrt(sigma_x**2 * sigma_y**2 - sigma_delta**4 * dx**2 * dy**2) * tau_m) \
                * f_factor_piwinski

        return rate


    def _assign_piwinski_total_scattering_rates(self):
        line = self.manager.line

        ii_tmarker = []
        for ii, nn in enumerate(line.element_names):
            if nn.startswith('TMarker_'):
                ii_tmarker.append(ii)

        ii_current_tmarker = 0
        integrated_rate = 0
        for ii, ee in enumerate(line.elements):
            if ii < ii_tmarker[ii_current_tmarker]:
                if hasattr(ee, 'length'):
                    rate = self._compute_piwinski_total_scattering_rate(ee)
                    integrated_rate += rate * ee.length
                else:
                    continue
            else:
                self.integrated_piwinski_total_scattering_rates[f'TMarker_{ii_current_tmarker}'] = integrated_rate
                # Reset the inegrated rate for the next Touschek marker
                integrated_rate = 0
                ii_current_tmarker += 1


    def compute_total_scattering_rate(self, phase_space_volume, phase_space_density):
        p0c = self.manager.ref_particle.p0c
        n_part = self.manager.n_part
        delta_min = self.manager.delta_min

        local_scattering_rate = self._compute_local_scattering_rate(phase_space_volume, phase_space_density)

        mask1_delta = abs(self.particles[0].delta) > self.manager.delta_min
        mask2_delta = abs(self.particles[1].delta) > self.manager.delta_min
        total_scattering_rate_mc = phase_space_volume / self.npart * np.sum(local_scattering_rate[0][mask1_delta] + local_scattering_rate[1][mask2_delta])

        piwinski_total_scattering_rate = self._compute_piwinski_total_scattering_rate()

        integrated_piwinski_total_scattering_rate = self.integrated_piwinski_total_scattering_rates[self.element]

        total_scattering_rate1 = local_scattering_rate[0] / np.sum(local_scattering_rate[0]) * total_scattering_rate_mc / piwinski_total_scattering_rate * integrated_piwinski_total_scattering_rate
        total_scattering_rate2 = local_scattering_rate[1] / np.sum(local_scattering_rate[1]) * total_scattering_rate_mc / piwinski_total_scattering_rate * integrated_piwinski_total_scattering_rate

        total_scattering_rate = np.concatenate((total_scattering_rate1, total_scattering_rate2))

        return total_scattering_rate
    

class TouschekElement():
    def __init__(self, particles, total_scattering_rate, manager):
        self.iscollective = False
        self.isthick = False
        self.particles = particles
        self.total_scattering_rate = total_scattering_rate
        self.manager = manager

class TouschekManager:
    def __init__(self, line, nemitt_x, nemitt_y, sigma_z, sigma_delta, n_elem, n_part, delta_min=0.005):
        self.line = line
        self.ref_particle = line.particle_ref
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.sigma_z = sigma_z
        self.sigma_delta = sigma_delta
        self.n_elem = n_elem
        self.n_part = n_part
        self.delta_min = delta_min
        self.s = []
        self.touschek_elems = []
        self.touschek = TouschekCalculator(self)


    def _get_s_elements_to_insert(self):
        ds = self.line.get_length() / self.n_elem
        s = np.linspace(ds, self.line.get_length(), self.n_elem)
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
        touschek_markers_dict = {}
        for ii in range(self.n_elem):
            touschek_markers_dict[f'TMarker_{ii}'] = xt.Marker()

        self.line._insert_thin_elements_at_s([(s_elem, [(key, touschek_markers_dict[key])]) for s_elem, key in zip(s, touschek_markers_dict.keys())])


    def _generate_particles(self, twiss, element):
        n_part_over_two = int(self.n_part / 2)
        # Particle object 1
        x_norm1, px_norm1 = xp.generate_2D_gaussian(n_part_over_two)
        y_norm1, py_norm1 = xp.generate_2D_gaussian(n_part_over_two)

        # The longitudinal closed orbit needs to be manually supplied for now
        element_index = line.element_names.index(element)
        zeta_co = twiss.zeta[element_index] 
        delta_co = twiss.delta[element_index] 

        zeta1, delta1 = xp.generate_longitudinal_coordinates(
                            line=self.line,
                            num_particles=self.npart_over_two, distribution='gaussian',
                            sigma_z=self.sigma_z, particle_ref=self.ref_particle)

        particles1 = self.line.build_particles(
                            particle_ref=self.ref_particle,
                            x_norm=x_norm1, px_norm=px_norm1,
                            y_norm=y_norm1, py_norm=py_norm1,
                            zeta=zeta1 + zeta_co,
                            delta=delta1 + delta_co,
                            nemitt_x=self.nemitt_x,
                            nemitt_y=self.nemitt_y
        )

        # Particle object 2
        # The particles are colliding  (x2=x1, y2=y1, zeta2=zeta1)
        x_norm2 = x_norm1
        _, px_norm2 = xp.generate_2D_gaussian(n_part_over_two)
        y_norm2 = y_norm1
        _, py_norm2 = xp.generate_2D_gaussian(n_part_over_two)
        zeta2 = zeta1
        _, delta2 = xp.generate_longitudinal_coordinates(
                            line=self.line,
                            num_particles=self.npart_over_two, distribution='gaussian',
                            sigma_z=self.sigma_z, particle_ref=self.ref_particle)


        particles2 = self.line.build_particles(
                            particle_ref=self.ref_particle,
                            x_norm=x_norm2, px_norm=px_norm2,
                            y_norm=y_norm2, py_norm=py_norm2,
                            zeta=zeta2 + zeta_co,
                            delta=delta2 + delta_co,
                            nemitt_x=self.nemitt_x,
                            nemitt_y=self.nemitt_y
        )

        return [particles1, particles2]
    

    def _compute_phase_space_density(self, particles):
        n_part_over_two = int(self.n_part / 2)
        
        data = np.vstack((particles.x,
                            particles.px,
                            particles.y,
                            particles.py,
                            particles.zeta,
                            particles.delta))

        kde = gaussian_kde(data)

        phase_space_density = []
        for ii in range(n_part_over_two):
            phase_space_point = np.array([
                particles.x[ii],
                particles.px[ii],
                particles.y[ii],
                particles.py[ii],
                particles.zeta[ii],
                particles.delta[ii]
            ])
            phase_space_density.append(kde(phase_space_point))

        return np.array(phase_space_density)
    

    def _compute_phase_space_volume(self, twiss, element):
        gemitt_x = self.nemitt_x / self.ref_particle.beta0[0] / self.ref_particle.gamma0[0] 
        sigma_x = np.sqrt(twiss['betx', element] * gemitt_x)

        gemitt_y = self.nemitt_y / self.ref_particle.beta0[0] / self.ref_particle.gamma0[0]
        sigma_y = np.sqrt(twiss['bety', element] * gemitt_y)

        sigma_z = self.sigma_z

        # NOTE: sigma_z / sigma_delta = beta_s
        phase_space_volume = np.sqrt(twiss['betx', element]*twiss['bety', element]*(self.sigma_z / self.sigma_delta)) * \
                                     (2*sigma_x)**3 * (2*sigma_y)**3 * (2*sigma_z)**3 
        
        return phase_space_volume


    def initialise_touschek_elems(self):
        self.s = self._get_s_elements_to_insert()
        self._install_touschek_markers(self.s)

        self.line.build_tracker()
        twiss = self.line.twiss()
        self.line.discard_tracker()

        # Pass the twiss table to the TouschekCalculator
        self.touschek.twiss = twiss

        # Assign the Piwikinski total scattering rate to each element with a length 
        # and to each Touschek marker
        self.touschek._assign_piwinski_total_scattering_rates()

        for ii in range(self.n_elem):
            # Pass the name of the Touschek marker to the TouschekCalculator
            self.touschek.element = f'TMarker_{ii}'

            particles = self._generate_particles(twiss, f'TMarker_{ii}')

            phase_space_density = []
            phase_space_density.append(self._compute_phase_space_density(particles[0]))
            phase_space_density.append(self._compute_phase_space_density(particles[1]))

            self.touschek.scatter(particles)

            phase_space_volume = self._compute_phase_space_volume(twiss, f'TMarker_{ii}')

            total_scattering_rate = self.touschek.compute_total_scattering_rate(phase_space_volume, phase_space_density)

            # particles is a list of two particle objects: merge the two particle objects into one particle object
            particles = xt.Particles.merge(particles)

            self.touschek_elems.append(TouschekElement(particles, total_scattering_rate, self))


    def install_touschek_elements(self):
        s = self._get_s_elements_to_insert()
        # self._install_touschek_markers(s)

        touschek_elem_dict = {}
        for ii in range(self.n_elem):
            touschek_elem_dict[f'Touschek_{ii}'] = self.touschek_elems[ii]
        
        elements_to_insert = [(s_elem, [(key, touschek_elem_dict[key])]) for s_elem, key in zip(s, touschek_elem_dict.keys())]

        self.line._insert_thin_elements_at_s(elements_to_insert)
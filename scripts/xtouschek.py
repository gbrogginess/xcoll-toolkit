import xtrack as xt
import xpart as xp
import xcoll as xc
import numpy as np
from scipy.integrate import quad
from scipy.special import i0
from scipy.constants import physical_constants

ELECTRON_MASS_EV = xt.ELECTRON_MASS_EV
C_LIGHT_VACUUM = physical_constants['speed of light in vacuum'][0]
CLASSICAL_ELECTRON_RADIUS = physical_constants['classical electron radius'][0]


class LorentzVector:
    def __init__(self, px=0, py=0, pz=0, e=0):
        self.vec = np.array([px, py, pz, e])


    def momvec(self):
        return np.array([self.vec[0], self.vec[1], self.vec[2]])


    def boost_vector(self):
        """Compute the boost vector for this Lorentz vector (momentum/energy if energy != 0)."""
        if self.vec[3] != 0:
            return self.vec[:3] / self.vec[3]
        else:
            return np.array([0, 0, 0])

    # def boost(self, bst):
    #     """Boost this Lorentz vector by the given boost vector."""
    #     bx, by, bz = bst
    #     b2 = bx**2 + by**2 + bz**2
    #     gamma = np.sqrt(1 - b2)**-1
    #     bp = bx * self.vec[0] + by * self.vec[1] + bz * self.vec[2]

    #     # Build the Lorentz transformation matrix
    #     LL = np.array([
    #         [gamma, gamma*bx, gamma*by, gamma*bz],
    #         [gamma*bx, 1+(gamma-1)*bz**2 / b2, (gamma-1)*bx*by / b2, (gamma-1)*bx*bz / b2],
    #         [gamma*by, (gamma-1)*by*bz / b2, 1+(gamma-1)*by**2 / b2, (gamma-1)*by*bz / b2],
    #         [gamma*bz, (gamma-1)*bx*bz / b2, (gamma-1)*by*bz / b2, 1+(gamma-1)*bz**2 / b2]
    #     ])

    #     # Apply the Lorentz transformation
    #     boosted_vec = LL @ self.vec

    #     return LorentzVector(boosted_vec[0], boosted_vec[1], boosted_vec[2], boosted_vec[3])
    

    def lab_to_cm(self, bst):
        bx, by, bz = bst
        b2 = np.linalg.norm(bst)**2
        gamma = np.sqrt(1 - b2)**-1

        bp = bx * self.px + by * self.py + bz * self.pz

        qx = self.px + (gamma - 1) * bp / b2 * bx - gamma * self.e * bx
        qy = self.py + (gamma - 1) * bp / b2 * by - gamma * self.e * by
        qz = self.pz + (gamma - 1) * bp / b2 * bz - gamma * self.e * bz

        q2 = qx**2 + qy**2 + qz**2
        e = np.sqrt(q2 + ELECTRON_MASS_EV**2)

        return LorentzVector(qx, qy, qz, e)


    def rotate(self, theta, phi):
        """Rotate the spatial components by angles theta (polar) and phi (azimuthal)."""
        p = self.momvec()
        p_abs = np.linalg.norm(p)

        # Compute spherical angles of v0
        th = np.arccos(self.pz / p_abs)
        ph = np.arctan2(self.py, self.px)

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
        self.vec[0] = p_abs * (s1 * c2 * x0 - s2 * y0 - c1 * c2 * z0)
        self.vec[1] = p_abs * (s1 * s2 * x0 + c2 * y0 - c1 * s2 * z0)
        self.vec[2] = p_abs * (c1 * x0 + s1 * z0)


    def cm_to_lab(self, bst):
        bx, by, bz = bst
        b2 = np.linalg.norm(bst)**2
        gamma = np.sqrt(1 - b2)**-1

        bq = bx * self.px + by * self.py + bz * self.pz

        px1 = self.px + gamma * bx * self.e + (gamma - 1) * bq / b2 * bx
        py1 = self.py + gamma * by * self.e + (gamma - 1) * bq / b2 * by
        pz1 = self.pz + gamma * bz * self.e + (gamma - 1) * bq / b2 * bz
        e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + ELECTRON_MASS_EV**2)

        px2 = -self.px + gamma * bx * self.e - (gamma - 1) * bq / b2 * bx
        py2 = -self.py + gamma * by * self.e - (gamma - 1) * bq / b2 * by
        pz2 = -self.pz + gamma * bz * self.e - (gamma - 1) * bq / b2 * bz
        e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + ELECTRON_MASS_EV**2)

        return LorentzVector(px1, py1, pz1, e1), LorentzVector(px2, py2, pz2, e2) 

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
    def beta(self):
        p_abs = np.linalg.norm(self.momvec())
        return p_abs / np.sqrt(p_abs**2 + ELECTRON_MASS_EV**2)
    
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
        self.gamma = []
        self.theta_cm = None
        self.gamma_cm = []
        self.integrated_piwinski_total_scattering_rates = {}
    

    def _etot_from_momvec(self, p):
        p = np.linalg.norm(p)
        return np.sqrt(p**2 + ELECTRON_MASS_EV**2)


    def _compute_pzeta_from_delta(self, particles):
        px = particles.px
        py = particles.py
        delta = particles.delta

        return np.sqrt((1 + delta)**2 - px**2 - py**2)


    def _get_momenta(self, particles1, particles2):
        pz_1 = self._compute_pzeta_from_delta(particles1)
        pz_2 = self._compute_pzeta_from_delta(particles2)

        p1 = particles1.p0c[0] * np.column_stack((particles1.px, particles1.py, pz_1))
        p2 = particles2.p0c[0] * np.column_stack((particles2.px, particles2.py, pz_2))

        return p1, p2


    def _get_fourmomenta(self, p1, p2):
        v1, v2 = [], []
        for ii in range(len(p1)):
            v1.append(LorentzVector(p1[ii][0], p1[ii][1], p1[ii][2], self._etot_from_momvec(p1[ii])))
            v2.append(LorentzVector(p2[ii][0], p2[ii][1], p2[ii][2], self._etot_from_momvec(p2[ii])))

        return v1, v2


    def scatter(self, particles):
        self.gamma = []
        self.gamma_cm = []

        p0c = self.manager.ref_particle.p0c[0]
        npart_over_two = self.npart_over_two

        p1, p2 = self._get_momenta(particles[0], particles[1])

        v1, v2 = self._get_fourmomenta(p1, p2)

        vtot = np.array([v1[ii] + v2[ii] for ii in range(len(v1))])
        boost_to_cm = np.array([v.boost_vector() for v in vtot])

        gamma = []
        for ii in range(npart_over_two):
            b2 = np.linalg.norm(boost_to_cm[ii])**2
            gamma.append(np.sqrt(1 - b2)**-1)
        self.gamma.extend([np.array(gamma), np.array(gamma)])
        
        q = []
        for ii in range(npart_over_two):
            q.append(v1[ii].lab_to_cm(boost_to_cm[ii]))

        gamma_cm = np.array([qq.gamma for qq in q])

        self.gamma_cm.extend([gamma_cm, gamma_cm])

        # Sample theta and phi in the cm frame
        self.theta_cm = np.random.uniform(0, np.pi/2, self.npart_over_two)
        phi_cm = np.random.uniform(0, np.pi, self.npart_over_two)

        # Apply the scattering angle
        for ii in range(npart_over_two):
            q[ii].rotate(self.theta_cm[ii], phi_cm[ii])

        # Boost back to the lab frame
        for ii in range(npart_over_two):
            v1[ii], v2[ii] = q[ii].cm_to_lab(boost_to_cm[ii])

        # Compute Moller DCS
        self.moller_dcs = self._compute_moller_dcs(self.theta_cm)

        particles[0].px = np.array([v1[ii].px / p0c for ii in range(self.npart_over_two)])
        particles[0].py = np.array([v1[ii].py / p0c for ii in range(self.npart_over_two)])
        particles[0].delta = np.array([(np.sqrt(v1[ii].px**2 + v1[ii].py**2 + v1[ii].pz**2) - p0c) / p0c 
                                    for ii in range(self.npart_over_two)])
        
        particles[1].px = np.array([v2[ii].px / p0c for ii in range(self.npart_over_two)])
        particles[1].py = np.array([v2[ii].py / p0c for ii in range(self.npart_over_two)])
        particles[1].delta = np.array([(np.sqrt(v2[ii].px**2 + v2[ii].py**2 + v2[ii].pz**2) - p0c) / p0c 
                                    for ii in range(self.npart_over_two)])

        self.particles = particles


    def _compute_moller_dcs(self, theta_cm):
        def _moller_dcs(gamma_cm, beta_cm, theta_cm):
            return CLASSICAL_ELECTRON_RADIUS**2 / (4 * gamma_cm**2) * ((1 + beta_cm**-2)**2 * (4 - 3*np.sin(theta_cm)**2) / (np.sin(theta_cm)**4) + 4/(np.sin(theta_cm)**2) + 1)

        gamma_cm1 = self.gamma_cm[0]
        gamma_cm2 = self.gamma_cm[1]

        beta_cm1 = np.sqrt(1 - gamma_cm1**-2)
        beta_cm2 = np.sqrt(1 - gamma_cm2**-2)

        moller_dcs1 = _moller_dcs(gamma_cm1, beta_cm1, theta_cm)
        moller_dcs2 = _moller_dcs(gamma_cm2, beta_cm2, theta_cm)
        
        return [moller_dcs1, moller_dcs2]


    def _compute_local_scattering_rate(self, phase_space_volume, phase_space_density):
        npart_over_two = self.manager.n_part / 2
        # v_prime_cm are the velocities of the particles after scattering in the cm frame
        v1_cm, v2_cm = [], []
        for ii in range(self.npart_over_two): 
            beta1_cm = np.sqrt(1 - self.gamma_cm[0][ii]**-2)
            v1_cm.append(beta1_cm * C_LIGHT_VACUUM)
            beta2_cm = np.sqrt(1 - self.gamma_cm[1][ii]**-2)
            v2_cm.append(beta2_cm * C_LIGHT_VACUUM)
        v_cm = [np.array(v1_cm), np.array(v2_cm)]

        # TODO: Check in detail that V/N with N being the number of scattering events is correct
        local_scattering_rate1, local_scattering_rate2 = [], []
        for ii in range(self.npart_over_two):
             local_scattering_rate1.append(phase_space_volume / npart_over_two * v_cm[0][ii] * self.gamma[0][ii]**-2 * self.moller_dcs[0][ii] * np.sin(self.theta_cm[ii]) * phase_space_density[0][ii] * phase_space_density[1][ii])
             local_scattering_rate2.append(phase_space_volume / npart_over_two * v_cm[1][ii] * self.gamma[1][ii]**-2 * self.moller_dcs[1][ii] * np.sin(self.theta_cm[ii]) * phase_space_density[0][ii] * phase_space_density[1][ii])

        return [np.array(local_scattering_rate1), np.array(local_scattering_rate2)]


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

        piwinski_integral = val * 2

        return piwinski_integral


    def _compute_piwinski_total_scattering_rate(self, element):
        # TODO: if element is thick, use the average optical functions
        p0c = self.manager.ref_particle.p0c[0]
        beta0 = self.manager.ref_particle.beta0[0]
        gamma0 = self.manager.ref_particle.gamma0[0]
        kb = self.manager.kb
        delta_min = self.manager.delta_min
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

        sigmab_x = np.sqrt(gemitt_x * betx) # Horizontal betatron beam size
        sigma_x = np.sqrt(gemitt_x * betx + dx**2 * sigma_delta**2) # Horizontal beam size

        sigmab_y = np.sqrt(gemitt_y * bety) # Vertical betatron beam size
        sigma_y = np.sqrt(gemitt_y * bety + dy**2 * sigma_delta**2) # Vertical beam size

        sigma_h = (sigma_delta**-2 + (dx**2 + dxt**2)/sigmab_x**2 + (dy**2 + dyt**2)/sigmab_y**2)**(-0.5)

        p = p0c * (1 + delta)
        gamma = np.sqrt(1 + p**2 / ELECTRON_MASS_EV**2)
        beta = np.sqrt(1 - gamma**-2)

        tm = beta**2 * delta_min**2

        b1 = betx**2 / (2 * beta**2 * gamma**2 * sigmab_x**2) * (1 - sigma_h**2 * dxt**2 / sigmab_x**2) \
             + bety**2 / (2 * beta**2 * gamma**2 * sigmab_y**2) * (1 - sigma_h**2 * dyt**2 / sigmab_y**2)
        
        b2 = np.sqrt(b1**2 - betx**2 * bety**2 * sigma_h**2 / (beta**4 * gamma**4 * sigmab_x**4 * sigmab_y**4 * sigma_delta**2) \
                             * (sigma_x**2 * sigma_y**2 - sigma_delta**4 * dx**2 * dy**2))

        piwinski_integral = self._compute_piwinski_integral(tm, b1, b2)

        rate = CLASSICAL_ELECTRON_RADIUS**2 * C_LIGHT_VACUUM * betx * bety * sigma_h * kb**2 \
                        / (8*np.sqrt(np.pi) * beta**2 * gamma**4 * sigmab_x**2 * sigmab_y**2 * sigma_z * sigma_delta) \
                        * piwinski_integral

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
                    rate = self._compute_piwinski_total_scattering_rate(line.element_names[ii])
                    integrated_rate += rate * ee.length
                else:
                    continue
            else:
                self.integrated_piwinski_total_scattering_rates[f'TMarker_{ii_current_tmarker}'] = integrated_rate
                # Reset the inegrated rate for the next Touschek marker
                integrated_rate = 0
                ii_current_tmarker += 1

                if ii_current_tmarker >= len(ii_tmarker):
                    break


    def compute_total_scattering_rate(self, phase_space_volume, phase_space_density):
        delta_min = self.manager.delta_min

        local_scattering_rate = self._compute_local_scattering_rate(phase_space_volume, phase_space_density)

        mask1_delta = abs(self.particles[0].delta) > delta_min
        mask2_delta = abs(self.particles[1].delta) > delta_min
        print(f'\nM = {sum(mask1_delta) + sum(mask2_delta)}\n')
        total_scattering_rate_mc = np.sum(local_scattering_rate[0][mask1_delta]) + np.sum(local_scattering_rate[1][mask2_delta])

        # This is OK (benchmarked with APS-ERL)
        piwinski_total_scattering_rate = self._compute_piwinski_total_scattering_rate(self.element)

        integrated_piwinski_total_scattering_rate = self.integrated_piwinski_total_scattering_rates[self.element]

        total_scattering_rate1 = local_scattering_rate[0] / np.sum(local_scattering_rate[0]) * total_scattering_rate_mc / piwinski_total_scattering_rate * integrated_piwinski_total_scattering_rate
        total_scattering_rate2 = local_scattering_rate[1] / np.sum(local_scattering_rate[1]) * total_scattering_rate_mc / piwinski_total_scattering_rate * integrated_piwinski_total_scattering_rate

        total_scattering_rate = np.concatenate((total_scattering_rate1, total_scattering_rate2))

        return total_scattering_rate_mc, piwinski_total_scattering_rate
    

class TouschekElement():
    def __init__(self, particles, total_scattering_rate, manager):
        self.iscollective = True
        self.isthick = False
        self.particles = particles
        self.total_scattering_rate = total_scattering_rate
        self.manager = manager

    def track(self, particles):
        return

class TouschekManager:
    def __init__(self, line, nemitt_x, nemitt_y, sigma_z, sigma_delta, n_elem, kb, n_part, delta_min=0.005, nx=3, ny=3, nz=3):
        self.line = line
        self.ref_particle = line.particle_ref
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.sigma_z = sigma_z
        self.sigma_delta = sigma_delta
        self.n_elem = n_elem
        self.kb = kb
        self.n_part = n_part
        self.delta_min = delta_min
        self.nx = nx
        self.ny = ny
        self.nz = nz
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
        print('\nInstalling Touschek markers...')
        touschek_markers_dict = {}
        for ii in range(self.n_elem):
            touschek_markers_dict[f'TMarker_{ii}'] = xt.Marker()

        self.line._insert_thin_elements_at_s([(s_elem, [(key, touschek_markers_dict[key])]) for s_elem, key in zip(s, touschek_markers_dict.keys())])

        print('Done.\n')


    def _generate_particles(self, twiss, element):
        n_part_over_two = int(self.n_part / 2)
        beta0 = self.ref_particle.beta0[0]
        gamma0 = self.ref_particle.gamma0[0]

        nx = self.nx
        ny = self.ny
        nz = self.nz

        sigma_z = self.sigma_z
        sigma_delta = self.sigma_delta

        gemitt_x = self.nemitt_x / beta0 / gamma0
        gemitt_y = self.nemitt_y / beta0 / gamma0   

        bets = sigma_z / sigma_delta
        gemitt_z = sigma_z**2 / bets 

        alfx = twiss['alfx', element]
        betx = twiss['betx', element]

        alfy = twiss['alfy', element]
        bety = twiss['bety', element]   

        dx = twiss['dx', element]
        dpx = twiss['dpx', element]

        dy = twiss['dy', element]
        dpy = twiss['dpy', element]     

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

        particles1 = xp.Particles(
            mass0=self.ref_particle.mass0,
            q0=self.ref_particle.q0,
            p0c=self.ref_particle.p0c[0],
            x=x1, px=px1,
            y=y1, py=py1,
            zeta=zeta1, delta=delta1
        )

        particles1.start_tracking_at_element = -1

        particles2 = xp.Particles(
            mass0=self.ref_particle.mass0,
            q0=self.ref_particle.q0,
            p0c=self.ref_particle.p0c[0],
            x=x2, px=px2,
            y=y2, py=py2,
            zeta=zeta2, delta=delta2
        )

        particles2.start_tracking_at_element = -1

        return [particles1, particles2]
    

    def _compute_phase_space_density(self, particles, element):
        kb = self.kb
        twiss = self.touschek.twiss

        alfx = twiss['alfx', element]
        betx = twiss['betx', element]
        alfy = twiss['alfy', element]
        bety = twiss['bety', element]
        dx = twiss['dx', element]
        dpx = twiss['dpx', element]
        dy = twiss['dy', element]
        dpy = twiss['dpy', element]

        beta0 = self.ref_particle.beta0[0]
        gamma0 = self.ref_particle.gamma0[0]
        gemitt_x = self.nemitt_x / beta0 / gamma0
        gemitt_y = self.nemitt_y / beta0 / gamma0

        sigmab_x = np.sqrt(gemitt_x * betx) # Horizontal betatron beam size
        sigmab_y = np.sqrt(gemitt_y * bety) # Vertical betatron beam size

        sigma_z = self.sigma_z
        sigma_delta = self.sigma_delta
        gemitt_z = sigma_z * sigma_delta

        x = particles.x - dx * particles.delta
        px = particles.px - dpx * particles.delta
        y = particles.y - dy * particles.delta
        py = particles.py - dpy * particles.delta
        zeta = particles.zeta
        delta = particles.delta

        phase_space_density = kb / (8 * np.pi**3 * gemitt_x * gemitt_y * gemitt_z) \
                                * np.exp(-zeta**2 / (2 * sigma_z**2) - delta**2 / (2 * sigma_delta**2)) \
                                * np.exp(-(x**2 + (alfx*x + betx*px)**2) / (2 * sigmab_x**2) \
                                         -(y**2 + (alfy*y + bety*py)**2) / (2 * sigmab_y**2))

        return np.array(phase_space_density)
    

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


    def initialise_touschek_elems(self):
        # self.s = self._get_s_elements_to_insert()
        self.s = [ 0.        ,  1.17777778,  2.35555556,  3.53333333,  4.71111111,
                   5.88888889,  7.06666667,  8.24444444,  9.42222222, 10.6       ]
        self._install_touschek_markers(self.s)

        self.line.build_tracker()
        twiss = self.line.twiss4d()

        # Pass the twiss table to the TouschekCalculator
        self.touschek.twiss = twiss

        # Assign the Piwikinski total scattering rate to each element with a length 
        # and to each Touschek marker
        print('\nComputing Piwinski total scattering rates...')
        self.touschek._assign_piwinski_total_scattering_rates()
        print('Done.\n')


        scattering_rates = {'s': self.s,
                            'mc': [],
                            'piwinski': []}
        for ii in range(self.n_elem):
            # Pass the name of the Touschek marker to the TouschekCalculator
            self.touschek.element = f'TMarker_{ii}'

            print(f'Generating particles at element {self.touschek.element}...')
            particles = self._generate_particles(twiss, self.touschek.element)

            print(f'Computing phase space density at element {self.touschek.element}...')
            phase_space_density = []
            phase_space_density.append(self._compute_phase_space_density(particles[0], self.touschek.element))
            phase_space_density.append(self._compute_phase_space_density(particles[1], self.touschek.element))

            print(f'Scattering particles at element {self.touschek.element}...')
            self.touschek.scatter(particles)

            print(f'Compute phase space volume at element {self.touschek.element}...')
            phase_space_volume = self._compute_phase_space_volume(twiss, self.touschek.element)

            # print(f'Computing total scattering rate at element {self.touschek.element}...')
            # total_scattering_rate = self.touschek.compute_total_scattering_rate(phase_space_volume, phase_space_density)
            # print('\n')

            print(f'Computing scattering rates at element {self.touschek.element}...')
            mc_scattering_rate, piwinski_scattering_rate = self.touschek.compute_total_scattering_rate(phase_space_volume, phase_space_density)
            print('\n')

            scattering_rates['mc'].append(mc_scattering_rate)
            scattering_rates['piwinski'].append(piwinski_scattering_rate)

            # # particles is a list of two particle objects: merge the two particle objects into one particle object
            # particles_merged = xt.Particles.merge(particles)

            # self.touschek_elems.append(TouschekElement(particles_merged, total_scattering_rate, self))

        self.line.discard_tracker()

        return scattering_rates


    def install_touschek_elements(self):
        print('Installing Touschek elements...')
        s = self._get_s_elements_to_insert()

        touschek_elem_dict = {}
        for ii in range(self.n_elem):
            touschek_elem_dict[f'Touschek_{ii}'] = self.touschek_elems[ii]
        
        elements_to_insert = [(s_elem, [(key, touschek_elem_dict[key])]) for s_elem, key in zip(s, touschek_elem_dict.keys())]

        self.line._insert_thin_elements_at_s(elements_to_insert)

        print('Done.\n')
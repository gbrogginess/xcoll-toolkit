"""
Python module for Monte Carlo simulation of Touschek effect in Xsuite.
=============================================
Author(s): Giacomo Broggi
Email:  giacomo.broggi@cern.ch
Date:   13-03-2025

This version aligns the Monte Carlo event generation with ELEGANT's TouschekDistribution
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
        # Keep a reference to the manager (beam/optics settings).
        self.manager = manager
        self.p0c = self.manager.ref_particle.p0c[0]  # reference momentum [eV]

        # to be set at initialise
        self.local_momentum_aperture = None
        self.twiss = None

        # scratch from the last scatter
        self.gamma_cm = None
        self.theta_cm = None
        self.beta0 = None

        # masks from last selection
        self.mask_PP1 = None
        self.mask_PP2 = None

        # bookkeeping
        self.element = None
        self.integrated_piwinski_total_scattering_rates = {}

    def _get_fourmomenta_matrix(self, PP):
        """
        PP columns: [x, px, y, py, zeta, delta]
        [ELEGANT: convert slopes to momenta and pz=(1+delta)*p0].
        """
        px = PP[:, 1] * self.p0c           # xp (slope) -> px (momentum)
        py = PP[:, 3] * self.p0c           # yp (slope) -> py (momentum)
        pz = (1.0 + PP[:, 5]) * self.p0c   # like Elegant: no px/py correction on pz

        p2 = px*px + py*py + pz*pz
        e  = np.sqrt(p2 + ELECTRON_MASS_EV**2)

        return np.column_stack((px, py, pz, e))

    def _get_boost_matrix(self, VV):
        PP = VV[:, :3]
        EE = VV[:, 3].reshape(-1, 1)
        BB = PP / EE     # beta = p/E

        return BB

    def _lab_to_cm(self, VV, BB):
        """
        Single-particle Lorentz boost to the CM defined by BB.
        [ELEGANT: bunch2cm outputs q = p1 boosted to CM]
        """
        bx, by, bz = BB[:, 0], BB[:, 1], BB[:, 2]
        px, py, pz, e = VV[:, 0], VV[:, 1], VV[:, 2], VV[:, 3]

        b2 = np.sum(BB**2, axis=1)
        b2 = np.clip(b2, 0.0, 1.0 - 1e-16)
        gamma = 1.0 / np.sqrt(1.0 - b2)
        bp = bx * px + by * py + bz * pz

        factor = np.where(b2 > 0.0, (gamma - 1.0) / b2, 0.0)

        qx = px + factor * bp * bx - gamma * e * bx
        qy = py + factor * bp * by - gamma * e * by
        qz = pz + factor * bp * bz - gamma * e * bz
        q2 = qx*qx + qy*qy + qz*qz
        e_star = np.sqrt(q2 + ELECTRON_MASS_EV**2)

        return np.column_stack((qx, qy, qz, e_star))

    def _rotate(self, QQ, theta, phi):
        """
        Rotate CM momentum by angles theta, phi.
        [ELEGANT: eulertrans]
        """
        px, py, pz = QQ[:, 0], QQ[:, 1], QQ[:, 2]
        p_abs = np.sqrt(px**2 + py**2 + pz**2)
        th = np.arccos(pz / p_abs)
        ph = np.arctan2(py, px)
        s1 = np.sin(th); s2 = np.sin(ph); c1 = np.cos(th); c2 = np.cos(ph)
        x0 = np.cos(theta)
        y0 = np.sin(theta) * np.cos(phi)
        z0 = np.sin(theta) * np.sin(phi)
        QQ[:, 0] = p_abs * (s1 * c2 * x0 - s2 * y0 - c1 * c2 * z0)
        QQ[:, 1] = p_abs * (s1 * s2 * x0 + c2 * y0 - c1 * s2 * z0)
        QQ[:, 2] = p_abs * (c1 * x0 + s1 * z0)

        return QQ

    def _cm_to_lab(self, QQ, BB):
        """
        Inverse boost from CM to LAB for both particles.
        [ELEGANT: cm2bunch]
        """
        bx, by, bz = BB[:, 0], BB[:, 1], BB[:, 2]
        qx, qy, qz, e = QQ[:, 0], QQ[:, 1], QQ[:, 2], QQ[:, 3]

        b2 = np.sum(BB**2, axis=1)
        b2 = np.clip(b2, 0.0, 1.0 - 1e-16)
        gamma = 1.0 / np.sqrt(1.0 - b2)
        bq = bx * qx + by * qy + bz * qz
        factor = np.where(b2 > 0.0, (gamma - 1.0) / b2, 0.0)

        px1 = qx + gamma * bx * e + factor * bq * bx
        py1 = qy + gamma * by * e + factor * bq * by
        pz1 = qz + gamma * bz * e + factor * bq * bz
        e1  = np.sqrt(px1*px1 + py1*py1 + pz1*pz1 + ELECTRON_MASS_EV**2)

        px2 = -qx + gamma * bx * e - factor * bq * bx
        py2 = -qy + gamma * by * e - factor * bq * by
        pz2 = -qz + gamma * bz * e - factor * bq * bz
        e2  = np.sqrt(px2*px2 + py2*py2 + pz2*pz2 + ELECTRON_MASS_EV**2)

        return (np.column_stack((px1, py1, pz1, e1)),
                np.column_stack((px2, py2, pz2, e2)))

    def _compute_moller_shape(self, theta_cm, beta0):
        """
        [ELEGANT: moeller()]
        cross = (1 - b^2) * ( (1+1/b^2)^2*(4/sin^2^2 - 3/sin^2) + 1 + 4/sin^2 )
        NB: here b = beta0 (CM speed of outgoing particle / c).
        """
        st2 = np.sin(theta_cm)**2
        b2  = beta0**2
        return (1.0 - b2) * ( ((1.0 + 1.0/b2)**2) * (4.0/(st2**2) - 3.0/st2) + 1.0 + 4.0/st2 )

    def _compute_piwinski_integral(self, tm, b1, b2):
        km = np.arctan(np.sqrt(tm))

        def int_piwinski(k, km, B1, B2):
            t = np.tan(k) ** 2
            tm = np.tan(km) ** 2
            fact = (
                (2*t + 1)**2 * (t/tm / (1+t) - 1) / t + t - np.sqrt(t*tm * (1 + t))
                - (2 + 1 / (2*t)) * np.log(t/tm / (1+t))
            )
            if B2 * t < 500:
                intp = fact * np.exp(-B1*t) * i0(B2*t) * np.sqrt(1+t)
            else:
                intp = (
                    fact
                    * np.exp(B2*t - B1*t)
                    / np.sqrt(2*np.pi * B2*t)
                    * np.sqrt(1+t)
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

        B1 = betx**2 / (2 * beta**2 * gamma**2 * sigmab_x**2) * (1 - sigma_h**2 * dxt**2 / sigmab_x**2) \
             + bety**2 / (2 * beta**2 * gamma**2 * sigmab_y**2) * (1 - sigma_h**2 * dyt**2 / sigmab_y**2)

        B2 = np.sqrt(B1**2 - betx**2 * bety**2 * sigma_h**2 / (beta**4 * gamma**4 * sigmab_x**4 * sigmab_y**4 * sigma_delta**2) \
                             * (sigma_x**2 * sigma_y**2 - sigma_delta**4 * dx**2 * dy**2))

        tmN = beta**2 * (deltaN**2)
        tmP = beta**2 * (deltaP**2)

        piwinski_integralN = self._compute_piwinski_integral(tmN, B1, B2)
        piwinski_integralP = self._compute_piwinski_integral(tmP, B1, B2)

        rateN = CLASSICAL_ELECTRON_RADIUS**2 * C_LIGHT_VACUUM * kb**2 \
                / (8*np.pi * gamma**2 * sigma_z * np.sqrt(sigma_x**2 * sigma_y**2 - sigma_delta**4 * dx**2 * dy**2)) \
                * 2 * np.sqrt(np.pi * (B1**2 - B2**2)) * piwinski_integralN

        rateP = CLASSICAL_ELECTRON_RADIUS**2 * C_LIGHT_VACUUM * kb**2 \
                / (8*np.pi * gamma**2 * sigma_z * np.sqrt(sigma_x**2 * sigma_y**2 - sigma_delta**4 * dx**2 * dy**2)) \
                * 2 * np.sqrt(np.pi * (B1**2 - B2**2)) * piwinski_integralP

        rate = (rateN + rateP) / 2

        return rate

    def _assign_piwinski_total_scattering_rates(self):
        """
        Integrate local Piwinski rate along s using trapezoidal rule and
        store the per-TMarker *integrated* rate per bunch.
        """
        line = self.manager.line
        tab = line.get_table()
        T_rev0 = float(self.twiss.T_rev0)

        # List of TMarkers in line order
        ii_tm = [i for i, nm in enumerate(line.element_names) if nm.startswith('TMarker_')]
        if not ii_tm:
            raise RuntimeError("No TMarker found in the line.")

        integrated = 0.0
        s0 = float(tab.rows[tab.name == line.element_names[0]].s[0])
        r0 = self._compute_piwinski_total_scattering_rate(line.element_names[0])
        s_before = s0
        rate_before = r0
        ii_current_tm = 0

        for ii, nn in enumerate(line.element_names):
            s = float(tab.rows[tab.name == nn].s[0])
            ds = s - s_before
            if ds > 0.0:
                rate = self._compute_piwinski_total_scattering_rate(nn)
                integrated += 0.5 * (rate_before + rate) * ds
                s_before = s
                rate_before = rate

            if ii_current_tm < len(ii_tm) and ii == ii_tm[ii_current_tm]:
                # divide by c and by T_rev0 -> per-bunch rate
                self.integrated_piwinski_total_scattering_rates[
                    f'TMarker_{ii_current_tm}'
                ] = integrated / C_LIGHT_VACUUM / T_rev0
                integrated = 0.0
                ii_current_tm += 1
                if ii_current_tm >= len(ii_tm):
                    break

    # ========= Elegant-style single-attempt generator =========
    def _selectPartGauss_from_uniforms(self, ran1, twiss, element):
        """
        [ELEGANT: selectPartGauss()]
        Inputs:
          ran1: 11 uniform random numbers in [0,1), shuffled.

        Returns:
          p1, p2 : arrays [x, y, s->zeta, xp->px, yp->py, dp/p->delta] in Xsuite ordering (x,px,y,py,zeta,delta)
          dens1, dens2 : phase-space densities for the two particles.

        Note on ordering/units:
        - ELEGANT p = [x, y, s, xp, yp, dp/p]. Here we return Xsuite order.
        - Slopes xp,yp are dimensionless; later converted to momenta with multiplication by p0c.
        """
        beta0 = self.manager.ref_particle.beta0[0]
        gamma0 = self.manager.ref_particle.gamma0[0]

        nx = self.manager.nx
        ny = self.manager.ny
        nz = self.manager.nz

        sigma_z = self.manager.sigma_z
        sigma_delta = self.manager.sigma_delta

        gemitt_x = self.manager.nemitt_x / beta0 / gamma0
        gemitt_y = self.manager.nemitt_y / beta0 / gamma0
        gemitt_z = sigma_z * sigma_delta
        bets = sigma_z / sigma_delta

        alfx = twiss['alfx', element]; betx = twiss['betx', element]
        alfy = twiss['alfy', element]; bety = twiss['bety', element]
        dx   = twiss['dx', element];   dpx  = twiss['dpx', element]
        dy   = twiss['dy', element];   dpy  = twiss['dpy', element]

        # Normalized coordinates U, V1, V2 in the three planes
        # Map ran1[0:9] as in ELEGANT: U[i], V1[i], V2[i]
        U  = np.empty(3); V1 = np.empty(3); V2 = np.empty(3)
        emitN = np.array([gemitt_x, gemitt_y, gemitt_z])

        # [ELEGANT] U[i] = (ran1[i]-0.5) * range[i] * sqrt(emitN[i]/gamma)
        # Qui usiamo range = 2*cutoff = nx,ny,nz analoghi (già impostati in Manager)
        rng = np.array([2*nx, 2*ny, 2*nz], dtype=float)
        for i in range(3):
            U[i]  = (ran1[i]   - 0.5) * rng[i] * np.sqrt(emitN[i] / gamma0)
            V1[i] = (ran1[i+3] - 0.5) * rng[i] * np.sqrt(emitN[i] / gamma0)
            V2[i] = (ran1[i+6] - 0.5) * rng[i] * np.sqrt(emitN[i] / gamma0)

        # Densities (Gaussian factors)
        densa = np.exp(-0.5 * (U*U + V1*V1) / emitN * gamma0)
        densb = np.empty(3)
        densb[2] = np.exp(-0.5 * (U[2]*U[2] + V2[2]*V2[2]) / emitN[2] * gamma0)

        # Change from normalized to real phase space
        x1b = np.sqrt(betx) * U[0]
        px1b= (V1[0] - alfx*U[0]) / np.sqrt(betx)

        y1b = np.sqrt(bety) * U[1]
        py1b= (V1[1] - alfy*U[1]) / np.sqrt(bety)

        # Plane z (here we use (zeta, delta) with bets = sigma_z/sigma_delta)
        zeta1 = np.sqrt(bets) * U[2]
        zeta2 = zeta1
        delta1= V1[2] / np.sqrt(bets)
        delta2= V2[2] / np.sqrt(bets)

        # Dispersion correction for particle 1
        x1 = x1b + dx * delta1
        px1= px1b + dpx * delta1
        y1 = y1b + dy * delta1
        py1= py1b + dpy * delta1

        # Build particle 2:
        # First back out betatron x,y given particle-2 delta, then recompute slopes
        x2b = x1 - dx * delta2
        y2b = y1 - dy * delta2
        Ux2 = x2b / np.sqrt(betx)
        Uy2 = y2b / np.sqrt(bety)

        px2b= (V2[0] - alfx*Ux2) / np.sqrt(betx)
        py2b= (V2[1] - alfy*Uy2) / np.sqrt(bety)

        x2  = x1
        px2 = px2b + dpx * delta2
        y2  = y1
        py2 = py2b + dpy * delta2

        densb[0] = np.exp(-0.5 * (Ux2*Ux2 + V2[0]*V2[0]) / emitN[0] * gamma0)
        densb[1] = np.exp(-0.5 * (Uy2*Uy2 + V2[1]*V2[1]) / emitN[1] * gamma0)

        # Pack to Xsuite ordering [x, px, y, py, zeta, delta]
        p1 = np.array([x1, px1, y1, py1, zeta1, delta1], dtype=float)
        p2 = np.array([x2, px2, y2, py2, zeta2, delta2], dtype=float)

        dens1 = float(densa[0]*densa[1]*densa[2])
        dens2 = float(densb[0]*densb[1]*densb[2])

        return p1, p2, dens1, dens2

    # ========= Elegant-style event attempt =========
    def _attempt_event(self, twiss, element, rng):
        """
        Perform one 'attempt' as in ELEGANT's main while-loop in TouschekDistribution():
        - draw 11 uniforms and shuffle
        - selectPartGauss
        - bunch2cm / eulertrans / cm2bunch
        - order by delta
        - compute weight 'temp'
        - return up to two accepted single-particle events (coords, temp)

        Returns:
          accepted (list of (PP_single, temp)), total_did_something(bool)
        """
        # draw 11 uniforms and randomize order
        ran1 = rng.random(11)
        rng.shuffle(ran1)  # [ELEGANT: randomizeOrder]

        # select particles and densities (skip if zero density)
        p1, p2, dens1, dens2 = self._selectPartGauss_from_uniforms(ran1, twiss, element)
        if dens1 == 0.0 or dens2 == 0.0:
            return [], True  # attempt consumed RNG, nothing accepted

        # Convert to four-momenta
        PP1 = p1.reshape(1, -1)  # [x,px,y,py,zeta,delta]
        PP2 = p2.reshape(1, -1)  # [x,px,y,py,zeta,delta]
        VV1 = self._get_fourmomenta_matrix(PP1) # [Px, Py, Pz, E]
        VV2 = self._get_fourmomenta_matrix(PP2) # [Px, Py, Pz, E]

        # CM boost from total momentum
        VVsum = VV1 + VV2
        BB = self._get_boost_matrix(VVsum)
        b2 = float(np.sum(BB**2, axis=1)[0])
        b2 = np.clip(b2, 0.0, 1.0 - 1e-16)
        gamma_cm = 1.0 / np.sqrt(1.0 - b2)
        self.gamma_cm = np.array([gamma_cm])

        # boost particle 1 to CM
        QQ = self._lab_to_cm(VV1, BB)

        # Scattering angles in the CM frame
        theta = (ran1[9] * 0.9999 + 0.00005) * np.pi
        phi   = ran1[10] * np.pi
        self.theta_cm = np.array([theta])

        # rotate 3-momentum in CM
        QQ_rot = QQ.copy()
        QQ_rot = self._rotate(QQ_rot, np.array([theta]), np.array([phi]))

        # compute beta0 at CM (|q'|/E*)
        qabs = float(np.linalg.norm(QQ_rot[0, :3]))
        E_star = float(QQ_rot[0, 3])
        beta0 = qabs / E_star
        self.beta0 = np.array([beta0])

        # Møller cross section shape
        cross = self._compute_moller_shape(theta, beta0)

        # back to LAB to get scattered momenta
        VV1, VV2 = self._cm_to_lab(QQ_rot, BB)

        # Convert back to Xsuite coordinates
        p0c = self.p0c
        pp1 = np.empty(6)
        pp2 = np.empty(6)
        pp1[0] = PP1[0,0]; pp1[2] = PP1[0,2]; pp1[4] = PP1[0,4]
        pp1[1] = VV1[0,0] / p0c
        pp1[3] = VV1[0,1] / p0c
        pp1[5] = VV1[0,2] / p0c - 1.0

        pp2[0] = PP2[0,0]; pp2[2] = PP2[0,2]; pp2[4] = PP2[0,4]
        pp2[1] = VV2[0,0] / p0c
        pp2[3] = VV2[0,1] / p0c
        pp2[5] = VV2[0,2] / p0c - 1.0

        # like Elegant: ensure delta1 <= delta2 (swap if needed)
        if pp1[5] > pp2[5]:
            pp1, pp2 = pp2.copy(), pp1.copy()

        # acceptance on momentum aperture (scaled by fdelta in outer loop)
        fdelta = self.manager.fdelta
        deltaN = self.local_momentum_aperture[element][0] * fdelta
        deltaP = self.local_momentum_aperture[element][1] * fdelta

        # event base weight before later scaling to integrated rate
        temp = dens1 * dens2 * np.sin(theta)
        # [ELEGANT] temp *= cross * beta0 / gamma^2 (gamma here is gamma_cm)
        temp *= cross * beta0 / (gamma_cm**2)

        accepted = []
        if pp1[5] < deltaN:
            accepted.append( (pp1, float(temp)) )
        if pp2[5] > deltaP:
            accepted.append( (pp2, float(temp)) )

        return accepted, True

    # ========= Driver used by Manager at each TMarker (Elegant-like) =========
    def simulate_at_element_elegant_mode(self, element, rng, verbose=True):
        """
        Generate scattered single-particle events at one TMarker following ELEGANT's logic.
        Returns:
          PP (N,6), weights_raw (N,), mc_rate_like (sum of raw temps), integrated_piwinski_rate
        """
        twiss = self.twiss
        self.element = element

        # Prepare attempt loop.
        n_target = self.manager.n_part_mc  # ELEGANT: n_simulated (number to *keep*, not attempts)
        accepted_coords = []
        accepted_weights_temp = []

        total_event = 0

        while len(accepted_coords) < n_target:
            acc, _ = self._attempt_event(twiss, element, rng)
            total_event += 1
            if acc:
                for pp, temp in acc:
                    # store single-particle event
                    accepted_coords.append(pp)
                    accepted_weights_temp.append(temp)
                    if len(accepted_coords) >= n_target:
                        break

        if verbose:
            print(f'[Elegant-like] element={element}: attempts={total_event}, accepted={len(accepted_coords)}')

        PP = np.array(accepted_coords, dtype=float)
        temps = np.array(accepted_weights_temp, dtype=float)

        # mc_rate_like := sum of raw temps (equivalent to Elegant's totalWeight when factor cancels later)
        mc_rate_like = float(temps.sum())

        # integrated Piwinski rate for this marker (per bunch)
        integrated_rate = self.integrated_piwinski_total_scattering_rates[element]

        return PP, temps, mc_rate_like, integrated_rate


class TouschekManager:
    def __init__(self, line, local_momaper, nemitt_x, nemitt_y, sigma_z, sigma_delta, kb, n_part_mc, n_elems=None, fdelta=0.85, nx=3, ny=3, nz=3):
        self.line = line
        self.local_momaper = local_momaper
        self.ref_particle = line.particle_ref
        self.n_elems = n_elems
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.sigma_z = sigma_z
        self.sigma_delta = sigma_delta
        self.kb = kb
        self.n_part_mc = n_part_mc  # [ELEGANT: n_simulated] = number of *kept* single-particle events
        self.fdelta = fdelta
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.s = []
        self.touschek_dict = {}
        self.touschek = TouschekCalculator(self)

        # fraction of rate to ignore in pickPart (ELEGANT: ignored_portion)
        self.ignored_portion = getattr(self, "ignored_portion", 0.01)

        if n_elems is None:
            # Check that the line contains at least one TMarker
            tab = line.get_table()
            tmarker_names = tab.rows['TMarker_.*'].name
            if len(tmarker_names) == 0:
                raise ValueError('If the line does not contain any TMarker, please specify n_elems.')

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

        # mask s positions inside any collimator body; snap just outside
        mask_in_coll_region = np.zeros_like(s, dtype=bool)
        for us, ds in coll_regions:
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

    # ========= pickPart identical behavior =========
    def _pick_part_indices(self, weights, weight_limit, weight_ave):
        """
        Python translation of ELEGANT pickPart():
        Split heavy/light around average, pick heavies first until weight_limit, recurse on remainder.
        Returns (indices_selected, wTotalSelected)
        """
        indices = np.arange(len(weights))
        weights = weights.copy()

        selected = []
        wTotal = 0.0

        def recurse(start, end, weight_limit, weight_ave):
            nonlocal wTotal, selected, indices, weights

            N = end - start
            if N < 3:
                return

            w = weights[start:end]
            idx = indices[start:end]
            mask_heavy = w > weight_ave
            heavy_idx = idx[mask_heavy]
            light_idx = idx[~mask_heavy]
            heavy_w = w[mask_heavy]
            light_w = w[~mask_heavy]

            i2 = heavy_idx.size
            i1 = light_idx.size
            w2 = float(heavy_w.sum())
            w1 = float(light_w.sum())

            if w2 + wTotal > weight_limit:
                if i2 == 0:
                    return
                indices[start:start+i2] = heavy_idx
                weights[start:start+i2] = heavy_w
                new_weight_ave = w2 / i2
                recurse(start, start+i2, weight_limit, new_weight_ave)
                return

            selected.extend(heavy_idx.tolist())
            wTotal += w2

            if i1 == 0:
                return
            indices[start+i2:end] = light_idx
            weights[start+i2:end] = light_w
            new_weight_ave = w1 / i1
            recurse(start+i2, end, weight_limit, new_weight_ave)

        total_w = float(weights.sum())
        recurse(0, len(weights), weight_limit, weight_ave)

        return np.array(selected, dtype=int), wTotal

    # ========= public entry point =========
    def initialise_touschek(self, element=None, seed=None):
        """
        Build TMarkers (if requested), compute twiss and integrated Piwinski rates,
        then for each TMarker run the Elegant-like event loop and return particles
        with weights scaled so that their sum = integrated Piwinski rate at that marker.
        """
        if self.n_elems is not None:
            self.s = self._get_s_elements_to_insert()
            self._install_touschek_markers(self.s)
        else:
            tab = self.line.get_table()
            self.s = tab.rows['TMarker_.*'].s

        self.line.build_tracker()
        twiss = self.line.twiss4d()

        # Pass twiss and momentum aperture to calculator
        self.touschek.twiss = twiss
        self.touschek.local_momentum_aperture = self.local_momaper

        # Compute per-TMarker integrated rates
        print('\nComputing Piwinski total scattering rates...')
        self.touschek._assign_piwinski_total_scattering_rates()
        print('Done.\n')

        # If user passed a single TMarker name, restrict processing
        if element is not None:
            ii_touschek_element = int(re.search(r'\d+', element).group())
            self.s = [self.s[ii_touschek_element]]

        # RNG aligned usage (not bitwise equal to Elegant, but same consumption pattern)
        rng = np.random.default_rng(seed)

        for ii, ss in enumerate(self.s):
            marker_name = f'TMarker_{ii if element is None else ii_touschek_element}'
            self.touschek.element = marker_name

            print(f'Generating scattered particles at {marker_name} (Elegant-like loop)...')

            # Run Elegant-like attempt loop at this marker
            PP_all, temps_raw, mc_rate_like, integrated_rate = self.touschek.simulate_at_element_elegant_mode(
                marker_name, rng, verbose=True
            )

            # Subsample with pickPart (ELEGANT: ignoredPortion)
            total_w = float(temps_raw.sum())
            weight_limit = total_w * (1 - getattr(self, "ignored_portion", 0.01))
            weight_ave   = total_w / len(temps_raw)

            keep_idx, wTotal_kept = self._pick_part_indices(temps_raw, weight_limit, weight_ave)
            PP = PP_all[keep_idx, :]
            temps_kept = temps_raw[keep_idx]

            # Final scaling so that sum(weights) == integrated_rate (per-bunch, Piwinski)
            if temps_kept.sum() <= 0.0:
                raise RuntimeError("No accepted Touschek events after selection.")
            scale = integrated_rate / float(temps_kept.sum())
            total_scattering_rate = temps_kept * scale  # final particle weights

            n_part_to_track = len(total_scattering_rate)
            print(f'{n_part_to_track} particles to track at {marker_name}')

            # Build Xsuite Particles object with weights
            particles = xt.Particles(
                mass0=self.ref_particle.mass0,
                q0=self.ref_particle.q0,
                p0c=self.ref_particle.p0c[0],
                x=PP[:,0], px=PP[:,1],
                y=PP[:,2], py=PP[:,3],
                zeta=PP[:,4], delta=PP[:,5],
                s=np.ones(n_part_to_track) * ss,
                weight=total_scattering_rate,
                _capacity=2*n_part_to_track
            )

            # Also compute local Piwinski rate at the TMarker position (for diagnostics)
            piwinski_local = self.touschek._compute_piwinski_total_scattering_rate(marker_name)

            self.touschek_dict[marker_name] = {
                'particles': particles,
                'mc_rate_like': mc_rate_like,                 # sum of raw temps over accepted (pre-pickPart)
                'piwinski_rate_local': piwinski_local,        # local (not integrated) Piwinski rate at marker
                'integrated_rate': integrated_rate,           # ∫ r(s) ds / (c T_rev0)
                'mc_to_piwinski_ratio': mc_rate_like / integrated_rate if integrated_rate>0 else np.nan
            }

        self.line.discard_tracker()
        return
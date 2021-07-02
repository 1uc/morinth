# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np

from morinth.euler import Euler

class RoeAverage(object):
    def __init__(self, rho_left, rho_right):
        self.roe_ratio = np.sqrt(rho_right/rho_left)

    def __call__(self, qL, qR):
        return (qL + qR*self.roe_ratio)/(1.0 + self.roe_ratio)

class HLLC(object):
    """HLLC approximate Riemann solver."""

    def __init__(self, model):
        self.model = model

    def __call__(self, uL, uR, axis):
        model = self.model

        pL, vL, wL = self.derived_variables(uL, axis)
        pR, vR, wR = self.derived_variables(uR, axis)

        sL, s_star, sR = self.wave_speeds(uL, pL, vL, wL, uR, pR, vR, wR, axis)

        fast_cells = 0.0 <= s_star
        uK = np.where(fast_cells, uL, uR)
        vK = np.where(fast_cells, vL, vR)
        pK = np.where(fast_cells, pL, pR)

        fK = model.flux(uK, axis, p = pK)

        off_axis = self.off_axis(axis)
        sK = np.where(fast_cells, sL, sR)
        sK = np.where(np.logical_and(sL < 0.0, 0.0 <= sR), sK, 0.0) # sK == 0.0 -> no increment

        cK = (sK - vK)/(sK - s_star)
        fK[0,...] += sK*((cK - 1.0)*uK[0,...])
        fK[axis+1,...] += sK*(cK*uK[0,...]*s_star - uK[axis+1,...])
        fK[off_axis+1,...] += sK*((cK - 1.0)*uK[off_axis+1,...])
        fK[-1,...] += sK*(cK*(uK[-1,...] + (s_star-vK)*(uK[0,...]*s_star + pK/(sK-vK)))
                          - uK[-1,...])

        return fK

    def off_axis(self, axis):
        """Get the index of the other axis."""

        if axis == 0:
            return 1
        else:
            return 0

    def derived_variables(self, u, axis):
        model = self.model
        off_axis = self.off_axis(axis)

        p = model.pressure(u)
        v = u[axis+1,...]/u[0,...]
        w = u[off_axis+1,...]/u[0,...]

        return p, v, w

    def H(self, u, p):
        return (u[-1,...] + p)/u[0,...]

    def wave_speeds(self, uL, pL, vL, wL, uR, pR, vR, wR, axis):
        """Compute the three wave-speeds `sL, s_star, sR`, using Batten's estimates."""

        model = self.model
        roe_average = RoeAverage(uL[0,...], uR[0,...])

        aL, aR = model.sound_speed(uL, pL), model.sound_speed(uR, pR)
        v_tilda = roe_average(vL, vR)
        w_tilda = roe_average(wL, wR)

        HL, HR = self.H(uL, pL), self.H(uR, pR)
        H_tilda = roe_average(HL, HR)

        vroe_square = v_tilda**2 + w_tilda**2
        a_tilda = np.sqrt((model.eos.gamma - 1.0)*(H_tilda - 0.5*vroe_square))

        sL = np.minimum(vL - aL, v_tilda - a_tilda)
        sR = np.maximum(vR + aR, v_tilda + a_tilda)

        s_star = ( (uR[axis+1,...]*(sR - vR) - uL[axis+1,...]*(sL - vL) + pL - pR)
                 / (uR[0,...]*(sR - vR) - uL[0,...]*(sL - vL)))

        return sL, s_star, sR

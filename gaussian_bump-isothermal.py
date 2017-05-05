#!/usr/bin/env python3
# encoding: utf-8

from gaussian_bump import GaussianBumpConvergence, GaussianBumpReference
from gaussian_bump import GaussianBumpConvergenceRates

if __name__ == "__main__":
    sim = GaussianBumpConvergenceRates()
    sim(GaussianBumpConvergence, GaussianBumpReference, "isothermal")

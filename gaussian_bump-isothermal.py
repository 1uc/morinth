#!/usr/bin/env python3
# encoding: utf-8

# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

from gaussian_bump import GaussianBumpConvergence, GaussianBumpReference
from gaussian_bump import GaussianBumpConvergenceRates

if __name__ == "__main__":
    sim = GaussianBumpConvergenceRates()
    sim(GaussianBumpConvergence, GaussianBumpReference, "isothermal")

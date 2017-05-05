#! /usr/bin/env python3
# -*- encoding: utf-8 -*-

from wave_propagation import StrongWavePropagation, StrongWavePropagationReference
from wave_propagation import WavePropagationRates

if __name__ == "__main__":
    sim = WavePropagationRates()
    sim(StrongWavePropagation, StrongWavePropagationReference, "isentropic")

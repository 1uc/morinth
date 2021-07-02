# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import pytest

def pytest_addoption(parser):
    parser.addoption("--manual-mode",
                     action="store_true",
                     help="run tests which require manual validation.")

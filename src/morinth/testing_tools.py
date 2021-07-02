# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np

import pytest

@pytest.fixture(autouse=True)
def pytest_config(request):
    return request.config

def is_manual_mode(pytest_config):
    return pytest_config.getoption("--manual-mode")

def pytest_collection_modifyitems(config, items):
    mark_manual = pytest.mark.skipif(not is_manual_mode(config),
                                     reason="pass `--run-manual` to run this")

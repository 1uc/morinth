# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import os

def ensure_directory_exists(filename=None, dirname=None):
    if filename is not None:
        dirname =  os.path.dirname(filename)

    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


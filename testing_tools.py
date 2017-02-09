import numpy as np

import pytest

def is_manual_mode():
    return pytest.config.getoption("--manual-mode")

mark_manual = pytest.mark.skipif(not is_manual_mode(),
                                 reason="pass `--run-manual` to run this")

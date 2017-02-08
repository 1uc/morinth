import numpy as np

import pytest

mark_manual = pytest.mark.skipif(not pytest.config.getoption("--run-manual"),
                                 reason="pass `--run-manual` to run this")

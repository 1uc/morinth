import pytest

def pytest_addoption(parser):
    parser.addoption("--manual-mode",
                     action="store_true",
                     help="run tests which require manual validation.")

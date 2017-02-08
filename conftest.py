import pytest

def pytest_addoption(parser):
    parser.addoption("--run-manual",
                     action="store_true",
                     help="run tests which require manual validation.")

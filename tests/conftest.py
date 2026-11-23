import shutil
from pathlib import Path

import pytest

from geocouple.stack import load_stack


@pytest.fixture(scope="session")
def stack():
    return load_stack("configs/stack/default.yaml")


def _getdp_available() -> bool:
    try:
        from geocouple.tools import find_tools
        find_tools()
        return True
    except FileNotFoundError:
        return False


def pytest_collection_modifyitems(config, items):
    if _getdp_available():
        return
    skip = pytest.mark.skip(reason="GetDP not found")
    for item in items:
        if "fem" in item.keywords:
            item.add_marker(skip)

import pytest

from geocouple.stack import load_stack


@pytest.fixture(scope="session")
def stack():
    return load_stack("configs/stack/default.yaml")


def _getdp_available() -> bool:
    from geocouple.tools import find_tools
    try:
        find_tools()
        return True
    except FileNotFoundError:
        return False


def _fast_solvers_available() -> bool:
    from geocouple.tools import FAST_SOLVERS, find_fast_solver
    return all(find_fast_solver(n, required=False) is not None for n in FAST_SOLVERS)


def pytest_collection_modifyitems(config, items):
    """fem: needs GetDP; solvers: needs FastHenry + FastCap (scripts/build_fast_solvers.sh). Skipped when missing."""
    missing = {"fem": not _getdp_available(), "solvers": not _fast_solvers_available()}
    reasons = {"fem": "GetDP not found (scripts/get_onelab.sh)",
               "solvers": "FastHenry / FastCap not built (scripts/build_fast_solvers.sh)"}
    for item in items:
        for marker, absent in missing.items():
            if absent and marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reasons[marker]))

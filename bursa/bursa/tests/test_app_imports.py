"""Check the dashboard's imports without launching Streamlit.

This exists because of a deploy failure that cost far more time than it
should have. Streamlit Community Cloud redacts the message of any uncaught
exception:

    ImportError: This app has encountered an error. The original error
    message is redacted to prevent data leaks.

For an ImportError the message IS the diagnosis. "cannot import name
walkforward_chart from tournament.charts" and "No module named matplotlib"
are unrelated problems with unrelated fixes, and the redacted form separates
them not at all -- you are left reading a traceback frame that tells you
only which line failed, which you could already see.

So: verify at test time that every name `app.py` imports actually exists.
A mismatch then fails here, by name, in a second -- instead of on a
deployed URL with the reason removed.

Nothing here imports app.py itself. Executing it runs the whole dashboard.
The module is parsed instead, which is the point: this tests the import
CONTRACT, not the app.
"""
import ast
import importlib
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "tournament" / "app.py"


def _imports_from(path: Path) -> list[tuple[str, str]]:
    """Every (module, name) pair in a `from X import a, b` at module level."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                out.append((node.module, alias.name))
    return out


LOCAL_PREFIXES = ("core", "research", "tournament")
LOCAL_IMPORTS = [(m, n) for m, n in _imports_from(APP)
                 if m.split(".")[0] in LOCAL_PREFIXES]


def test_the_app_imports_something_worth_checking():
    """Guard the guard. If the parse silently found nothing, every test
    below would pass while checking absolutely nothing."""
    assert len(LOCAL_IMPORTS) >= 10, (
        f"only found {len(LOCAL_IMPORTS)} local imports in app.py -- the "
        f"parser is probably broken, not the app")


@pytest.mark.parametrize("module,name", LOCAL_IMPORTS,
                         ids=[f"{m}.{n}" for m, n in LOCAL_IMPORTS])
def test_every_name_the_dashboard_imports_exists(module, name):
    """The test that would have caught the deploy failure."""
    mod = importlib.import_module(module)
    assert hasattr(mod, name), (
        f"app.py does `from {module} import {name}` but {module} has no "
        f"such name. Exports: {sorted(x for x in dir(mod) if not x.startswith('_'))}")


def test_every_third_party_package_the_app_needs_is_declared():
    """A package that imports here but is missing from the deployed
    requirements fails on Cloud only -- the worst place to find out."""
    declared = set()
    for req in (APP.parent / "requirements.txt",
                APP.parent.parent / "requirements.txt"):
        if req.exists():
            for line in req.read_text().splitlines():
                line = line.split("#")[0].strip()
                if line:
                    declared.add(line.split(">")[0].split("=")[0]
                                 .split("<")[0].strip().lower())

    tree = ast.parse(APP.read_text(encoding="utf-8"))
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            used.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            used.add(node.module.split(".")[0])

    stdlib = {"sys", "pathlib", "__future__", "ast", "dataclasses", "math",
              "typing", "datetime", "json", "time", "argparse", "importlib"}
    third_party = used - stdlib - set(LOCAL_PREFIXES)
    missing = sorted(p for p in third_party if p.lower() not in declared)
    assert not missing, (
        f"app.py imports {missing} but no requirements.txt beside it or at "
        f"the repo root declares them. Declared: {sorted(declared)}")


def test_charts_exports_everything_the_app_asks_of_it():
    """Named separately because this is the module that actually broke."""
    charts = importlib.import_module("tournament.charts")
    wanted = {n for m, n in LOCAL_IMPORTS if m == "tournament.charts"}
    missing = sorted(n for n in wanted if not hasattr(charts, n))
    assert not missing, f"tournament.charts is missing {missing}"


def test_the_guarded_import_block_is_still_in_place():
    """The guard is load-bearing. Someone tidying the imports back into a
    bare block would restore the redacted, undiagnosable failure."""
    src = APP.read_text(encoding="utf-8")
    assert "except ImportError" in src, (
        "app.py's imports are no longer guarded -- a Cloud ImportError will "
        "again surface with its message redacted")
    # set_page_config must precede any other st call, including st.error in
    # the guard's failure path, or Streamlit raises a second, confusing error
    # on top of the first.
    assert src.index("set_page_config") < src.index("except ImportError")


# ------------------------------------------------------- the pick script
def test_a_ragged_cache_edge_does_not_produce_a_pick_from_five_names():
    """Caught on a real refresh, and the quiet version is the dangerous one.

    The cache is written one ticker at a time, so refreshing some names and
    not others leaves a last date holding only the tickers just fetched. Here
    that date contained nothing but illiquid delisted companies and the screen
    emptied the universe -- a loud failure. Had the threshold been slightly
    different it would instead have ranked a handful of names and returned a
    confident pick from a universe of five.
    """
    import numpy as np
    import pandas as pd

    from scripts.pick import latest_tradable_date

    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    full = pd.concat([
        pd.DataFrame({"date": dates, "ticker": f"T{i:03d}.KL",
                      "eligible": True})
        for i in range(80)], ignore_index=True)
    # Three stragglers get one extra, later bar -- the ragged edge.
    edge = dates[-1] + pd.Timedelta(days=1)
    ragged = pd.DataFrame({"date": [edge] * 3,
                           "ticker": ["T000.KL", "T001.KL", "T002.KL"],
                           "eligible": True})
    panel = pd.concat([full, ragged], ignore_index=True)

    assert panel["date"].max() == edge
    assert latest_tradable_date(panel) == dates[-1], (
        "ranked the ragged edge instead of the last real trading day")


def test_a_genuinely_complete_last_day_is_used():
    import pandas as pd

    from scripts.pick import latest_tradable_date

    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    panel = pd.concat([
        pd.DataFrame({"date": dates, "ticker": f"T{i:03d}.KL",
                      "eligible": True})
        for i in range(80)], ignore_index=True)
    assert latest_tradable_date(panel) == dates[-1]

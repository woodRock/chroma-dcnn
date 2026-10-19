"""scripts/13_build_site_data.py: the site's prose numbers come from the ledger the paper uses."""

import html
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def site():
    spec = importlib.util.spec_from_file_location(
        "site_data", ROOT / "scripts" / "13_build_site_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("RESULTS", "DATA", "OUT", "HTML", "LEDGER"):
        setattr(mod, name, ROOT / getattr(mod, name))
    return mod


def labels():
    """The fish oil labels' shape (57 Blue Cod of 103) without the private data."""
    return np.array([0] * 16 + [1] * 15 + [2] * 15 + [3] * 57)


def test_fill_html(site):
    page = (
        '<meta property="og:description" content="old">'
        '<p><data data-key="a">0.1</data> and <span data-count="1" data-count-key="b" data-dec="1" '
        'data-prefix="+">+0.0</span> and '
        '<div data-count="1" data-count-key="c" data-dec="0">0</div></p>'
    )
    nums = {"a": "0.857", "b": "15.9", "c": "9553", "pretrain_ba": "0.857", "gain_best_pp": "15.9"}
    out = site.fill_html(page, nums)
    assert '<data data-key="a">0.857</data>' in out
    assert 'data-count="15.9" data-count-key="b" data-dec="1" data-prefix="+">+15.9</span>' in out
    assert 'data-count="9553" data-count-key="c" data-dec="0">9,553</div>' in out
    assert "with 0.857 balanced accuracy, +15.9 pp over" in out
    assert site.fill_html(out, nums) == out
    with pytest.raises(KeyError, match="nope"):
        site.fill_html('<data data-key="nope">1</data>', nums)


def test_committed_page_matches_the_ledger(site):
    """docs/index.html prints what the ledger says now (rerun the script after the experiments)."""
    page = site.HTML.read_text()
    nums = site.numbers(labels())
    assert nums["n_samples"] == "103" and nums["n_bco"] == "57"
    shown = {m.group(2): html.unescape(m.group(3)) for m in site.DATA_RE.finditer(page)}
    assert shown and all(shown[k] == nums[k] for k in shown), {
        k: (shown[k], nums[k]) for k in shown if shown[k] != nums[k]
    }
    counts = {m.group(3): m.group(1) for m in site.COUNT_RE.finditer(page)}
    assert counts and all(f'data-count="{nums[k]}"' in tag for k, tag in counts.items())
    assert site.DESCRIPTION.format(**nums) in html.unescape(page)

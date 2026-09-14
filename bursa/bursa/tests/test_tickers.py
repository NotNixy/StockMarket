"""Tests for the candidate ticker list and its overrides."""
import pandas as pd
import pytest

from core.tickers import CANDIDATES, LARGE_CAP, MID_CAP, load_candidates, save_verified


def test_candidates_are_unique():
    assert len(CANDIDATES) == len(set(CANDIDATES)), "duplicate stock code"


def test_candidates_look_like_bursa_codes():
    # Bursa codes are 4 characters, digits (some start 0).
    for c in CANDIDATES:
        assert len(c) == 4 and c.isdigit(), f"{c!r} is not a 4-digit code"


def test_large_and_mid_caps_do_not_overlap():
    assert not (set(LARGE_CAP) & set(MID_CAP))


def test_load_candidates_falls_back_to_the_builtin_list(tmp_path):
    codes = load_candidates(tmp_path / "absent.csv")
    assert set(codes) == set(CANDIDATES)


def test_load_candidates_prefers_a_real_listings_file(tmp_path):
    p = tmp_path / "listings.csv"
    pd.DataFrame({"code": ["1155", "5347"], "name": ["A", "B"]}).to_csv(p, index=False)
    assert load_candidates(p) == ["1155", "5347"]


def test_listings_file_codes_are_zero_padded(tmp_path):
    # Bursa codes like 0097 lose their leading zero if read as an integer.
    p = tmp_path / "listings.csv"
    pd.DataFrame({"code": [97, 1155]}).to_csv(p, index=False)
    assert load_candidates(p) == ["0097", "1155"]


def test_listings_file_without_a_code_column_is_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"ticker": ["1155"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="code"):
        load_candidates(p)


def test_save_verified_writes_sorted_unique_codes(tmp_path):
    out = save_verified(["5347", "1155", "1155"], tmp_path / "u.csv")
    assert pd.read_csv(out, dtype=str)["code"].tolist() == ["1155", "5347"]

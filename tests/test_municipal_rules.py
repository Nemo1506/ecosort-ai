"""Tests for src/municipal_rules.py global jurisdiction standards."""
import config
from src.municipal_rules import MUNICIPAL_JURISDICTIONS, get_jurisdiction_info


def test_jurisdictions_presence():
    expected_regions = ["United States", "European Union", "United Kingdom", "Japan", "India"]
    assert len(MUNICIPAL_JURISDICTIONS) >= 5
    for r in expected_regions:
        assert any(r in k for k in MUNICIPAL_JURISDICTIONS.keys()), f"Missing region: {r}"


def test_jurisdiction_info_mapping():
    for jur_name in MUNICIPAL_JURISDICTIONS:
        for cls in config.CLASS_NAMES:
            info = get_jurisdiction_info(jur_name, cls)
            assert isinstance(info, dict)
            assert "bin_name" in info and len(info["bin_name"]) > 0
            assert "bin_color" in info and info["bin_color"].startswith("#")
            assert "rules" in info and len(info["rules"]) > 0

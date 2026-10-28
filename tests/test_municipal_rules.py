import re
import config
from src.municipal_rules import (
    MUNICIPAL_JURISDICTIONS,
    get_all_jurisdictions,
    get_jurisdiction_comparison,
    get_jurisdiction_info,
)


def test_jurisdictions_presence():
    expected_regions = ["United States", "European Union", "United Kingdom", "Japan", "India"]
    assert len(MUNICIPAL_JURISDICTIONS) >= 5
    for r in expected_regions:
        assert any(r in k for k in MUNICIPAL_JURISDICTIONS), f"Missing region: {r}"


def test_jurisdiction_info_mapping():
    hex_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
    for jur_name in MUNICIPAL_JURISDICTIONS:
        for cls in config.CLASS_NAMES:
            info = get_jurisdiction_info(jur_name, cls)
            assert isinstance(info, dict)
            assert "bin_name" in info and len(info["bin_name"]) > 0
            assert "bin_color" in info and hex_pattern.match(info["bin_color"])
            assert "rules" in info and len(info["rules"]) > 0


def test_jurisdiction_fallback_unknown():
    info = get_jurisdiction_info("NonExistentJurisdiction", "plastic")
    assert isinstance(info, dict)
    assert "bin_name" in info
    assert "bin_color" in info

    info_unknown_class = get_jurisdiction_info("🇺🇸 United States (Single-Stream Standard)", "unknown_waste_type")
    assert isinstance(info_unknown_class, dict)
    assert "bin_name" in info_unknown_class


def test_get_all_jurisdictions():
    jurs = get_all_jurisdictions()
    assert isinstance(jurs, list)
    assert len(jurs) == 5
    assert "🇺🇸 United States (Single-Stream Standard)" in jurs


def test_get_jurisdiction_comparison():
    comp = get_jurisdiction_comparison("plastic")
    assert isinstance(comp, dict)
    assert len(comp) == 5
    for jur_name, jur_info in comp.items():
        assert "bin_name" in jur_info
        assert "bin_color" in jur_info
        assert "rules" in jur_info
        assert "region_code" in jur_info


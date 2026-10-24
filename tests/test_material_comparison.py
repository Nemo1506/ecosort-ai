"""Tests for src/material_comparison.py Life-Cycle Assessment data."""
import config
from src.material_comparison import MATERIAL_LCA_DATA


def test_lca_data_coverage():
    for cls in config.CLASS_NAMES:
        assert cls in MATERIAL_LCA_DATA, f"Missing LCA data for class: {cls}"
        data = MATERIAL_LCA_DATA[cls]
        assert "display_name" in data
        assert "degradation_years" in data
        assert "virgin_energy_mj" in data and data["virgin_energy_mj"] > 0
        assert "recycled_energy_mj" in data and data["recycled_energy_mj"] >= 0
        assert "carbon_intensity_kg" in data and data["carbon_intensity_kg"] > 0
        assert "hazard_score" in data and 1 <= data["hazard_score"] <= 10


def test_energy_savings_logic():
    # Recycled energy should always be less than virgin energy
    for cls, data in MATERIAL_LCA_DATA.items():
        assert data["recycled_energy_mj"] <= data["virgin_energy_mj"], (
            f"Recycled energy ({data['recycled_energy_mj']}) should be <= virgin energy ({data['virgin_energy_mj']}) for {cls}"
        )

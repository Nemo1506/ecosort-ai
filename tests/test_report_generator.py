"""Tests for src/report_generator.py formal ESG certificate generation."""
from src.report_generator import generate_html_audit_certificate


def test_certificate_generation():
    html_out = generate_html_audit_certificate(
        material_label="battery",
        display_name="Household Battery",
        confidence=0.985,
        bin_name="Hazardous E-Waste Depot",
        jurisdiction="United States (Single-Stream)",
        steps=["Tape terminals", "Store in dry place"],
        model_name="MobileNetV2-Transfer",
    )
    assert isinstance(html_out, str)
    assert "<!DOCTYPE html>" in html_out
    assert "ECO-" in html_out  # Certificate ID prefix
    assert "Household Battery" in html_out
    assert "98.5%" in html_out
    assert "Hazardous E-Waste Depot" in html_out
    assert "Tape terminals" in html_out

"""Tests for src/assistant_kb.py knowledge base search and retrieval."""
from src.assistant_kb import ECO_ASSISTANT_FAQ, search_assistant_faq


def test_faq_structure():
    assert len(ECO_ASSISTANT_FAQ) >= 10
    for item in ECO_ASSISTANT_FAQ:
        assert "keywords" in item and len(item["keywords"]) > 0
        assert "question" in item
        assert "answer" in item
        assert "stream" in item


def test_faq_search_exact_and_fuzzy():
    # Test query for pizza box
    results = search_assistant_faq("pizza")
    assert len(results) > 0
    assert "pizza" in results[0]["question"].lower()

    # Test query for battery
    results_battery = search_assistant_faq("swollen lithium battery")
    assert len(results_battery) > 0
    assert any("batter" in k for k in results_battery[0]["keywords"])


    # Test unknown query fallback
    results_empty = search_assistant_faq("")
    assert len(results_empty) > 0  # Returns default suggestions


def test_generate_assistant_response_black_plastic():
    from src.assistant_kb import generate_assistant_response

    resp = generate_assistant_response(
        "Can I recycle a black plastic food tray marked with a recycling symbol",
        jurisdiction="United States"
    )
    assert "black plastic" in resp["headline"].lower()
    assert "DO NOT RECYCLE" in resp["verdict"].upper()
    assert "No" in resp["direct_answer"]
    assert "infrared" in resp["scientific_reason"].lower() or "nir" in resp["scientific_reason"].lower()
    assert len(resp["action_steps"]) >= 3
    assert "Landfill" in resp["target_bin"] or "Trash" in resp["target_bin"]


def test_generate_assistant_response_jurisdiction_bins():
    from src.assistant_kb import generate_assistant_response

    resp_us = generate_assistant_response("pizza box grease", jurisdiction="United States")
    resp_uk = generate_assistant_response("pizza box grease", jurisdiction="United Kingdom")
    resp_eu = generate_assistant_response("pizza box grease", jurisdiction="European Union")

    assert resp_us["target_bin"] != resp_uk["target_bin"] or "Cart" in resp_us["target_bin"] or "Bin" in resp_uk["target_bin"]
    assert "Compost" in resp_us["target_bin"] or "Caddy" in resp_uk["target_bin"] or "Biotonne" in resp_eu["target_bin"]

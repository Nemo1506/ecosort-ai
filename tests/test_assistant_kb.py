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

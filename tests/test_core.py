import pytest
from lab.core import create_panel, aggregate, validate_pmf, weighted_choice, record_csv_rows
from lab.store import connect, save_study, save_response, load_responses, load_study


def test_panel_is_deterministic():
    a = create_panel(25, 20, 55, seed=4)
    assert a == create_panel(25, 20, 55, seed=4)
    assert len({p["respondent_id"] for p in a}) == 25
    assert all(20 <= p["age"] <= 55 for p in a)


def test_invalid_panel():
    with pytest.raises(ValueError): create_panel(5, 70, 20)
    with pytest.raises(ValueError): weighted_choice(__import__("random").Random(), ["a", "b"], [0, 0])


def test_aggregate_and_export():
    rows = [{"concept_id": "a", "respondent_id": "P0001", "persona": {"region": "West"}, "response": "yes", "status": "scored", "pmf": [0, 0, 0, 0, 1]},
            {"concept_id": "b", "respondent_id": "P0001", "persona": {"region": "West"}, "response": "no", "status": "scored", "pmf": [1, 0, 0, 0, 0]}]
    assert [x["concept_id"] for x in aggregate(rows)] == ["a", "b"]
    assert aggregate(rows, {"region": "Midwest"}) == []
    assert record_csv_rows(rows)[0]["rating_5_probability"] == 1
    with pytest.raises(ValueError): validate_pmf([0, 0, 0, 0, 2])


def test_persistence(tmp_path):
    c = connect(tmp_path / "example.sqlite")
    study = {"name": "Sample", "concepts": []}
    study_id = save_study(c, study)
    assert load_study(c, study_id)["name"] == "Sample"
    save_response(c, study_id, {"concept_id": "a", "respondent_id": "P0001", "response": "maybe", "pmf": None})
    assert load_responses(c, study_id)[0]["response"] == "maybe"
    c.close()

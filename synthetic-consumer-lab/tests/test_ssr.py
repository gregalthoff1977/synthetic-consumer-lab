import pytest
pytest.importorskip("polars")
from lab.ssr import ANCHORS, references


def test_reference_set_shape():
    assert len(ANCHORS) == 6
    assert all(len(s) == 5 for s in ANCHORS)
    refs = references()
    assert refs.height == 30
    assert refs.filter(refs["id"] == "anchor_1")["int_response"].to_list() == [1, 2, 3, 4, 5]

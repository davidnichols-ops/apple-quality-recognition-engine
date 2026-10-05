from scripts.distill_labels import discover_sequences


def test_calyx_capture_uses_declared_capture_id(tmp_path) -> None:
    image = tmp_path / (
        "raw_20261005_120000_000000_batch-a-stem-00000_unknown_calyx_view_0.jpg"
    )
    image.touch()

    sequences = discover_sequences(tmp_path)

    assert sequences == {"batch-a-stem-00000": [image]}

import pytest

import camera_utils


def test_configured_index_comes_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("ARDUCAM_CAMERA_INDEX", "2")
    assert camera_utils.configured_arducam_index() == 2


def test_invalid_configured_index_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("ARDUCAM_CAMERA_INDEX", "camera-two")
    with pytest.raises(ValueError, match="integer"):
        camera_utils.configured_arducam_index()


def test_capture_can_fail_closed_when_arducam_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(camera_utils, "arducam_connected", lambda: False)
    with pytest.raises(RuntimeError, match="Arducam not detected"):
        camera_utils.detect_arducam_index(allow_builtin_fallback=False)


def test_connected_arducam_uses_configured_index(monkeypatch) -> None:
    monkeypatch.setattr(camera_utils, "arducam_connected", lambda: True)
    monkeypatch.setenv("ARDUCAM_CAMERA_INDEX", "1")
    assert camera_utils.detect_arducam_index(allow_builtin_fallback=False) == 1

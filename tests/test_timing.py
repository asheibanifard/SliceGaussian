import pytest

from slicegaussian.timing import first_crossing


def test_first_observed_crossing_survives_later_quality_drop():
    rows = [{"wall_seconds": t, "psnr": p, "iteration": i}
            for i, (t, p) in enumerate([(5., 20.), (10., 32.), (20., 25.), (30., 35.)])]
    result = first_crossing(rows, 30.)
    assert result["seconds"] == 10. and result["previous_observation_seconds"] == 5.
    assert result["iteration"] == 1
    assert first_crossing(rows, 36.)["status"] == "not_reached"
    assert first_crossing(rows, 36.)["seconds"] is None


def test_clock_must_not_go_backward_or_be_nonfinite():
    for times in [(2., 1.), (1., float("nan")), (1., float("inf"))]:
        rows = [{"wall_seconds": t, "psnr": 20., "iteration": i} for i, t in enumerate(times)]
        with pytest.raises(ValueError):
            first_crossing(rows, 10.)

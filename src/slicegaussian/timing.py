"""First observed quality crossings on measured wall-clock timelines; never extrapolate."""
import math


def first_crossing(rows, threshold):
    """Return the first observed crossing or an explicit unreached record.

    Input must be chronological. 'previous_observation_seconds' describes sampling
    resolution, not a guarantee about unobserved earlier quality excursions.
    """
    previous = None
    found = None
    last_time = -math.inf
    for row in rows:
        seconds, psnr = float(row["wall_seconds"]), float(row["psnr"])
        if not math.isfinite(seconds) or seconds < 0 or seconds < last_time or not math.isfinite(psnr):
            raise ValueError("wall timeline must be finite, nonnegative and chronological")
        if found is None and psnr >= threshold:
            found = {"status": "reached", "seconds": seconds, "iteration": int(row["iteration"]),
                     "observed_psnr": psnr, "previous_observation_seconds": previous}
        previous, last_time = seconds, seconds
    return found or {"status": "not_reached", "seconds": None, "iteration": None,
                     "observed_psnr": None, "previous_observation_seconds": previous}

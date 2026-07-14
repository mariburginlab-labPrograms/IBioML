"""Unit tests for the out-of-corridor trial filter in preprocess_data.

The filter removes whole trials whose peak position leaves the physical
corridor by a large margin — a session-start / segmentation artifact seen in
the DG-naive recordings (first trial ramps to ~26500 deg while a valid trial
never exceeds ~1200 deg). These tests are data-free: they build synthetic
recordings so they run anywhere without the lab's .mat files.
"""

import numpy as np
from ibioml.preprocess_data import get_idx_by_out_of_corridor_trials


def _make_recording(trial_peaks, trial_len=100):
    """Build (pos_binned (T,1), trialFinalBin) from per-trial peak positions.

    Each trial's position ramps linearly from 0 to its peak. trialFinalBin is
    the 0-indexed last bin of each trial (the convention used inside
    preprocess_data after load_data subtracts 1 from the MATLAB indices).
    """
    pos, tfb, cursor = [], [], 0
    for pk in trial_peaks:
        pos.append(np.linspace(0, pk, trial_len))
        cursor += trial_len
        tfb.append(cursor - 1)
    return np.concatenate(pos)[:, None], np.array(tfb, dtype=float)


def test_flags_only_the_out_of_corridor_trial():
    # three normal trials (peak 500 deg) with one pathological (3000 deg) in the middle
    pos, tfb = _make_recording([500, 3000, 500, 500])
    idx = get_idx_by_out_of_corridor_trials(pos, tfb, thPosition=1500)
    assert idx.tolist() == list(range(100, 200))


def test_pathological_first_trial():
    # real-world DG-naive case: the FIRST trial is the broken one
    pos, tfb = _make_recording([26508, 500, 500])
    idx = get_idx_by_out_of_corridor_trials(pos, tfb, thPosition=1500)
    assert idx.tolist() == list(range(0, 100))


def test_healthy_dataset_untouched():
    # every trial bounded within corridor + drift -> nothing removed
    pos, tfb = _make_recording([500, 600, 550, 700])
    idx = get_idx_by_out_of_corridor_trials(pos, tfb, thPosition=1500)
    assert idx.size == 0


def test_returns_int_array():
    pos, tfb = _make_recording([3000, 500])
    idx = get_idx_by_out_of_corridor_trials(pos, tfb, thPosition=1500)
    assert idx.dtype == np.dtype(int)


def test_nan_bins_do_not_mask_a_high_peak():
    # a trial with real high positions AND some NaN bins must still be flagged
    # (np.max would propagate NaN and hide it; the filter must use nanmax)
    pos, tfb = _make_recording([3000, 500])
    pos[10:20, 0] = np.nan
    idx = get_idx_by_out_of_corridor_trials(pos, tfb, thPosition=1500)
    assert idx.tolist() == list(range(0, 100))

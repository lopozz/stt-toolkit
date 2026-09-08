#!/usr/bin/env python
"""
SpecAugment (Park et al., 2019 - https://arxiv.org/pdf/1904.08779).

Three policies from the paper, all optional/independently toggleable:
  1. Time warping - warps the time axis via a smooth deformation anchored at
     one random point. The paper's own ablations found this contributes the
     least of the three (LB vs LD policies differ mainly in masking, not
     warping) and it's the most expensive to implement correctly (the paper
     uses a full 2D sparse image warp). This implementation uses a cheaper
     per-frequency-row 1D piecewise-linear time warp instead of true 2D image
     warping - a reasonable approximation, not a faithful reproduction.
  2. Frequency masking - zero out `num_freq_masks` random contiguous bands of
     mel channels, each up to `freq_mask_param` channels wide.
  3. Time masking - zero out `num_time_masks` random contiguous spans of time
     frames, each up to `time_mask_param` frames wide (capped further by
     `time_mask_ratio` * n_frames, the paper's "p", so a single mask can't
     wipe out a disproportionate fraction of a short/already-short-relative-
     to-padding utterance).

Defaults below roughly match the paper's "LD" (LibriSpeech Double) policy:
W=80, F=27, mF=2, T=100, p=1.0, mT=2.

Training uses coordinated augmentation via spec_augment_pair; evaluation stays clean.
"""

from __future__ import annotations

import numpy as np


# Table 1 masking policies and variants with time warping disabled.
# Warped presets use W=80, but our warp is the 1D approximation described above.
# Pass valid_frames and an optional demo seed separately at the call site.
SPEC_AUGMENT_PRESETS = {
    "lb_no_warp": {
        "freq_mask_param": 27,
        "num_freq_masks": 1,
        "time_mask_param": 100,
        "num_time_masks": 1,
        "time_mask_ratio": 1.0,
        "time_warp_param": 0,
        "mask_value": "mean",
    },
    "ld_no_warp": {
        "freq_mask_param": 27,
        "num_freq_masks": 2,
        "time_mask_param": 100,
        "num_time_masks": 2,
        "time_mask_ratio": 1.0,
        "time_warp_param": 0,
        "mask_value": "mean",
    },
}
SPEC_AUGMENT_PRESETS["lb"] = {
    **SPEC_AUGMENT_PRESETS["lb_no_warp"],
    "time_warp_param": 80,
}
SPEC_AUGMENT_PRESETS["ld"] = {
    **SPEC_AUGMENT_PRESETS["ld_no_warp"],
    "time_warp_param": 80,
}


def _freq_mask(
    spec: np.ndarray,
    freq_mask_param: int,
    num_masks: int,
    mask_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_mels = spec.shape[0]
    for _ in range(num_masks):
        f = int(rng.integers(0, freq_mask_param + 1))
        if f == 0 or f >= n_mels:
            continue
        f0 = int(rng.integers(0, n_mels - f + 1))
        spec[f0 : f0 + f, :] = mask_value
    return spec


def _time_mask(
    spec: np.ndarray,
    time_mask_param: int,
    num_masks: int,
    time_mask_ratio: float,
    mask_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_frames = spec.shape[1]
    max_t = min(time_mask_param, int(n_frames * time_mask_ratio))
    for _ in range(num_masks):
        t = int(rng.integers(0, max_t + 1))
        if t == 0 or t >= n_frames:
            continue
        t0 = int(rng.integers(0, n_frames - t + 1))
        spec[:, t0 : t0 + t] = mask_value
    return spec


def _time_warp(
    spec: np.ndarray, time_warp_param: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Cheap approximation of the paper's time warping: pick one anchor frame in
    the interior of the spectrogram and displace it left/right by up to
    `time_warp_param` frames, then re-sample each frequency row along the
    time axis with a piecewise-linear map through that one displaced point.
    Not the paper's true 2D sparse image warp, but captures the same idea
    (locally stretch/compress time) far more cheaply.
    """
    n_mels, n_frames = spec.shape
    w = time_warp_param
    if w <= 0 or n_frames <= 2 * w + 1:
        return spec

    center = int(rng.integers(w, n_frames - w))
    displacement = int(rng.integers(-w, w + 1))
    if displacement == 0:
        return spec
    warped_center = int(np.clip(center + displacement, 1, n_frames - 2))

    # Piecewise-linear map: original frame indices [0, center, n_frames-1]
    # land at [0, warped_center, n_frames-1]; every other frame is resampled
    # by linearly interpolating along that map.
    src_anchors = np.array([0, center, n_frames - 1], dtype=np.float64)
    dst_anchors = np.array([0, warped_center, n_frames - 1], dtype=np.float64)
    dst_positions = np.arange(n_frames, dtype=np.float64)
    # For each output position, find the corresponding source position.
    src_positions = np.interp(dst_positions, dst_anchors, src_anchors)

    warped = np.empty_like(spec)
    frame_indices = np.arange(n_frames, dtype=np.float64)
    for mel_idx in range(n_mels):
        warped[mel_idx] = np.interp(src_positions, frame_indices, spec[mel_idx])
    return warped


def spec_augment(
    mel_spectrogram: np.ndarray,
    freq_mask_param: int = 27,
    num_freq_masks: int = 2,
    time_mask_param: int = 100,
    num_time_masks: int = 2,
    time_mask_ratio: float = 1.0,
    time_warp_param: int = 0,
    mask_value: float | str = "mean",
    seed: int | None = None,
    valid_frames: int | None = None,
) -> np.ndarray:
    """
    Apply SpecAugment to one example's log-mel spectrogram.

    Args:
        mel_spectrogram: shape (n_mels, n_frames), e.g. `input_features[0]` /
            `inputs.input_features` from WhisperFeatureExtractor.
        freq_mask_param: max width (mel channels) of each frequency mask.
        num_freq_masks: how many frequency masks to apply.
        time_mask_param: max width (frames) of each time mask.
        num_time_masks: how many time masks to apply.
        time_mask_ratio: caps each time mask to at most this fraction of
            valid_frames (or the full frame count when valid_frames is omitted).
        time_warp_param: max frame displacement for time warping; 0 disables
            it entirely (recommended default - see the module docstring on
            why this policy contributes least and costs most).
        mask_value: value written into masked regions. "mean" (default) uses
            this spectrogram's own mean, matching the paper's recommendation
            (masking to a fixed 0 can look identical to real silence/padding
            elsewhere in the spectrogram, which literal silence is not the
            same signal as "this information was randomly withheld").
        seed: optional seed for reproducibility; omit for real training runs
            (each call should get fresh randomness), set for tests/demos.
        valid_frames: number of leading frames containing the unpadded audio.
            All operations and the mean fill value use only this region;
            trailing padding is preserved. Zero returns an unchanged copy.
            Omit to augment the full spectrogram as before.

    Returns:
        A new array (input is not modified in place) of the same shape.
    """
    rng = np.random.default_rng(seed)
    spec = np.array(mel_spectrogram, dtype=np.float32, copy=True)
    if spec.ndim != 2:
        raise ValueError("mel_spectrogram must have shape (n_mels, n_frames)")
    if valid_frames is None:
        valid_frames = spec.shape[1]
    if (
        not isinstance(valid_frames, (int, np.integer))
        or isinstance(valid_frames, (bool, np.bool_))
        or not 0 <= valid_frames <= spec.shape[1]
    ):
        raise ValueError(
            "valid_frames must be an integer between 0 and the spectrogram frame count"
        )
    if valid_frames == 0:
        return spec
    active_spec = spec[:, :valid_frames]

    if mask_value == "mean":
        fill_value = float(active_spec.mean())
    else:
        fill_value = float(mask_value)

    if time_warp_param > 0:
        active_spec = _time_warp(active_spec, time_warp_param, rng)

    active_spec = _freq_mask(
        active_spec, freq_mask_param, num_freq_masks, fill_value, rng
    )
    active_spec = _time_mask(
        active_spec, time_mask_param, num_time_masks, time_mask_ratio, fill_value, rng
    )
    spec[:, :valid_frames] = active_spec

    return spec


if __name__ == "__main__":
    # Minimal self-test/demo: fake a Whisper-shaped log-mel spectrogram
    # (80 mel bins x 3000 frames) and confirm augmentation runs and actually
    # changes something, without needing a real audio pipeline.
    rng = np.random.default_rng(0)
    fake_spec = rng.normal(loc=-1.0, scale=2.0, size=(80, 3000)).astype(np.float32)

    augmented = spec_augment(fake_spec, time_warp_param=80, seed=0)

    n_changed = np.sum(~np.isclose(fake_spec, augmented))
    print(f"shape: {augmented.shape}, dtype: {augmented.dtype}")
    print(
        f"changed {n_changed} / {fake_spec.size} values ({100 * n_changed / fake_spec.size:.1f}%)"
    )
    assert augmented.shape == fake_spec.shape
    assert n_changed > 0, "expected spec_augment to change something"
    print("OK")


def spec_augment_pair(student, teacher, *, valid_frames, seed, **policy):
    """Share time masks and proportionally map frequency masks across mel grids."""
    outputs = [np.array(x, dtype=np.float32, copy=True) for x in (student, teacher)]
    if any(x.ndim != 2 or not 0 <= valid_frames <= x.shape[1] for x in outputs):
        raise ValueError("Invalid spectrogram shape or valid_frames")
    if valid_frames == 0:
        return tuple(outputs)
    rng = np.random.default_rng(seed)
    active = [x[:, :valid_frames] for x in outputs]
    value = policy.get("mask_value", "mean")
    fills = [float(x.mean()) if value == "mean" else float(value) for x in active]
    warp = policy.get("time_warp_param", 0)
    if warp:
        warp_seed = int(rng.integers(0, 2**32))
        active = [_time_warp(x, warp, np.random.default_rng(warp_seed)) for x in active]
    n_mels = student.shape[0]
    for _ in range(policy.get("num_freq_masks", 2)):
        width = int(rng.integers(0, policy.get("freq_mask_param", 27) + 1))
        if width == 0 or width >= n_mels:
            continue
        start = int(rng.integers(0, n_mels - width + 1))
        for x, fill in zip(active, fills):
            # Approximate correspondence: the two mel filter banks differ.
            lo, hi = np.rint(
                np.array([start, start + width]) * x.shape[0] / n_mels
            ).astype(int)
            x[lo:hi, :] = fill
    max_width = min(
        policy.get("time_mask_param", 100),
        int(valid_frames * policy.get("time_mask_ratio", 1.0)),
    )
    for _ in range(policy.get("num_time_masks", 2)):
        width = int(rng.integers(0, max_width + 1))
        if width == 0 or width >= valid_frames:
            continue
        start = int(rng.integers(0, valid_frames - width + 1))
        for x, fill in zip(active, fills):
            x[:, start : start + width] = fill
    for output, x in zip(outputs, active):
        output[:, :valid_frames] = x
    return tuple(outputs)

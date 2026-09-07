#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrated Lab colour features for small blocks on a white background."""

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


CALIBRATION_COLORS = ("red", "yellow", "green", "blue")
MODEL_VERSION = 1

# Keep much of a small segmentation mask, then remove achromatic background
# pixels.  The connected-component step prevents isolated coloured noise from
# influencing the median.
CORE_FRACTION = 0.60
MIN_CORE_PIXELS = 20
MIN_VALID_PIXELS = 12
MIN_CHROMA = 12.0
MIN_LIGHTNESS = 35.0
BLACK_LIGHTNESS = 58.0
WHITE_LIGHTNESS = 178.0
ACHROMATIC_CHROMA = 13.0
MIN_CLASS_MARGIN = 6.0
MIN_CENTER_SEPARATION = 18.0
MIN_CLASS_CHROMA_RATIO = 0.65


def adaptive_mask_core(mask):
    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    area = int(np.count_nonzero(mask_u8))
    if area == 0:
        return mask_u8
    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    keep = min(area, max(MIN_CORE_PIXELS, int(np.ceil(area * CORE_FRACTION))))
    if keep >= area:
        return mask_u8
    inside = distance[mask_u8.astype(bool)]
    threshold = float(np.partition(inside, inside.size - keep)[inside.size - keep])
    core = ((distance >= threshold) & (mask_u8 > 0)).astype(np.uint8)
    return core if np.any(core) else mask_u8


def _largest_component(mask):
    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    if count <= 1:
        return mask_u8
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8)


def estimate_white_balance_gains(bgr_image, object_mask):
    """Estimate conservative per-frame BGR gains from the surrounding white table."""
    image = np.asarray(bgr_image, dtype=np.uint8)
    mask = (np.asarray(object_mask, dtype=np.uint8) > 0).astype(np.uint8)
    dilated = cv2.dilate(mask, np.ones((21, 21), np.uint8), iterations=1)
    ring = (dilated > 0) & (mask == 0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_like = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 120)
    candidates = ring & white_like
    if np.count_nonzero(candidates) < 100:
        candidates = (mask == 0) & white_like
    if np.count_nonzero(candidates) < 100:
        return np.ones(3, dtype=np.float32), 0

    white_bgr = np.median(image[candidates].astype(np.float32), axis=0)
    neutral = float(np.mean(white_bgr))
    gains = neutral / np.maximum(white_bgr, 1.0)
    gains = np.clip(gains, 0.75, 1.33).astype(np.float32)
    return gains, int(np.count_nonzero(candidates))


def extract_lab_feature(bgr_image, object_mask):
    """Return a robust per-object Lab feature and diagnostic information."""
    image = np.asarray(bgr_image, dtype=np.uint8)
    core = adaptive_mask_core(object_mask)
    core_count = int(np.count_nonzero(core))
    if core_count == 0:
        return None

    gains, white_pixels = estimate_white_balance_gains(image, object_mask)
    corrected = np.clip(image.astype(np.float32) * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_centered = lab[:, :, 1] - 128.0
    b_centered = lab[:, :, 2] - 128.0
    chroma = np.hypot(a_centered, b_centered)

    core_bool = core.astype(bool)
    central_lab = lab[core_bool]
    central_l = float(np.median(central_lab[:, 0]))
    central_chroma = float(np.median(chroma[core_bool]))

    valid = core_bool & (lab[:, :, 0] >= MIN_LIGHTNESS) & (chroma >= MIN_CHROMA)
    valid = _largest_component(valid)
    pixels = lab[valid.astype(bool)]
    valid_count = int(len(pixels))

    result = {
        "core_pixels": core_count,
        "valid_pixels": valid_count,
        "white_reference_pixels": white_pixels,
        "white_balance_gains": gains.astype(float).tolist(),
        "central_lightness": central_l,
        "central_chroma": central_chroma,
        "valid_mask": valid,
    }
    if valid_count < MIN_VALID_PIXELS:
        result["lab"] = None
        return result

    feature = np.median(pixels, axis=0)
    deviation = np.median(np.abs(pixels - feature), axis=0)
    result["lab"] = feature.astype(float).tolist()
    result["lab_mad"] = deviation.astype(float).tolist()
    return result


def _colour_distance(first, second):
    """Lighting-tolerant Lab distance: chroma dominates, L has a small weight."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    delta = first - second
    return float(np.sqrt(0.15 * delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2))


def build_calibration(samples_by_color):
    classes = {}
    sample_arrays = {}
    for color in CALIBRATION_COLORS:
        values = np.asarray(samples_by_color.get(color, []), dtype=float)
        if values.ndim != 2 or values.shape[0] < 10 or values.shape[1] != 3:
            raise ValueError(f"{color} requires at least 10 valid Lab samples")
        center = np.median(values, axis=0)
        distances = np.asarray([_colour_distance(value, center) for value in values])
        raw_limit = max(10.0, float(np.percentile(distances, 95)) * 1.8 + 2.0)
        classes[color] = {
            "center_lab": np.round(center, 4).tolist(),
            "sample_count": int(len(values)),
            "sample_distance_median": round(float(np.median(distances)), 4),
            "sample_distance_p95": round(float(np.percentile(distances, 95)), 4),
            "raw_max_distance": raw_limit,
        }
        sample_arrays[color] = values

    centers = {name: item["center_lab"] for name, item in classes.items()}
    for color, item in classes.items():
        separations = [
            _colour_distance(centers[color], centers[other])
            for other in CALIBRATION_COLORS
            if other != color
        ]
        nearest = min(separations)
        if nearest < MIN_CENTER_SEPARATION:
            raise ValueError(
                f"{color} is only {nearest:.1f} Lab units from another colour; "
                "repeat calibration with cleaner masks and stable lighting"
            )
        item["nearest_center_distance"] = round(nearest, 4)
        item["max_distance"] = round(max(8.0, min(item.pop("raw_max_distance"), nearest * 0.45)), 4)

    return {
        "version": MODEL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "colors": list(CALIBRATION_COLORS),
        "feature": "OpenCV Lab median; white-background balanced; weighted Lab distance",
        "classes": classes,
    }


def load_color_calibration(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("version") != MODEL_VERSION:
        raise ValueError(f"unsupported colour calibration version: {model.get('version')}")
    classes = model.get("classes", {})
    for color in CALIBRATION_COLORS:
        item = classes.get(color)
        if item is None or len(item.get("center_lab", [])) != 3 or float(item.get("max_distance", 0)) <= 0:
            raise ValueError(f"invalid or missing calibration for {color}")
    return model


def _classify_extracted_feature(feature, model, distance_scale=1.0):
    """Classify one already-extracted feature without re-reading an image."""
    if feature is None:
        return "unknown", 0.0, 0, 0.0, {"reason": "empty mask"}

    core_count = int(feature["core_pixels"])
    central_l = float(feature["central_lightness"])
    central_chroma = float(feature["central_chroma"])
    if central_l <= BLACK_LIGHTNESS and central_chroma <= ACHROMATIC_CHROMA + 5.0:
        confidence = float(np.clip((BLACK_LIGHTNESS - central_l + 12.0) / 30.0, 0.0, 1.0))
        return "black", confidence, core_count, confidence, feature
    if central_l >= WHITE_LIGHTNESS and central_chroma <= ACHROMATIC_CHROMA:
        confidence = float(
            np.clip(
                0.5 * (central_l - WHITE_LIGHTNESS + 20.0) / 45.0
                + 0.5 * (ACHROMATIC_CHROMA - central_chroma + 4.0) / 17.0,
                0.0,
                1.0,
            )
        )
        return "white", confidence, core_count, confidence, feature

    if feature.get("lab") is None:
        feature["reason"] = "too few chromatic pixels"
        return "unknown", 0.0, int(feature["valid_pixels"]), 0.0, feature

    ranked = sorted(
        (
            _colour_distance(feature["lab"], item["center_lab"]),
            color,
            float(item["max_distance"]),
        )
        for color, item in model["classes"].items()
    )
    best_distance, color, calibrated_max_distance = ranked[0]
    max_distance = float(calibrated_max_distance) * float(distance_scale)
    second_distance = ranked[1][0]
    distance_gap = second_distance - best_distance
    margin = float(np.clip(distance_gap / max(max_distance, 1e-6), 0.0, 1.0))
    center_lab = np.asarray(model["classes"][color]["center_lab"], dtype=float)
    center_chroma = float(np.hypot(center_lab[1] - 128.0, center_lab[2] - 128.0))
    minimum_class_chroma = max(MIN_CHROMA, center_chroma * MIN_CLASS_CHROMA_RATIO)
    feature["best_distance"] = best_distance
    feature["second_distance"] = second_distance
    feature["calibrated_max_distance"] = calibrated_max_distance
    feature["max_distance"] = max_distance
    feature["minimum_class_chroma"] = minimum_class_chroma

    if central_chroma < minimum_class_chroma:
        feature["reason"] = "insufficient chroma for nearest calibrated colour"
        return "unknown", 0.0, int(feature["valid_pixels"]), margin, feature
    if best_distance > max_distance:
        feature["reason"] = "outside calibrated colour range"
        return "unknown", 0.0, int(feature["valid_pixels"]), margin, feature
    if distance_gap < MIN_CLASS_MARGIN:
        feature["reason"] = "ambiguous nearest colours"
        return "unknown", 0.0, int(feature["valid_pixels"]), margin, feature

    proximity = float(np.clip(1.0 - best_distance / max_distance, 0.0, 1.0))
    confidence = float(np.clip(0.65 * proximity + 0.35 * margin, 0.0, 1.0))
    return color, confidence, int(feature["valid_pixels"]), margin, feature


def classify_lab_features(features, model, distance_scale=1.0):
    """Classify the median Lab feature from several matched RGB-D frames.

    Only compact diagnostics are retained by the grasp pipeline, so this
    function deliberately does not require the original images or masks.
    Aggregating in calibrated Lab space is more stable than mixing per-frame
    class labels, especially around the yellow/green boundary.
    """
    usable = [dict(feature) for feature in features if feature]
    if not usable:
        return "unknown", 0.0, 0, 0.0, {"reason": "no Lab observations"}

    lab_values = [feature["lab"] for feature in usable if feature.get("lab") is not None]
    aggregate = {
        "core_pixels": int(sum(int(item.get("core_pixels", 0)) for item in usable)),
        "valid_pixels": int(sum(int(item.get("valid_pixels", 0)) for item in usable)),
        "white_reference_pixels": int(
            sum(int(item.get("white_reference_pixels", 0)) for item in usable)
        ),
        "central_lightness": float(
            np.median([float(item["central_lightness"]) for item in usable])
        ),
        "central_chroma": float(
            np.median([float(item["central_chroma"]) for item in usable])
        ),
        "lab": (
            np.median(np.asarray(lab_values, dtype=float), axis=0).tolist()
            if lab_values
            else None
        ),
        "frames": len(usable),
    }
    gains = [item.get("white_balance_gains") for item in usable]
    gains = [item for item in gains if item is not None]
    if gains:
        aggregate["white_balance_gains"] = np.median(
            np.asarray(gains, dtype=float), axis=0
        ).tolist()
    deviations = [item.get("lab_mad") for item in usable]
    deviations = [item for item in deviations if item is not None]
    if deviations:
        aggregate["lab_mad"] = np.median(
            np.asarray(deviations, dtype=float), axis=0
        ).tolist()
    return _classify_extracted_feature(aggregate, model, distance_scale)


def classify_lab_color(bgr_image, object_mask, model, distance_scale=1.0):
    feature = extract_lab_feature(bgr_image, object_mask)
    return _classify_extracted_feature(feature, model, distance_scale)

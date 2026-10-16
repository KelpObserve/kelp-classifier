"""Spatial geometry of the Sentinel-2 input passed to frozen Clay."""
import math

NATIVE_SIZE = 128
NATIVE_GSD_M = 10.0
ENCODER_SIZE = 224


def resized_gsd(native_gsd_m, native_size, resized_size):
    """Preserve the ground footprint when resampling a square image."""
    if any(not math.isfinite(v) or v <= 0 for v in (native_gsd_m, native_size, resized_size)):
        raise ValueError("Grid spacing and image sizes must be finite and positive")
    return native_gsd_m * native_size / resized_size


ENCODER_GSD_M = resized_gsd(NATIVE_GSD_M, NATIVE_SIZE, ENCODER_SIZE)


def validate_geometry(metadata, *, checkpoint=False):
    """Reject caches or heads whose spatial metadata cannot match this encoder."""
    key = "gsd" if checkpoint else "gsd_m"
    value = metadata.get(key)
    if not isinstance(value, (int, float)) or not math.isclose(value, ENCODER_GSD_M, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Expected encoder {key}={ENCODER_GSD_M}; got {value!r}")
    if checkpoint:
        if metadata.get("resize") != [ENCODER_SIZE, ENCODER_SIZE]:
            raise ValueError("Checkpoint resize does not match the encoder grid")
    else:
        for key, expected in (("native_gsd_m", NATIVE_GSD_M), ("native_size", NATIVE_SIZE), ("resized_size", ENCODER_SIZE)):
            if metadata.get(key) != expected:
                raise ValueError(f"Encoder protocol {key} must be {expected}")
    return value

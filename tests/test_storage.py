import numpy as np
import pytest

from slicegaussian.storage import encode_arrays, decode_arrays, PRECISIONS


def test_fp32_is_exact_and_decoder_metadata_roundtrips():
    rng = np.random.default_rng(0)
    source = {"centres": rng.normal(size=(25, 3)).astype(np.float32),
              "buffer": np.arange(9, dtype=np.int64), "scalar": np.array(.5, np.float32)}
    metadata = {"shape": [125, 256, 256], "architecture": "test"}
    decoded, got = decode_arrays(encode_arrays(source, metadata))
    assert got == metadata
    for key in source:
        assert np.array_equal(source[key], decoded[key])
        assert source[key].dtype == decoded[key].dtype


def test_all_precisions_reconstruct_shapes_constants_and_bounded_errors():
    rng = np.random.default_rng(1)
    source = {"matrix": rng.normal(size=(101, 3)).astype(np.float32),
              "bias": rng.normal(size=17).astype(np.float32), "constant": np.full((4, 2), 7., np.float32)}
    for precision in PRECISIONS:
        decoded, _ = decode_arrays(encode_arrays(source, {}, precision))
        for key, value in source.items():
            assert decoded[key].shape == value.shape and np.isfinite(decoded[key]).all()
            if precision.startswith("u"):
                axis = 0 if value.ndim > 1 else None
                step = (value.max(axis) - value.min(axis)) / (2 ** int(precision[1:]) - 1)
                assert (np.abs(decoded[key] - value) <= step * .501 + 1e-6).all()
        assert np.array_equal(decoded["constant"], source["constant"])


def test_invalid_archive_and_nonfinite_weights_are_rejected():
    with pytest.raises(ValueError):
        decode_arrays(b"not an archive")
    with pytest.raises(ValueError):
        encode_arrays({"bad": np.array([np.nan], np.float32)}, {})

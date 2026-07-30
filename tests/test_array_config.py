"""Tests for rfi_simulator.array_config.ArrayConfig."""

from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from rfi_simulator import ArrayConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARRAY_YAML = REPO_ROOT / "configs" / "array_default.yaml"


def test_from_yaml_round_trip():
    """Loading the default 10-antenna array YAML gives sane geometry."""
    config = ArrayConfig.from_yaml(DEFAULT_ARRAY_YAML)

    assert config.n_antennas == 10
    assert config.antenna_positions_enu_m.shape == (10, 3)
    assert config.antenna_positions_enu_m.dtype == np.float64
    assert config.latitude_deg == pytest.approx(37.234)
    assert config.longitude_deg == pytest.approx(-118.282)
    assert config.height_m == pytest.approx(1222.0)
    assert config.name == "array_default"

    # All baselines within ~100 m as documented in the YAML.
    baselines, _ = config.baselines_enu_m()
    baseline_lengths = np.linalg.norm(baselines, axis=1)
    assert np.all(baseline_lengths < 150.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"antenna_positions_enu_m": np.zeros((1, 3))},
            "at least 2 antennas",
        ),
        (
            {"antenna_positions_enu_m": np.zeros((5, 2))},
            "shape",
        ),
        (
            {"antenna_positions_enu_m": np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 0.0]])},
            "non-finite",
        ),
    ],
)
def test_validation_errors_raise_value_error(kwargs, match):
    base_kwargs = dict(latitude_deg=37.234, longitude_deg=-118.282, height_m=1222.0)
    base_kwargs.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ArrayConfig(**base_kwargs)


def test_duplicate_positions_warn_but_are_allowed():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
        ]
    )
    with pytest.warns(UserWarning, match="duplicate"):
        config = ArrayConfig(
            antenna_positions_enu_m=positions,
            latitude_deg=37.234,
            longitude_deg=-118.282,
            height_m=1222.0,
        )
    assert config.n_antennas == 3


def test_baseline_count_matches_n_choose_2():
    n = 10
    rng = np.random.default_rng(42)
    positions = rng.uniform(-50, 50, size=(n, 2))
    positions = np.hstack([positions, np.zeros((n, 1))])

    config = ArrayConfig(
        antenna_positions_enu_m=positions,
        latitude_deg=37.234,
        longitude_deg=-118.282,
        height_m=1222.0,
    )

    vectors, index_pairs = config.baselines_enu_m()
    expected_n_baselines = n * (n - 1) // 2

    assert vectors.shape == (expected_n_baselines, 3)
    assert index_pairs.shape == (expected_n_baselines, 2)
    assert np.all(index_pairs[:, 0] < index_pairs[:, 1])


def test_quantity_and_float_inputs_are_equivalent():
    positions_m = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, -5.0, 0.0],
            [-20.0, 15.0, 0.0],
        ]
    )

    config_float = ArrayConfig(
        antenna_positions_enu_m=positions_m,
        latitude_deg=37.234,
        longitude_deg=-118.282,
        height_m=1222.0,
    )
    config_quantity = ArrayConfig(
        antenna_positions_enu_m=positions_m * u.m,
        latitude_deg=37.234 * u.deg,
        longitude_deg=-118.282 * u.deg,
        height_m=1222.0 * u.m,
    )

    np.testing.assert_allclose(
        config_float.antenna_positions_enu_m, config_quantity.antenna_positions_enu_m
    )
    assert config_float.latitude_deg == pytest.approx(config_quantity.latitude_deg)
    assert config_float.longitude_deg == pytest.approx(config_quantity.longitude_deg)
    assert config_float.height_m == pytest.approx(config_quantity.height_m)

    # Quantities in non-default but compatible units should also convert correctly.
    config_quantity_km = ArrayConfig(
        antenna_positions_enu_m=(positions_m * u.m).to(u.km),
        latitude_deg=37.234,
        longitude_deg=-118.282,
        height_m=1.222 * u.km,
    )
    np.testing.assert_allclose(config_quantity_km.antenna_positions_enu_m, positions_m, atol=1e-9)
    assert config_quantity_km.height_m == pytest.approx(1222.0)

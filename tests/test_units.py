"""Acceptance criterion 6: unit hygiene.

Passing astropy `~astropy.units.Quantity` values at the public API
boundary must give bit-identical results to passing plain SI floats.
Quantities are converted once, at construction; nothing downstream ever
sees a unit.
"""

import numpy as np
import pytest
from astropy import units as u
from conftest import SOURCE_L, SOURCE_M, zenith_phase_center

from rfi_simulator import (
    ArrayConfig,
    PointSource,
    VoltageSimulator,
    correlate,
    dirty_image,
)

POSITIONS_M = np.array(
    [
        [0.0, 0.0, 0.0],
        [12.4, -8.7, 0.0],
        [-19.3, 5.2, 0.0],
        [27.8, 21.6, 0.0],
        [-33.1, -14.9, 0.0],
    ]
)

CENTER_FREQ_HZ = 1.405e9
CHAN_WIDTH_HZ = 30517.578125
FLUX_JY = 2.5
NOISE_STD = 1.5


def build(start_time, *, quantities: bool):
    """Build the same observation twice, with and without Quantity inputs."""
    if quantities:
        array = ArrayConfig(
            antenna_positions_enu_m=(POSITIONS_M * u.m).to(u.km),
            latitude_deg=37.234 * u.deg,
            longitude_deg=-118.282 * u.deg,
            height_m=1.222 * u.km,
        )
        center_freq = (CENTER_FREQ_HZ * u.Hz).to(u.GHz)
        chan_width = (CHAN_WIDTH_HZ * u.Hz).to(u.kHz)
        flux = (FLUX_JY * u.Jy).to(u.mJy)
        noise_std = NOISE_STD * u.Jy**0.5
    else:
        array = ArrayConfig(
            antenna_positions_enu_m=POSITIONS_M,
            latitude_deg=37.234,
            longitude_deg=-118.282,
            height_m=1222.0,
        )
        center_freq = CENTER_FREQ_HZ
        chan_width = CHAN_WIDTH_HZ
        flux = FLUX_JY
        noise_std = NOISE_STD

    phase_center = zenith_phase_center(array, start_time, duration_s=0.5)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=flux)
    sim = VoltageSimulator(
        array,
        phase_center,
        start_time,
        [source],
        center_freq_hz=center_freq,
        chan_width_hz=chan_width,
        n_chan=16,
        n_blocks=3,
        n_time_per_block=128,
        noise_std=noise_std,
        rng=np.random.default_rng(4321),
    )
    return sim, correlate(sim.blocks())


def test_quantity_inputs_give_identical_results(start_time):
    """Acceptance criterion 6, end to end: same numbers either way."""
    sim_float, vis_float = build(start_time, quantities=False)
    sim_quantity, vis_quantity = build(start_time, quantities=True)

    np.testing.assert_allclose(sim_quantity.freq_hz, sim_float.freq_hz, rtol=0, atol=1e-6)
    assert sim_quantity.sample_period_s == pytest.approx(sim_float.sample_period_s)
    np.testing.assert_array_equal(vis_quantity.data, vis_float.data)

    grid = np.linspace(-0.01, 0.01, 9)
    image_float, _, _ = dirty_image(vis_float, grid, grid)
    image_quantity, _, _ = dirty_image(vis_quantity, grid, grid)
    np.testing.assert_allclose(image_quantity, image_float, rtol=1e-12, atol=1e-15)


def test_scalar_attributes_are_plain_floats(start_time):
    """Nothing downstream of the constructor carries a unit."""
    sim, vis = build(start_time, quantities=True)

    for value in (sim.center_freq_hz, sim.chan_width_hz, sim.noise_std, sim.sample_period_s):
        assert isinstance(value, float)
        assert not isinstance(value, u.Quantity)
    assert isinstance(sim.sources[0].flux_jy, float)
    assert sim.freq_hz.dtype == np.float64
    assert not isinstance(vis.data, u.Quantity)


def test_flux_quantity_matches_plain_float():
    """PointSource flux conversion is exact for the units students will use."""
    from astropy.coordinates import SkyCoord

    coord = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg, frame="icrs")
    assert PointSource(flux_jy=FLUX_JY * u.Jy, coord=coord).flux_jy == FLUX_JY
    assert PointSource(flux_jy=FLUX_JY, coord=coord).flux_jy == FLUX_JY

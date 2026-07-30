"""Tests for rfi_simulator.aircraft.

The transponder is the strictest test of the band-placement rule: its
default carrier is nowhere near the package's default band, so the
default configuration *must* raise rather than contribute nothing.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from conftest import DEFAULT_ARRAY_YAML, zenith_phase_center

from rfi_simulator import (
    ADSB_FREQ_HZ,
    ADSBTransponder,
    SatelliteTransmitter,
    TwoLineElement,
    VoltageSimulator,
    correlate,
)
from rfi_simulator.rfi import path_delays_s

# A cruising aircraft 40 km west, closing from the west at 250 m/s.
AIRCRAFT_POSITION_M = (-40000.0, 15000.0, 11000.0)
AIRCRAFT_VELOCITY_M_S = (250.0, 0.0, 0.0)
IN_BAND_CARRIER_HZ = 1.4052e9


def make_transponder(**kwargs):
    """A loud, frequent transponder placed inside the default band."""
    options = dict(
        position_enu_m=AIRCRAFT_POSITION_M,
        velocity_enu_m_s=AIRCRAFT_VELOCITY_M_S,
        carrier_freq_hz=IN_BAND_CARRIER_HZ,
        bandwidth_hz=2.0e5,
        received_power_jy=1.0e4,
        message_rate_hz=2000.0,
        name="transponder",
    )
    options.update(kwargs)
    return ADSBTransponder(**options)


def make_simulator(array, start_time, rfi_sources, **kwargs):
    """Small-but-real simulator, matching the other interference tests."""
    options = dict(
        n_chan=32,
        n_blocks=3,
        n_time_per_block=400,
        noise_std=0.0,
        rng=np.random.default_rng(20261001),
    )
    options.update(kwargs)
    phase_center = zenith_phase_center(array, start_time, duration_s=0.1)
    return VoltageSimulator(array, phase_center, start_time, [], rfi_sources=rfi_sources, **options)


# ----------------------------------------------------------------------
# Band placement
# ----------------------------------------------------------------------
def test_default_carrier_is_out_of_the_default_band(default_array, start_time):
    """The real transponder frequency raises at the package defaults.

    1090 MHz is over 300 MHz from the default band center, so a
    transponder configured with real-world numbers cannot contribute.
    Raising makes that impossible to overlook; silently emitting nothing
    would look like a working configuration.
    """
    transponder = make_transponder(carrier_freq_hz=ADSB_FREQ_HZ, bandwidth_hz=2.0e6)
    assert ADSB_FREQ_HZ == pytest.approx(1.09e9)

    sim = make_simulator(default_array, start_time, [transponder], n_blocks=1)
    with pytest.raises(ValueError, match="outside the simulated band"):
        sim.block(0)

    # Re-centering the band on the transponder is the documented fix.
    recentered = make_simulator(
        default_array, start_time, [transponder], n_blocks=1, center_freq_hz=ADSB_FREQ_HZ
    )
    assert recentered.block(0).rfi_mask[0].any()


def test_bursts_occupy_only_the_carrier_channels(default_array, start_time):
    """Emission is confined to the channels the burst spectrum covers."""
    transponder = make_transponder()
    sim = make_simulator(default_array, start_time, [transponder], n_blocks=1)
    block = sim.block(0)

    expected = np.abs(sim.freq_hz - IN_BAND_CARRIER_HZ) <= 0.5 * transponder.bandwidth_hz
    assert expected.sum() > 1

    occupied = np.abs(block.data).max(axis=(0, 2)) > 0.0
    np.testing.assert_array_equal(occupied, expected)
    np.testing.assert_array_equal(block.rfi_mask[0].any(axis=1), expected)


# ----------------------------------------------------------------------
# Pulse statistics
# ----------------------------------------------------------------------
def test_burst_rate_matches_the_requested_message_rate(default_array, start_time):
    """Burst counts follow a Poisson process with the requested mean."""
    rate_hz = 2000.0
    transponder = make_transponder(message_rate_hz=rate_hz)
    sim = make_simulator(
        default_array, start_time, [transponder], n_blocks=20, n_time_per_block=200, n_chan=8
    )

    counts = [
        transponder.draw_burst_samples(sim.block_context(index, sim.rfi_block_rngs(index)[0])).size
        for index in range(sim.n_blocks)
    ]
    total = int(np.sum(counts))
    expected = rate_hz * sim.n_blocks * sim.block_duration_s

    assert expected > 100, "too few bursts for the statistical test to bite"
    assert abs(total - expected) < 4.0 * np.sqrt(expected)
    assert len(set(counts)) > 1


def test_bursts_are_one_sample_long_by_default(default_array, start_time):
    """A burst occupies a single time sample, and the mask says so."""
    transponder = make_transponder(message_rate_hz=200.0)
    assert transponder.pulse_width_samples == 1

    sim = make_simulator(default_array, start_time, [transponder], n_blocks=4)
    fired = 0
    for index in range(sim.n_blocks):
        block = sim.block(index)
        flagged = block.rfi_mask[0].any(axis=0)
        nonzero = np.abs(block.data).max(axis=(0, 1)) > 0.0
        np.testing.assert_array_equal(flagged, nonzero)

        # Sparse in time: the aircraft is off far more often than on.
        assert flagged.mean() < 0.5
        fired += int(flagged.sum())

        starts = transponder.draw_burst_samples(
            sim.block_context(index, sim.rfi_block_rngs(index)[0])
        )
        for start in starts:
            assert flagged[int(start)]
    assert fired > 0


def test_wider_pulses_flag_consecutive_samples(default_array, start_time):
    """`pulse_width_samples` widens each burst, contiguously."""
    transponder = make_transponder(message_rate_hz=100.0, pulse_width_samples=3)
    sim = make_simulator(default_array, start_time, [transponder], n_blocks=1)
    flagged = sim.block(0).rfi_mask[0].any(axis=0)

    starts = transponder.draw_burst_samples(sim.block_context(0, sim.rfi_block_rngs(0)[0]))
    assert starts.size > 0
    for start in starts:
        stop = min(int(start) + 3, flagged.size)
        assert flagged[int(start) : stop].all()


def test_burst_power_is_measured_at_the_array_origin(default_array, start_time):
    """Band-summed power during a burst equals received_power_jy at the origin."""
    transponder = make_transponder(received_power_jy=8000.0, message_rate_hz=5000.0)
    sim = make_simulator(
        default_array, start_time, [transponder], n_blocks=1, n_time_per_block=2000
    )
    block = sim.block(0)
    active = block.rfi_mask[0].any(axis=0)
    assert active.sum() > 50

    band_power_jy = (np.abs(block.data[0]) ** 2).sum(axis=0)[active].mean()
    assert band_power_jy == pytest.approx(8000.0, rel=0.1)


# ----------------------------------------------------------------------
# Motion
# ----------------------------------------------------------------------
def test_the_aircraft_moves_between_blocks(default_array, start_time):
    """Position advances linearly, and the delays follow it."""
    transponder = make_transponder()
    sim = make_simulator(default_array, start_time, [transponder], n_blocks=5)

    positions = [
        transponder.block_position_enu_m(sim.block_context(index, np.random.default_rng(0)))
        for index in range(sim.n_blocks)
    ]
    steps = np.diff(np.stack(positions), axis=0)
    expected_step = np.asarray(AIRCRAFT_VELOCITY_M_S) * sim.block_duration_s
    np.testing.assert_allclose(steps, np.broadcast_to(expected_step, steps.shape), rtol=1e-9)

    # Start of the observation is the reference epoch, half a block back.
    np.testing.assert_allclose(
        positions[0],
        np.asarray(AIRCRAFT_POSITION_M) + expected_step * 0.5,
        rtol=1e-9,
    )

    # Moving the aircraft moves the delays.
    first = path_delays_s(positions[0], default_array.antenna_positions_enu_m)
    last = path_delays_s(positions[-1], default_array.antenna_positions_enu_m)
    assert np.abs(first - last).max() > 0.0


def test_positions_accept_quantities(default_array, start_time):
    """Quantity inputs match plain floats exactly."""
    plain = make_transponder()
    with_units = make_transponder(
        position_enu_m=np.asarray(AIRCRAFT_POSITION_M) / 1000.0 * u.km,
        velocity_enu_m_s=np.asarray(AIRCRAFT_VELOCITY_M_S) * u.m / u.s,
        received_power_jy=1.0e4 * u.Jy,
    )
    np.testing.assert_allclose(with_units.position_enu_m, plain.position_enu_m)

    def run(source):
        return make_simulator(default_array, start_time, [source], n_blocks=1).block(0).data

    np.testing.assert_array_equal(run(plain), run(with_units))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bandwidth_hz": -1.0}, "bandwidth_hz"),
        ({"received_power_jy": -1.0}, "received_power_jy"),
        ({"message_rate_hz": -1.0}, "message_rate_hz"),
        ({"pulse_width_samples": 0}, "pulse_width_samples"),
    ],
)
def test_transponder_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_transponder(**kwargs)


# ----------------------------------------------------------------------
# Framework contracts still hold
# ----------------------------------------------------------------------
def test_transponder_preserves_purity_and_pairing(default_array, start_time):
    """Acceptance criteria 2 and 7 hold with a transponder attached."""
    transponder = make_transponder(message_rate_hz=500.0)

    def build(with_rfi):
        return make_simulator(
            default_array,
            start_time,
            [transponder] if with_rfi else [],
            n_blocks=3,
            noise_std=1.0,
            rng=np.random.default_rng(8191),
        )

    clean, dirty = build(False), build(True)

    touched = False
    for index in range(3):
        clean_block, dirty_block = clean.block(index), dirty.block(index)
        union = dirty_block.rfi_mask.any(axis=0)
        np.testing.assert_array_equal(dirty_block.data[:, ~union], clean_block.data[:, ~union])
        touched |= bool(union.any())
    assert touched

    reference = [block.data for block in build(True).blocks()]
    shuffled = build(True)
    _ = shuffled.block(2)
    for index in (1, 0, 2):
        np.testing.assert_array_equal(shuffled.block(index).data, reference[index])


def test_transponder_labels_reach_the_visibilities(default_array, start_time):
    """Acceptance criterion 6 for a pulsed source: occupancy is a fraction."""
    transponder = make_transponder(message_rate_hz=1000.0)
    sim = make_simulator(default_array, start_time, [transponder], n_blocks=3)
    blocks = [sim.block(index) for index in range(3)]
    vis = correlate(iter(blocks))

    assert vis.rfi_source_names == ("transponder",)
    assert vis.rfi_fraction.shape == (3, 1, sim.n_chan)
    for index, block in enumerate(blocks):
        np.testing.assert_array_equal(
            vis.rfi_fraction[index], block.rfi_mask.mean(axis=2, dtype=np.float64)
        )
    # Pulsed, so occupancy is a small fraction of the integration -- never 1.
    occupied = vis.rfi_fraction.max(axis=(0, 1)) > 0.0
    assert occupied.any()
    assert 0.0 < vis.rfi_fraction.max() < 0.5


def test_transponder_and_satellite_coexist(default_array, start_time):
    """Several moving sources label independently and add linearly."""
    tle = TwoLineElement.from_file(DEFAULT_ARRAY_YAML.parent / "tle_sample.txt")
    satellite = SatelliteTransmitter(
        tle, carrier_freq_hz=1.4048e9, received_power_jy=200.0, name="downlink"
    )
    transponder = make_transponder(message_rate_hz=500.0)

    epoch = Time("2026-07-30T06:00:00", scale="utc")
    both = make_simulator(default_array, epoch, [satellite, transponder], n_blocks=2)
    satellite_only = make_simulator(default_array, epoch, [satellite], n_blocks=2)

    assert both.block(0).rfi_source_names == ("downlink", "transponder")
    # The satellite's own labels are unaffected by its new neighbour.
    np.testing.assert_array_equal(both.block(0).rfi_mask[0], satellite_only.block(0).rfi_mask[0])
    # Their occupied channels are disjoint, so the masks do not overlap.
    assert not (both.block(0).rfi_mask[0] & both.block(0).rfi_mask[1]).any()

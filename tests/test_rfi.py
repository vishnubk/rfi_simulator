"""Tests for rfi_simulator.rfi and its wiring into the simulator.

The tests that carry the most weight here are:

* `test_near_field_delays_match_hand_computation` -- the whole point of
  modelling interference at the voltage level is that a nearby
  transmitter's wavefront is curved. The expected delays are worked out
  on paper in that test's docstring, and a plane-wave approximation
  fails one of the two cases by a factor of infinity (it predicts zero).
* `test_clean_and_contaminated_runs_share_the_sky_realization` -- labels
  are only meaningful if a clean/contaminated pair differs by the
  interference and nothing else.
"""

import numpy as np
import pytest
from astropy import units as u
from conftest import SOURCE_L, SOURCE_M, zenith_phase_center

from rfi_simulator import (
    ArrayConfig,
    ImpulsiveBroadband,
    NarrowbandTransmitter,
    PointSource,
    VoltageSimulator,
    correlate,
    dirty_image,
    enu_from_geodetic,
    enu_from_horizontal,
    path_delays_s,
)
from rfi_simulator.delays import SPEED_OF_LIGHT_M_S
from rfi_simulator.rfi import (
    OCCUPANCY_THRESHOLD,
    occupancy_mask,
    spreading_amplitudes,
)

# A transmitter due east of the array, just above the horizon.
TOWER_ENU_M = np.array([2000.0, 0.0, 0.0])
TOWER_CENTER_FREQ_HZ = 1.4053e9
TOWER_BANDWIDTH_HZ = 1.5e5


def make_tower(**kwargs):
    """A narrowband transmitter 2 km east, on by default."""
    options = dict(
        position_enu_m=TOWER_ENU_M,
        center_freq_hz=TOWER_CENTER_FREQ_HZ,
        bandwidth_hz=TOWER_BANDWIDTH_HZ,
        received_power_jy=200.0,
        name="tower",
    )
    options.update(kwargs)
    return NarrowbandTransmitter(**options)


def make_simulator(array, start_time, sources=(), rfi_sources=(), **kwargs):
    """Small-but-real simulator, mirroring the one in test_voltages."""
    options = dict(
        n_chan=32,
        n_blocks=3,
        n_time_per_block=200,
        noise_std=0.0,
        rng=np.random.default_rng(20261001),
    )
    options.update(kwargs)
    phase_center = zenith_phase_center(array, start_time, duration_s=0.1)
    return VoltageSimulator(
        array, phase_center, start_time, sources, rfi_sources=rfi_sources, **options
    )


def three_antenna_array() -> ArrayConfig:
    """Antennas at the origin, 100 m east and 100 m north.

    Deliberately hand-chosen positions so the near-field path lengths are
    exact decimal numbers (see
    `test_near_field_delays_match_hand_computation`).
    """
    return ArrayConfig(
        antenna_positions_enu_m=np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]),
        latitude_deg=37.234,
        longitude_deg=-118.282,
        height_m=1222.0,
        name="hand_computed",
    )


# ----------------------------------------------------------------------
# Geometry: near-field path delays (acceptance criterion 3)
# ----------------------------------------------------------------------
def test_near_field_delays_match_hand_computation():
    r"""Exact path delays for a transmitter 2 km east, computed on paper.

    Geometry, all in local ENU meters:

    * transmitter at ``(2000, 0, 0)``,
    * antenna A at the origin ``(0, 0, 0)``,
    * antenna B 100 m east, ``(100, 0, 0)``,
    * antenna C 100 m north, ``(0, 100, 0)``.

    Path lengths:

    * ``|x - r_A| = 2000`` m exactly.
    * ``|x - r_B| = 2000 - 100 = 1900`` m exactly, so the A-B path
      difference is **100.000000 m**, i.e.
      ``100 / c = 3.335640951981520e-7 s``. B lies along the line to the
      transmitter, so here the exact and plane-wave answers agree --
      this leg checks the scale and sign.
    * ``|x - r_C| = sqrt(2000**2 + 100**2) = sqrt(4010000)
      = 2002.4984394500783`` m, so the A-C path difference is
      **-2.4984394500783 m**, i.e. ``-8.333894...e-9 s``. A plane wave
      arriving from due east would predict **exactly zero** here, because
      the baseline is perpendicular to the source direction. The 2.5 m
      discrepancy -- 12 wavelengths at 1.4 GHz -- is the wavefront
      curvature, and it is why this package never plane-wave approximates
      an interference source.
    """
    array = three_antenna_array()
    tau_s = path_delays_s(TOWER_ENU_M, array.antenna_positions_enu_m)

    expected_lengths_m = np.array([2000.0, 1900.0, np.sqrt(4010000.0)])
    np.testing.assert_allclose(tau_s, expected_lengths_m / SPEED_OF_LIGHT_M_S, rtol=0.0, atol=1e-18)

    # The two hand-computed differences, spelled out.
    assert tau_s[0] - tau_s[1] == pytest.approx(100.0 / SPEED_OF_LIGHT_M_S, rel=1e-12)
    assert tau_s[0] - tau_s[2] == pytest.approx(-2.4984394500783 / SPEED_OF_LIGHT_M_S, rel=1e-9)

    # A plane wave from due east predicts zero on the north baseline; the
    # exact geometry predicts 8.3 ns. This is the near-field signature.
    plane_wave_prediction_s = 0.0
    assert abs((tau_s[0] - tau_s[2]) - plane_wave_prediction_s) > 8e-9


def test_spreading_amplitudes_are_normalized_at_the_origin():
    """The stated received power is the power at the array origin."""
    array = three_antenna_array()
    amplitudes = spreading_amplitudes(TOWER_ENU_M, array.antenna_positions_enu_m)

    assert amplitudes[0] == pytest.approx(1.0, rel=1e-15)
    # The closer antenna is louder, by exactly the distance ratio.
    assert amplitudes[1] == pytest.approx(2000.0 / 1900.0, rel=1e-12)
    assert amplitudes[2] == pytest.approx(2000.0 / np.sqrt(4010000.0), rel=1e-12)


def test_degenerate_transmitter_positions_are_rejected():
    array = three_antenna_array()
    with pytest.raises(ValueError, match="array origin"):
        spreading_amplitudes(np.zeros(3), array.antenna_positions_enu_m)
    with pytest.raises(ValueError, match="antenna position"):
        spreading_amplitudes(np.array([100.0, 0.0, 0.0]), array.antenna_positions_enu_m)


def test_horizontal_and_geodetic_positions_agree(default_array):
    """The two ways of naming a transmitter position describe the same point."""
    east = enu_from_horizontal(90.0, 0.0, 2000.0)
    np.testing.assert_allclose(east, [2000.0, 0.0, 0.0], atol=1e-9)

    north = enu_from_horizontal(0.0, 0.0, 500.0)
    np.testing.assert_allclose(north, [0.0, 500.0, 0.0], atol=1e-9)

    # A point ~1 km north of the array origin in latitude lands ~1 km
    # north in ENU, with a small downward "Up" from Earth curvature.
    offset_deg = 1000.0 / 111320.0
    position = enu_from_geodetic(
        default_array.latitude_deg + offset_deg,
        default_array.longitude_deg,
        default_array.height_m,
        default_array,
    )
    assert abs(position[0]) < 1.0
    assert position[1] == pytest.approx(1000.0, rel=5e-3)
    assert position[2] == pytest.approx(-(1000.0**2) / (2 * 6.371e6), abs=0.05)

    with pytest.raises(ValueError, match="distance_m"):
        enu_from_horizontal(0.0, 0.0, 0.0)


def test_units_at_the_api_boundary(default_array, start_time):
    """Quantity inputs give exactly the same source as plain floats."""
    plain = make_tower()
    with_units = make_tower(
        position_enu_m=TOWER_ENU_M * u.m,
        center_freq_hz=TOWER_CENTER_FREQ_HZ / 1e6 * u.MHz,
        bandwidth_hz=TOWER_BANDWIDTH_HZ / 1e3 * u.kHz,
        received_power_jy=200.0 * u.Jy,
    )
    assert with_units.center_freq_hz == pytest.approx(plain.center_freq_hz)
    assert with_units.bandwidth_hz == pytest.approx(plain.bandwidth_hz)

    def run(tower):
        sim = make_simulator(default_array, start_time, rfi_sources=[tower], n_blocks=1)
        return sim.block(0).data

    np.testing.assert_array_equal(run(plain), run(with_units))


# ----------------------------------------------------------------------
# Narrowband occupancy and labels (acceptance criterion 1)
# ----------------------------------------------------------------------
def test_narrowband_occupies_exactly_the_expected_channels(default_array, start_time):
    """Acceptance criterion 1: the emission lands in the expected channels only."""
    tower = make_tower()
    sim = make_simulator(default_array, start_time, rfi_sources=[tower], n_blocks=1)
    block = sim.block(0)

    expected = np.abs(sim.freq_hz - TOWER_CENTER_FREQ_HZ) <= 0.5 * TOWER_BANDWIDTH_HZ
    assert expected.sum() > 0

    occupied = np.abs(block.data).max(axis=(0, 2)) > 0.0
    np.testing.assert_array_equal(occupied, expected)

    # The mask says the same thing, on every time sample (duty cycle 1).
    assert block.rfi_mask.shape == (1, sim.n_chan, sim.n_time_per_block)
    assert block.rfi_source_names == ("tower",)
    np.testing.assert_array_equal(block.rfi_mask[0].any(axis=1), expected)
    assert block.rfi_mask[0][expected].all()


def test_narrowband_received_power_is_measured_at_the_array_origin(default_array, start_time):
    """Band-summed power at the origin antenna equals received_power_jy."""
    array = ArrayConfig(
        antenna_positions_enu_m=np.array([[0.0, 0.0, 0.0], [40.0, -25.0, 0.0]]),
        latitude_deg=37.234,
        longitude_deg=-118.282,
        height_m=1222.0,
    )
    tower = make_tower(received_power_jy=400.0)
    sim = make_simulator(array, start_time, rfi_sources=[tower], n_time_per_block=4000)
    data = sim.block(0).data

    band_power_jy = (np.abs(data[0]) ** 2).sum(axis=0).mean()
    assert band_power_jy == pytest.approx(400.0, rel=0.05)


def test_narrowband_outside_the_band_raises(default_array, start_time):
    """Emitting outside the simulated band is an error, not a silent no-op."""
    far_tower = make_tower(center_freq_hz=1.09e9, bandwidth_hz=2.0e6)
    sim = make_simulator(default_array, start_time, rfi_sources=[far_tower], n_blocks=1)
    with pytest.raises(ValueError, match="outside the simulated band"):
        sim.block(0)


def test_duty_cycle_switches_the_transmitter_in_frames(default_array, start_time):
    """Half-duty frames are off for about half the samples, in blocks."""
    frame_duration_s = 0.001
    tower = make_tower(duty_cycle=0.5, frame_duration_s=frame_duration_s)
    sim = make_simulator(
        default_array, start_time, rfi_sources=[tower], n_blocks=8, n_time_per_block=1000
    )
    masks = np.stack([block.rfi_mask[0] for block in sim.blocks()])

    on_fraction = masks.any(axis=1).mean()
    n_frames_total = 8 * int(np.ceil(1000 * sim.sample_period_s / frame_duration_s))
    assert on_fraction == pytest.approx(0.5, abs=4.0 / np.sqrt(n_frames_total))

    # Samples within one frame share a state: the number of transitions is
    # far smaller than the number of samples.
    on_per_sample = masks[0].any(axis=0)
    transitions = int(np.count_nonzero(np.diff(on_per_sample)))
    assert transitions < on_per_sample.size // 10

    # Duty cycle 0 emits nothing at all.
    silent = make_tower(duty_cycle=0.0)
    quiet_sim = make_simulator(default_array, start_time, rfi_sources=[silent], n_blocks=1)
    quiet_block = quiet_sim.block(0)
    assert not quiet_block.rfi_mask.any()
    np.testing.assert_array_equal(quiet_block.data, np.zeros_like(quiet_block.data))


def test_occupancy_mask_uses_the_documented_threshold():
    """Cells below 1 % of the block's peak power are not labelled."""
    envelope = np.array([[100.0, 1.5, 0.5, 0.0]])
    np.testing.assert_array_equal(occupancy_mask(envelope), [[True, True, False, False]])
    assert OCCUPANCY_THRESHOLD == 0.01

    # A silent block is labelled all-clean rather than dividing by zero.
    assert not occupancy_mask(np.zeros((2, 3))).any()


def test_sky_source_still_images_correctly_under_contamination(default_array, start_time):
    """Acceptance criterion 1: moderate interference does not move the source."""
    pixel_rad = 2e-4
    grid = np.arange(-60, 61) * pixel_rad

    phase_center = zenith_phase_center(default_array, start_time, duration_s=0.5)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=1.0)
    tower = make_tower(received_power_jy=5.0)

    def peak(rfi_sources):
        sim = VoltageSimulator(
            default_array,
            phase_center,
            start_time,
            [source],
            rfi_sources=rfi_sources,
            n_chan=32,
            n_blocks=4,
            n_time_per_block=500,
            noise_std=0.0,
            rng=np.random.default_rng(4242),
        )
        image, l_grid, m_grid = dirty_image(correlate(sim.blocks()), grid, grid)
        i_m, i_l = np.unravel_index(np.argmax(image), image.shape)
        return l_grid[i_l], m_grid[i_m]

    l_clean, m_clean = peak([])
    l_dirty, m_dirty = peak([tower])

    assert abs(l_dirty - SOURCE_L) <= 0.5 * pixel_rad
    assert abs(m_dirty - SOURCE_M) <= 0.5 * pixel_rad
    assert (l_dirty, m_dirty) == (l_clean, m_clean)


# ----------------------------------------------------------------------
# Clean/contaminated pairing (acceptance criterion 2)
# ----------------------------------------------------------------------
def test_clean_and_contaminated_runs_share_the_sky_realization(default_array, start_time):
    """Acceptance criterion 2: the pair differs only where the masks say.

    Same seed, one run with interference and one without: outside the
    union of the occupancy masks the voltages must be *bit-identical*,
    which is only true if the interference seeds come from their own
    branch of the seed tree.
    """
    phase_center = zenith_phase_center(default_array, start_time, duration_s=0.1)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=1.0)
    rfi_sources = [
        make_tower(duty_cycle=0.5, frame_duration_s=0.001),
        ImpulsiveBroadband(rate_hz=400.0, received_power_jy=500.0, name="sparks"),
    ]

    def build(with_rfi):
        return make_simulator(
            default_array,
            start_time,
            [source],
            rfi_sources=rfi_sources if with_rfi else [],
            noise_std=1.0,
            n_blocks=4,
            rng=np.random.default_rng(31415),
        )

    clean, dirty = build(False), build(True)

    touched_any = False
    for index in range(clean.n_blocks):
        clean_block, dirty_block = clean.block(index), dirty.block(index)
        union = dirty_block.rfi_mask.any(axis=0)

        np.testing.assert_array_equal(dirty_block.data[:, ~union], clean_block.data[:, ~union])
        if union.any():
            touched_any = True
            assert not np.array_equal(dirty_block.data[:, union], clean_block.data[:, union])

    assert touched_any, "the interference never fired; the test would be vacuous"


def test_adding_a_source_does_not_disturb_the_others(default_array, start_time):
    """Each source draws from its own generator, so they do not interact."""
    tower = make_tower(duty_cycle=0.5, frame_duration_s=0.001)
    sparks = ImpulsiveBroadband(rate_hz=400.0, received_power_jy=500.0, name="sparks")

    alone = make_simulator(default_array, start_time, rfi_sources=[tower], n_blocks=2)
    together = make_simulator(default_array, start_time, rfi_sources=[tower, sparks], n_blocks=2)

    for index in range(2):
        np.testing.assert_array_equal(
            alone.block(index).rfi_mask[0], together.block(index).rfi_mask[0]
        )


def test_zero_sources_leave_the_block_labelled_clean(default_array, start_time):
    """A clean run still carries labels -- an empty source axis."""
    block = make_simulator(default_array, start_time, n_blocks=1).block(0)
    assert block.rfi_mask.shape == (0, block.n_chan, block.n_time)
    assert block.rfi_source_names == ()
    assert block.n_rfi_sources == 0


# ----------------------------------------------------------------------
# Visibility-domain behaviour (acceptance criterion 3)
# ----------------------------------------------------------------------
def test_visibility_phase_slope_recovers_the_path_delay_difference(start_time):
    """Acceptance criterion 3: visibility phases match the exact geometry.

    A noiseless, un-fringe-stopped visibility of a single transmitter is
    ``exp(-2 pi i f (tau_i - tau_j))``, so the slope of its unwrapped
    phase against frequency measures the path-delay difference directly.
    Both hand-computed differences of
    `test_near_field_delays_match_hand_computation` are recovered,
    including the 8.3 ns one that a plane wave says should be zero.
    """
    array = three_antenna_array()
    wideband_tower = make_tower(center_freq_hz=1.405e9, bandwidth_hz=1.0e7, received_power_jy=100.0)
    sim = make_simulator(array, start_time, rfi_sources=[wideband_tower], n_chan=64, n_blocks=1)
    vis = correlate(sim.blocks(), fringe_stop=False)

    tau_s = path_delays_s(TOWER_ENU_M, array.antenna_positions_enu_m)
    for ant_1, ant_2 in ((0, 1), (0, 2)):
        row = vis.data[0, vis.baseline_index(ant_1, ant_2)]
        phase = np.unwrap(np.angle(row))
        slope = np.polyfit(vis.freq_hz - vis.freq_hz.mean(), phase, 1)[0]
        measured_delay_s = -slope / (2.0 * np.pi)
        assert measured_delay_s == pytest.approx(tau_s[ant_1] - tau_s[ant_2], rel=1e-4)


def test_terrestrial_source_fringe_winds_after_fringe_stopping(start_time):
    """Acceptance criterion 3: interference is not stopped by tracking the field.

    Fringe stopping removes the phase-center delay, which is the right
    thing to do for the sky and the wrong thing for a transmitter bolted
    to the ground -- so terrestrial interference winds in phase across the
    observation. That winding is the signature, not a defect; the test
    asserts it is present, and that a source *at* the phase center shows
    none.
    """
    array = three_antenna_array()
    phase_center = zenith_phase_center(array, start_time, duration_s=2.0)
    common = dict(n_chan=4, n_blocks=61, n_time_per_block=1000, noise_std=0.0)

    tower = make_tower(received_power_jy=100.0, bandwidth_hz=1.0e7)
    interference = VoltageSimulator(
        array,
        phase_center,
        start_time,
        [],
        rfi_sources=[tower],
        rng=np.random.default_rng(5),
        **common,
    )
    sky = VoltageSimulator(
        array,
        phase_center,
        start_time,
        [PointSource.from_lm(phase_center, (0.0, 0.0), flux_jy=1.0)],
        rng=np.random.default_rng(5),
        **common,
    )

    def phase_drift_rad(sim):
        vis = correlate(sim.blocks())
        row = vis.data[:, vis.baseline_index(0, 1), 0]
        return np.unwrap(np.angle(row))

    interference_phase = phase_drift_rad(interference)
    sky_phase = phase_drift_rad(sky)

    assert abs(interference_phase[-1] - interference_phase[0]) > 0.1
    assert np.abs(sky_phase - sky_phase[0]).max() < 0.01


# ----------------------------------------------------------------------
# Impulsive events (acceptance criterion 5)
# ----------------------------------------------------------------------
def test_impulsive_event_count_matches_the_poisson_rate(default_array, start_time):
    """Acceptance criterion 5: the realized rate matches the requested one."""
    rate_hz = 3000.0
    sparks = ImpulsiveBroadband(rate_hz=rate_hz, received_power_jy=100.0, max_power_ratio=1.0)
    sim = make_simulator(
        default_array,
        start_time,
        rfi_sources=[sparks],
        n_blocks=30,
        n_time_per_block=200,
        n_chan=8,
    )

    counts = []
    for index in range(sim.n_blocks):
        ctx = sim.block_context(index, sim.rfi_block_rngs(index)[0])
        starts, _ = sparks.draw_events(ctx)
        counts.append(starts.size)

    total = int(np.sum(counts))
    expected = rate_hz * sim.n_blocks * sim.block_duration_s
    assert expected > 100, "too few events for the statistical test to bite"
    assert abs(total - expected) < 4.0 * np.sqrt(expected)

    # Blocks are independent draws, not one repeated realization.
    assert len(set(counts)) > 1


def test_impulsive_mask_flags_the_event_samples(default_array, start_time):
    """Acceptance criterion 5: the mask marks exactly the samples that fired."""
    sparks = ImpulsiveBroadband(
        rate_hz=2000.0,
        received_power_jy=100.0,
        max_power_ratio=1.0,
        pulse_width_samples=2,
        name="sparks",
    )
    sim = make_simulator(
        default_array, start_time, rfi_sources=[sparks], n_blocks=4, n_time_per_block=400
    )

    fired_anywhere = False
    for index in range(sim.n_blocks):
        block = sim.block(index)
        mask = block.rfi_mask[0]
        # Flat across the band: a flagged sample is flagged in every channel.
        np.testing.assert_array_equal(mask.all(axis=0), mask.any(axis=0))

        nonzero_samples = np.abs(block.data).max(axis=(0, 1)) > 0.0
        np.testing.assert_array_equal(mask.any(axis=0), nonzero_samples)
        fired_anywhere |= bool(nonzero_samples.any())

        ctx = sim.block_context(index, sim.rfi_block_rngs(index)[0])
        starts, _ = sparks.draw_events(ctx)
        for start in starts:
            assert mask[0, int(start)]

    assert fired_anywhere


def test_impulsive_events_do_not_repeat_between_blocks(default_array, start_time):
    """Events correlate between blocks only through the rate."""
    sparks = ImpulsiveBroadband(rate_hz=3000.0, received_power_jy=100.0)
    sim = make_simulator(
        default_array, start_time, rfi_sources=[sparks], n_blocks=3, n_time_per_block=400
    )
    masks = [sim.block(index).rfi_mask[0].any(axis=0) for index in range(3)]
    assert not np.array_equal(masks[0], masks[1])
    assert not np.array_equal(masks[1], masks[2])


def test_impulsive_power_law_spans_the_requested_range(default_array, start_time):
    """Event powers fill [received_power_jy, max_power_ratio * it], tail-heavy."""
    sparks = ImpulsiveBroadband(
        rate_hz=20000.0, received_power_jy=10.0, max_power_ratio=100.0, power_law_index=2.0
    )
    sim = make_simulator(
        default_array, start_time, rfi_sources=[sparks], n_blocks=1, n_time_per_block=1000
    )
    ctx = sim.block_context(0, sim.rfi_block_rngs(0)[0])
    _, powers_jy = sparks.draw_events(ctx)

    assert powers_jy.size > 100
    assert powers_jy.min() >= 10.0
    assert powers_jy.max() <= 1000.0
    # p(x) ~ x^-2 on [1, 100]: the median event is at about 1.98x the
    # minimum, so most events are weak and a few are not.
    assert np.median(powers_jy) < 30.0
    assert powers_jy.max() > 100.0


def test_impulsive_uses_a_deterministic_default_position():
    """The default position is fixed, so the source does not wander per block."""
    first = ImpulsiveBroadband(rate_hz=1.0)
    second = ImpulsiveBroadband(rate_hz=1.0)
    np.testing.assert_array_equal(first.position_enu_m, second.position_enu_m)
    # Low elevation, a few km away: over the horizon, not overhead.
    assert np.linalg.norm(first.position_enu_m) == pytest.approx(5000.0)
    assert first.position_enu_m[2] / 5000.0 == pytest.approx(np.sin(np.deg2rad(1.0)), rel=1e-9)


# ----------------------------------------------------------------------
# Label propagation to visibilities (acceptance criterion 6)
# ----------------------------------------------------------------------
def test_rfi_fraction_is_empty_for_a_clean_run(default_array, start_time):
    """Acceptance criterion 6: no interference, no occupancy to report."""
    sim = make_simulator(default_array, start_time, n_blocks=3)
    vis = correlate(sim.blocks())

    assert vis.rfi_fraction.shape == (3, 0, sim.n_chan)
    assert vis.rfi_source_names == ()
    assert vis.n_rfi_sources == 0
    assert (vis.rfi_fraction == 0.0).all()
    assert vis.rfi_fraction.max(initial=0.0) == 0.0


def test_rfi_fraction_matches_the_block_occupancy(default_array, start_time):
    """Acceptance criterion 6: the fraction is the mask averaged over time."""
    tower = make_tower(duty_cycle=0.5, frame_duration_s=0.001)
    sparks = ImpulsiveBroadband(rate_hz=2000.0, received_power_jy=200.0, name="sparks")
    sim = make_simulator(
        default_array,
        start_time,
        rfi_sources=[tower, sparks],
        n_blocks=4,
        n_time_per_block=400,
    )

    blocks = [sim.block(index) for index in range(sim.n_blocks)]
    vis = correlate(iter(blocks))

    assert vis.rfi_fraction.shape == (4, 2, sim.n_chan)
    assert vis.rfi_source_names == ("tower", "sparks")
    for index, block in enumerate(blocks):
        np.testing.assert_allclose(
            vis.rfi_fraction[index], block.rfi_mask.mean(axis=2), rtol=0.0, atol=0.0
        )
    assert (vis.rfi_fraction >= 0.0).all() and (vis.rfi_fraction <= 1.0).all()
    assert vis.rfi_fraction[:, 0].max() > 0.0

    # The narrowband source only ever occupies its own channels.
    expected = np.abs(sim.freq_hz - TOWER_CENTER_FREQ_HZ) <= 0.5 * TOWER_BANDWIDTH_HZ
    assert not vis.rfi_fraction[:, 0][:, ~expected].any()


def test_correlate_rejects_inconsistent_labels(default_array, start_time):
    """Mixing labelled and unlabelled blocks is a configuration error."""
    clean = make_simulator(default_array, start_time, n_blocks=1).block(0)
    dirty = make_simulator(default_array, start_time, rfi_sources=[make_tower()], n_blocks=1).block(
        0
    )
    with pytest.raises(ValueError, match="same interference-source labels"):
        correlate([clean, dirty])


def test_block_rejects_a_mismatched_mask(default_array, start_time):
    """The block will not carry labels that do not match its source list."""
    block = make_simulator(default_array, start_time, n_blocks=1).block(0)
    with pytest.raises(ValueError, match="rfi_mask must have shape"):
        type(block)(
            data=block.data,
            time=block.time,
            center_time=block.center_time,
            freq_hz=block.freq_hz,
            sample_period_s=block.sample_period_s,
            phase_center_delays_s=block.phase_center_delays_s,
            antenna_positions_enu_m=block.antenna_positions_enu_m,
            e_l_enu=block.e_l_enu,
            e_m_enu=block.e_m_enu,
            s0_enu=block.s0_enu,
            rfi_mask=np.zeros((2, block.n_chan, block.n_time), dtype=bool),
            rfi_source_names=("only_one",),
        )


# ----------------------------------------------------------------------
# Reproducibility with interference present (acceptance criterion 7)
# ----------------------------------------------------------------------
def test_block_is_pure_in_seed_and_index_with_interference(default_array, start_time):
    """Acceptance criterion 7: `block(i)` purity survives interference sources.

    Interference adds two more stateful things -- duty-cycle frames and
    Poisson events -- and `SeedSequence.spawn` is itself stateful, so this
    is exactly where out-of-order generation would start returning
    different data.
    """
    phase_center = zenith_phase_center(default_array, start_time, duration_s=0.1)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=1.0)
    rfi_sources = [
        make_tower(duty_cycle=0.5, frame_duration_s=0.001),
        ImpulsiveBroadband(rate_hz=1000.0, received_power_jy=300.0, name="sparks"),
    ]

    def new_sim():
        return make_simulator(
            default_array,
            start_time,
            [source],
            rfi_sources=rfi_sources,
            noise_std=1.0,
            n_blocks=5,
            rng=np.random.default_rng(2718),
        )

    sim = new_sim()
    np.testing.assert_array_equal(sim.block(3).data, sim.block(3).data)
    np.testing.assert_array_equal(sim.block(3).rfi_mask, sim.block(3).rfi_mask)

    reference = [block.data for block in new_sim().blocks()]
    reference_masks = [block.rfi_mask for block in new_sim().blocks()]

    peeked = new_sim()
    _ = peeked.block(4)
    for index in (2, 0, 4, 1, 3):
        block = peeked.block(index)
        np.testing.assert_array_equal(block.data, reference[index])
        np.testing.assert_array_equal(block.rfi_mask, reference_masks[index])

    # And a different seed really does give different interference.
    other = make_simulator(
        default_array,
        start_time,
        [source],
        rfi_sources=rfi_sources,
        noise_std=1.0,
        n_blocks=5,
        rng=np.random.default_rng(1729),
    )
    assert not np.array_equal(other.block(0).data, reference[0])


# ----------------------------------------------------------------------
# Constructor validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bandwidth_hz": -1.0}, "bandwidth_hz"),
        ({"received_power_jy": -1.0}, "received_power_jy"),
        ({"duty_cycle": 1.5}, "duty_cycle"),
        ({"frame_duration_s": 0.0}, "frame_duration_s"),
    ],
)
def test_narrowband_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_tower(**kwargs)


@pytest.mark.parametrize(
    ("param", "match"),
    [
        ("center_freq_hz", "center_freq_hz"),
        ("bandwidth_hz", "bandwidth_hz"),
        ("received_power_jy", "received_power_jy"),
        ("frame_duration_s", "frame_duration_s"),
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_narrowband_rejects_non_finite_parameters(param, match, value):
    """A naive ``< 0`` guard lets NaN and Inf through; these must not.

    NaN and Inf both fail every ``< 0`` comparison, so a transmitter built
    with either would otherwise carry a non-finite parameter into
    `NarrowbandTransmitter.contribution` and emit NaN voltages under a
    ground-truth mask that still reads as clean.
    """
    with pytest.raises(ValueError, match=match):
        make_tower(**{param: value})


def test_narrowband_nan_power_cannot_reach_a_block(default_array, start_time):
    """A NaN `received_power_jy` is rejected at construction time.

    It never survives to be handed to a simulator, let alone reach
    `block`, so it cannot produce NaN voltages under a clean-looking mask.
    """
    with pytest.raises(ValueError, match="received_power_jy"):
        make_tower(received_power_jy=float("nan"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rate_hz": -1.0}, "rate_hz"),
        ({"received_power_jy": -1.0}, "received_power_jy"),
        ({"power_law_index": 1.0}, "power_law_index"),
        ({"max_power_ratio": 0.5}, "max_power_ratio"),
        ({"pulse_width_samples": 0}, "pulse_width_samples"),
    ],
)
def test_impulsive_validation(kwargs, match):
    options = dict(rate_hz=10.0)
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ImpulsiveBroadband(**options)


@pytest.mark.parametrize(
    ("param", "match"),
    [
        ("rate_hz", "rate_hz"),
        ("received_power_jy", "received_power_jy"),
        ("power_law_index", "power_law_index"),
        ("max_power_ratio", "max_power_ratio"),
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_impulsive_rejects_non_finite_parameters(param, match, value):
    """As with the narrowband guards, NaN and Inf must not pass a ``< 0`` check."""
    options = dict(rate_hz=10.0)
    options[param] = value
    with pytest.raises(ValueError, match=match):
        ImpulsiveBroadband(**options)


def test_a_source_returning_the_wrong_shape_is_caught(default_array, start_time):
    """The simulator validates what a plug-in source hands back."""
    from rfi_simulator.rfi import RFISource

    class Broken(RFISource):
        def contribution(self, ctx):
            return np.zeros((1, 1, 1), dtype=np.complex64), np.zeros(
                (ctx.n_chan, ctx.n_time), dtype=bool
            )

    sim = make_simulator(default_array, start_time, rfi_sources=[Broken("broken")], n_blocks=1)
    with pytest.raises(ValueError, match="returned voltages of shape"):
        sim.block(0)

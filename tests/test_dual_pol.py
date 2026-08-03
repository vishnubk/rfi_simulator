"""Tests for dual-polarization synthesis, correlation and imaging.

The physics claim this file exists to protect is a single asymmetry:
**terrestrial interference is polarized and the sky is not**. A transmitter
has a definite polarization state, so the two receptors of a feed see it
with a fixed amplitude ratio and a fixed relative phase; sky and receiver
noise are unpolarized, so the two receptors see independent realizations of
equal power. A flagger can exploit that, and
`test_polarization_ratio_separates_occupied_from_clean_channels` is the
motivating case written out as an assertion.

The other half of the file guards the invariant that matters just as much:
``n_pol=1`` -- the default -- must be **bit-for-bit** what the simulator
produced before polarization existed. The digests in `REFERENCE_DIGESTS`
were recorded by running `reference_simulator` against the package at
commit 2825a9a, before any of this code was written, and they cover the
whole stack: sky, spectral line, narrowband and impulsive interference,
the filterbank, per-antenna gains, 4-bit quantization and the correlator.
"""

import hashlib

import numpy as np
import pytest
from conftest import zenith_phase_center

from rfi_simulator import (
    ImpulsiveBroadband,
    InstrumentModel,
    NarrowbandTransmitter,
    PFBChannelizer,
    PointSource,
    SpectralLineForeground,
    Visibilities,
    VoltageBlock,
    VoltageSimulator,
    correlate,
    dirty_image,
    resolve_polarization,
)
from rfi_simulator.correlator import PARALLEL_HAND_NAMES
from rfi_simulator.io.packed_voltage import (
    PackedVoltageLayout,
    pack_from_voltage_block,
    unpack_block,
)
from rfi_simulator.rfi import RFISource, enu_from_horizontal

# A transmitter this far from the receptor axis splits its power strongly
# but leaves a measurable amount in the weak receptor, which is a harder
# case for a ratio test than 0 deg (where the weak receptor is exactly
# silent).
POLARIZATION_ANGLE_DEG = 15.0


# ----------------------------------------------------------------------
# Scene builders
# ----------------------------------------------------------------------
def reference_simulator(array, start_time, **kwargs):
    """A little of everything, at a fixed seed: the bit-identity scene.

    Sky source, spectral line, narrowband and impulsive interference, a
    filterbank, per-antenna gains and 4-bit quantization -- every stage
    that dual polarization touches, in one configuration.
    """
    phase_center = zenith_phase_center(array, start_time, duration_s=1.0)
    source = PointSource.from_lm(phase_center, (0.004, -0.002), flux_jy=3.0)
    line = SpectralLineForeground(
        center_freq_hz=1.40505e9, fwhm_hz=6.0e4, line_flux_jy=2.0, name="line"
    )
    transmitter = NarrowbandTransmitter(
        enu_from_horizontal(70.0, 1.0, 4000.0), 1.40512e9, 1.5e5, 400.0, name="tx"
    )
    burst = ImpulsiveBroadband(
        rate_hz=200.0,
        received_power_jy=800.0,
        position_enu_m=enu_from_horizontal(10.0, 3.0, 900.0),
    )
    return VoltageSimulator(
        array,
        phase_center,
        start_time,
        [source],
        rfi_sources=[transmitter, burst],
        spectral_lines=[line],
        n_chan=32,
        n_time_per_block=64,
        n_blocks=3,
        instrument=InstrumentModel.from_params(
            array.n_antennas, seed=11, gain_scatter_db=0.7, bandpass_ripple_db=0.4
        ),
        channelizer=PFBChannelizer(),
        quantization="int4",
        rng=np.random.default_rng(20260803),
        **kwargs,
    )


def transmitter_simulator(array, start_time, polarization=None, **kwargs):
    """A single narrowband transmitter, no sky, tunable noise."""
    options = {
        "n_chan": 16,
        "n_time_per_block": 1024,
        "n_blocks": 2,
        "noise_std": 0.0,
        "n_pol": 2,
    }
    options.update(kwargs)
    phase_center = zenith_phase_center(array, start_time, duration_s=1.0)
    transmitter = NarrowbandTransmitter(
        enu_from_horizontal(70.0, 1.0, 4000.0),
        1.40512e9,
        1.5e5,
        400.0,
        polarization=polarization,
        name="tx",
    )
    return VoltageSimulator(
        array,
        phase_center,
        start_time,
        [],
        rfi_sources=[transmitter],
        rng=np.random.default_rng(4242),
        **options,
    )


def point_source_simulator(array, start_time, **kwargs):
    """One noiseless point source off the phase center."""
    options = {"n_chan": 16, "n_time_per_block": 1024, "n_blocks": 2, "noise_std": 0.0}
    options.update(kwargs)
    phase_center = zenith_phase_center(array, start_time, duration_s=1.0)
    source = PointSource.from_lm(phase_center, (0.004, -0.002), flux_jy=5.0)
    return VoltageSimulator(
        array,
        phase_center,
        start_time,
        [source],
        rng=np.random.default_rng(31415),
        **options,
    )


def pol_power(data):
    """Mean power per receptor of ``(n_ant, n_pol, n_chan, n_time)`` data."""
    return (np.abs(data) ** 2).mean(axis=(0, 2, 3))


def pol_coherence(data):
    """Normalized complex coherence between the two receptors' voltages."""
    cross = np.mean(data[:, 0] * np.conj(data[:, 1]))
    power = np.sqrt(np.mean(np.abs(data[:, 0]) ** 2) * np.mean(np.abs(data[:, 1]) ** 2))
    return complex(cross / power)


# ----------------------------------------------------------------------
# Default off: the single-polarization data must not move
# ----------------------------------------------------------------------
#: sha256 of ``VoltageBlock.data`` for blocks 0-2 of `reference_simulator`,
#: recorded from the package as it stood before dual polarization existed.
#: A contract: ``n_pol=1`` is the default output of this simulator.
REFERENCE_DIGESTS = (
    "cddd5c122a3347813294dcf8ef5f80db930c159da92eb82cc71fed2a62cba56a",
    "107690a72961d937acdc62566490cff44f27e89ce9b3962f8a695f1fdf07b053",
    "8df866459e72b1be091750006797d9095ac0cc35aeadf61d85692197eaecf926",
)

#: sha256 of ``Visibilities.data`` for the same scene, and of block 0's
#: per-antenna ``clip_fraction`` -- the correlator and the quantizer's
#: ground truth are part of the same contract.
REFERENCE_VIS_DIGEST = "0b476b6dc5d78b43308d694621782a4181f62ad4de3a8d613be3030c51e529d1"
REFERENCE_CLIP_DIGEST = "0903b3e06165e187762f307471b04fb07e16345aa29434ac942e9a21265e2246"


def test_single_pol_output_is_bit_identical(default_array, start_time):
    """n_pol=1 reproduces the recorded bytes of the whole stack."""
    sim = reference_simulator(default_array, start_time)
    assert sim.n_pol == 1
    blocks = list(sim.blocks())
    for index, expected in enumerate(REFERENCE_DIGESTS):
        data = np.ascontiguousarray(blocks[index].data)
        assert hashlib.sha256(data.tobytes()).hexdigest() == expected, (
            f"block {index} changed; the single-polarization output is a contract"
        )
    vis = correlate(blocks)
    vis_bytes = np.ascontiguousarray(vis.data).tobytes()
    assert hashlib.sha256(vis_bytes).hexdigest() == REFERENCE_VIS_DIGEST
    clip_bytes = np.ascontiguousarray(blocks[0].clip_fraction).tobytes()
    assert hashlib.sha256(clip_bytes).hexdigest() == REFERENCE_CLIP_DIGEST


def test_single_pol_shapes_are_unchanged(default_array, start_time):
    """No polarization axis appears anywhere in a single-receptor run."""
    sim = reference_simulator(default_array, start_time)
    block = sim.block(0)
    n_ant, n_chan, n_time = default_array.n_antennas, 32, 64
    assert block.data.shape == (n_ant, n_chan, n_time)
    assert block.n_pol == 1
    assert block.pol_data.shape == (n_ant, 1, n_chan, n_time)
    assert block.gains.shape == (n_ant, n_chan)
    assert block.clip_fraction.shape == (n_ant,)
    assert block.rfi_polarization.shape == (2, 1)
    assert np.array_equal(block.rfi_polarization, np.ones((2, 1)))

    vis = correlate([block])
    assert vis.data.ndim == 3
    assert vis.n_pol == 1
    assert vis.pol_names == ()
    assert vis.stokes_i() is vis.data


def test_polarization_state_is_a_no_op_in_a_single_pol_run(default_array, start_time):
    """One receptor cannot be asymmetric: the state must not change the data."""
    unpolarized = transmitter_simulator(default_array, start_time, n_pol=1)
    polarized = transmitter_simulator(
        default_array,
        start_time,
        polarization={"type": "linear", "angle_deg": POLARIZATION_ANGLE_DEG},
        n_pol=1,
    )
    assert np.array_equal(unpolarized.block(0).data, polarized.block(0).data)


# ----------------------------------------------------------------------
# Shapes and labels of a dual-polarization run
# ----------------------------------------------------------------------
def test_dual_pol_shapes_and_labels(default_array, start_time):
    """The polarization axis sits after the antennas, and is labelled."""
    sim = reference_simulator(default_array, start_time, n_pol=2)
    block = sim.block(0)
    n_ant, n_chan, n_time = default_array.n_antennas, 32, 64
    assert block.data.shape == (n_ant, 2, n_chan, n_time)
    assert block.n_pol == 2
    assert block.pol_data is block.data
    assert block.gains.shape == (n_ant, 2, n_chan)
    assert block.clip_fraction.shape == (n_ant, 2)
    assert block.rfi_polarization.shape == (2, 2)
    # Occupancy is pol-independent by construction: no receptor axis here.
    assert block.rfi_mask.shape == (2, n_chan, n_time)
    assert block.celestial_mask.shape == (1, n_chan, n_time)

    vis = correlate(sim.blocks())
    assert vis.data.shape == (3, 55, 2, n_chan)
    assert vis.pol_names == PARALLEL_HAND_NAMES == ("XX", "YY")
    assert vis.pol_index("YY") == 1
    assert vis.stokes_i().shape == (3, 55, n_chan)
    # Ground truth rides along, and the occupancy fractions do not grow a
    # receptor axis.
    assert vis.rfi_polarization.shape == (2, 2)
    assert vis.rfi_fraction.shape == (3, 2, n_chan)


def test_where_a_dual_run_reuses_the_single_run_draws(default_array, start_time):
    """Documented exactly: which receptor-0 streams survive adding a receptor.

    Adding a receptor prepends or inserts an axis into every draw rather
    than opening a new generator, so the block generator is consumed in the
    same order with the same values -- only laid out differently. Where the
    polarization axis is the *outermost* one of a draw, receptor 0 is
    therefore bit-identical to the single-polarization run: that is the
    case for a sky source, whose spectrum is drawn ``(n_pol, n_chan,
    n_time)``. Where it sits inside an antenna axis, as for receiver noise
    drawn ``(n_ant, n_pol, n_chan, n_time)``, the layout interleaves and
    only the first antenna's receptor 0 lines up.

    Neither is a requirement -- what is required is that the
    single-polarization path itself never moves, which
    `test_single_pol_output_is_bit_identical` pins. This test exists so
    that the coincidence is a recorded property rather than a surprise.
    """
    single = point_source_simulator(default_array, start_time, n_pol=1).block(0)
    dual = point_source_simulator(default_array, start_time, n_pol=2).block(0)
    assert np.array_equal(dual.data[:, 0], single.data)
    assert not np.array_equal(dual.data[:, 1], single.data)

    phase_center = zenith_phase_center(default_array, start_time, duration_s=1.0)

    def noise_only(n_pol):
        return VoltageSimulator(
            default_array,
            phase_center,
            start_time,
            [],
            n_chan=8,
            n_time_per_block=64,
            n_blocks=1,
            noise_std=1.0,
            n_pol=n_pol,
            rng=np.random.default_rng(99),
        ).block(0)

    noisy_single, noisy_dual = noise_only(1), noise_only(2)
    assert np.array_equal(noisy_dual.data[0, 0], noisy_single.data[0])
    assert not np.array_equal(noisy_dual.data[1, 0], noisy_single.data[1])


def test_blocks_must_agree_on_the_receptor_count(default_array, start_time):
    """Correlating a mixed stream is a configuration error, loudly."""
    single = point_source_simulator(default_array, start_time, n_pol=1).block(0)
    dual = point_source_simulator(default_array, start_time, n_pol=2).block(0)
    with pytest.raises(ValueError, match="same number of polarizations"):
        correlate([single, dual])


# ----------------------------------------------------------------------
# Unpolarized: sky, spectral lines and receiver noise
# ----------------------------------------------------------------------
def test_receiver_noise_is_unpolarized(default_array, start_time):
    """Equal power in both receptors, and no coherence between them."""
    phase_center = zenith_phase_center(default_array, start_time, duration_s=1.0)
    sim = VoltageSimulator(
        default_array,
        phase_center,
        start_time,
        [],
        n_chan=8,
        n_time_per_block=2048,
        n_blocks=2,
        noise_std=2.0,
        n_pol=2,
        rng=np.random.default_rng(5),
    )
    data = sim.block(0).data
    power = pol_power(data)
    assert power == pytest.approx([4.0, 4.0], rel=0.02)
    assert abs(pol_coherence(data)) < 0.02


def test_spectral_lines_are_unpolarized(default_array, start_time):
    """A celestial line splits like the noise does: equal, and incoherent."""
    phase_center = zenith_phase_center(default_array, start_time, duration_s=1.0)
    line = SpectralLineForeground(
        center_freq_hz=1.405e9, fwhm_hz=2.0e5, line_flux_jy=50.0, name="line"
    )
    sim = VoltageSimulator(
        default_array,
        phase_center,
        start_time,
        [],
        spectral_lines=[line],
        n_chan=8,
        n_time_per_block=2048,
        n_blocks=1,
        noise_std=0.0,
        n_pol=2,
        rng=np.random.default_rng(6),
    )
    data = sim.block(0).data
    power = pol_power(data)
    assert power[0] == pytest.approx(power[1], rel=0.03)
    assert abs(pol_coherence(data)) < 0.02


def test_stokes_i_preserves_a_point_source_flux_and_position(default_array, start_time):
    """A dual-polarization image of a scene matches the single-pol image.

    The convention under test is ``I = (XX + YY) / 2`` with each receptor
    carrying the source's full Stokes-I flux, which is what makes the two
    runs comparable at all.
    """
    l_grid = np.linspace(-0.01, 0.01, 41)
    images = {}
    for n_pol in (1, 2):
        sim = point_source_simulator(default_array, start_time, n_pol=n_pol)
        vis = correlate(sim.blocks())
        image, l_axis, m_axis = dirty_image(vis, l_grid, l_grid, warn_on_w_term=False)
        images[n_pol] = image

    peak_1 = np.unravel_index(np.argmax(images[1]), images[1].shape)
    peak_2 = np.unravel_index(np.argmax(images[2]), images[2].shape)
    assert peak_1 == peak_2
    assert l_axis[peak_1[1]] == pytest.approx(0.004, abs=6e-4)
    assert m_axis[peak_1[0]] == pytest.approx(-0.002, abs=6e-4)
    assert images[2][peak_2] == pytest.approx(5.0, rel=0.05)
    assert images[2][peak_2] == pytest.approx(images[1][peak_1], rel=0.05)


def test_imaging_a_single_receptor_of_an_unpolarized_source(default_array, start_time):
    """Each receptor alone carries the full Stokes-I flux of the sky."""
    sim = point_source_simulator(default_array, start_time, n_pol=2)
    vis = correlate(sim.blocks())
    l_grid = np.linspace(-0.01, 0.01, 41)
    xx, _, _ = dirty_image(vis, l_grid, l_grid, pol="XX", warn_on_w_term=False)
    yy, _, _ = dirty_image(vis, l_grid, l_grid, pol=1, warn_on_w_term=False)
    assert xx.max() == pytest.approx(5.0, rel=0.06)
    assert yy.max() == pytest.approx(5.0, rel=0.06)
    with pytest.raises(KeyError, match="XY"):
        dirty_image(vis, l_grid, l_grid, pol="XY", warn_on_w_term=False)


# ----------------------------------------------------------------------
# Polarized interference
# ----------------------------------------------------------------------
@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 45.0, 90.0])
def test_linear_polarization_splits_power_as_cos2_sin2(default_array, start_time, angle_deg):
    """Deterministic amplitude ratio, and a fixed (zero) relative phase."""
    sim = transmitter_simulator(
        default_array, start_time, polarization={"type": "linear", "angle_deg": angle_deg}
    )
    data = sim.block(0).data
    power = pol_power(data)

    angle_rad = np.deg2rad(angle_deg)
    expected = 2.0 * np.array([np.cos(angle_rad) ** 2, np.sin(angle_rad) ** 2])
    # Both receptors carry the *same* realization, so the split is exact
    # rather than statistical.
    assert power / power.sum() == pytest.approx(expected / expected.sum(), abs=1e-6)

    if 0.0 < angle_deg < 90.0:
        coherence = pol_coherence(data)
        assert abs(coherence) == pytest.approx(1.0, abs=1e-5)
        assert np.angle(coherence) == pytest.approx(0.0, abs=1e-5)
    else:
        # All the power in one receptor: the other is silent down to the
        # rounding of cos(90 deg), i.e. 30-odd orders of magnitude down.
        assert min(power) / max(power) < 1e-20


def test_stokes_i_of_a_polarized_transmitter_equals_its_received_power(default_array, start_time):
    """The polarization state redistributes power; it does not create it."""
    unpolarized = transmitter_simulator(default_array, start_time)
    polarized = transmitter_simulator(
        default_array,
        start_time,
        polarization={"type": "linear", "angle_deg": POLARIZATION_ANGLE_DEG},
    )
    assert pol_power(polarized.block(0).data).sum() == pytest.approx(
        pol_power(unpolarized.block(0).data).sum(), rel=0.05
    )


def test_full_jones_carries_a_fixed_relative_phase(default_array, start_time):
    """A circular state splits evenly and holds a 90 deg receptor phase."""
    sim = transmitter_simulator(
        default_array, start_time, polarization={"type": "full", "jones": [1.0, 1.0j]}
    )
    data = sim.block(0).data
    power = pol_power(data)
    assert power[0] == pytest.approx(power[1], rel=1e-5)
    coherence = pol_coherence(data)
    assert abs(coherence) == pytest.approx(1.0, abs=1e-5)
    assert np.angle(coherence) == pytest.approx(-np.pi / 2, abs=1e-4)


def test_unpolarized_interference_is_equal_and_incoherent(default_array, start_time):
    """The default state replicates the historical behavior, per receptor."""
    sim = transmitter_simulator(default_array, start_time, n_time_per_block=4096)
    data = sim.block(0).data
    power = pol_power(data)
    assert power[0] == pytest.approx(power[1], rel=0.05)
    assert abs(pol_coherence(data)) < 0.05


def test_partial_polarization_interpolates_between_the_two(default_array, start_time):
    """Fraction p puts 2p|c|**2 + (1 - p) of the power in each receptor."""
    fraction = 0.5
    sim = transmitter_simulator(
        default_array,
        start_time,
        polarization={"type": "linear", "angle_deg": 0.0, "fraction": fraction},
        n_time_per_block=4096,
    )
    power = pol_power(sim.block(0).data)
    ratio = power / power.sum()
    expected = np.array([2.0 * fraction + (1.0 - fraction), 1.0 - fraction])
    assert ratio == pytest.approx(expected / expected.sum(), abs=0.03)


def test_ground_truth_amplitudes_are_recorded_per_source(default_array, start_time):
    """The resolved per-receptor amplitudes ride on the block and the vis."""
    sim = transmitter_simulator(
        default_array,
        start_time,
        polarization={"type": "linear", "angle_deg": POLARIZATION_ANGLE_DEG},
        n_blocks=1,
    )
    block = sim.block(0)
    angle_rad = np.deg2rad(POLARIZATION_ANGLE_DEG)
    expected = np.sqrt(2.0) * np.array([np.cos(angle_rad), np.sin(angle_rad)])
    assert block.rfi_polarization[0] == pytest.approx(expected)

    # ...and the truth predicts the measured split.
    power = pol_power(block.data)
    weights = np.abs(block.rfi_polarization[0]) ** 2
    assert power / power.sum() == pytest.approx(weights / weights.sum(), abs=1e-6)

    vis = correlate(sim.blocks())
    assert vis.rfi_polarization == pytest.approx(block.rfi_polarization)


def test_polarization_ratio_separates_occupied_from_clean_channels(default_array, start_time):
    """The motivating case: a pol-ratio discriminant finds the transmitter.

    Polarized interference on top of unpolarized noise makes the ratio of
    the two receptors' powers depart from one *only* in the contaminated
    channels. This is the discriminant a polarization-aware flagger uses,
    and it must work on this simulator's output without any other cue.
    """
    sim = transmitter_simulator(
        default_array,
        start_time,
        polarization={"type": "linear", "angle_deg": POLARIZATION_ANGLE_DEG},
        noise_std=1.0,
        n_chan=32,
        n_time_per_block=2048,
        n_blocks=1,
    )
    block = sim.block(0)
    power = (np.abs(block.data) ** 2).mean(axis=(0, 3))  # (n_pol, n_chan)
    ratio = power[0] / power[1]

    occupied = block.rfi_mask[0].any(axis=1)
    assert occupied.any() and not occupied.all()
    assert ratio[occupied].min() > 3.0 * ratio[~occupied].max()
    assert ratio[~occupied] == pytest.approx(np.ones(int((~occupied).sum())), abs=0.1)


# ----------------------------------------------------------------------
# The rest of the stack: gains, filterbank, quantizer, packed I/O
# ----------------------------------------------------------------------
def test_one_instrument_model_is_broadcast_to_both_receptors(default_array, start_time):
    """A single model means the two feeds share a receiver chain."""
    model = InstrumentModel.from_params(default_array.n_antennas, seed=3, gain_scatter_db=2.0)
    sim = transmitter_simulator(default_array, start_time, instrument=model, n_blocks=1)
    gains = sim.block(0).gains
    assert gains.shape == (default_array.n_antennas, 2, 16)
    assert np.array_equal(gains[:, 0], gains[:, 1])


def test_per_receptor_instrument_models(default_array, start_time):
    """Two models give the two receptors genuinely different bandpasses."""
    n_ant = default_array.n_antennas
    models = [
        InstrumentModel.from_params(n_ant, seed=3, gain_scatter_db=3.0),
        InstrumentModel.from_params(n_ant, seed=4, gain_scatter_db=3.0),
    ]
    sim = transmitter_simulator(default_array, start_time, instrument=models, n_blocks=1)
    block = sim.block(0)
    gains = block.gains
    assert gains.shape == (n_ant, 2, 16)
    assert not np.allclose(gains[:, 0], gains[:, 1])
    # The recorded gains are truthful: they predict the per-antenna power
    # ratio between the receptors.
    power = (np.abs(block.data) ** 2).mean(axis=(2, 3))  # (n_ant, n_pol)
    predicted = (np.abs(gains) ** 2).mean(axis=2)
    assert power[:, 0] / power[:, 1] == pytest.approx(predicted[:, 0] / predicted[:, 1], rel=0.05)


def test_instrument_sequence_length_must_match_n_pol(default_array, start_time):
    """A mismatched receptor count is a loud error, not a broadcast."""
    model = InstrumentModel.from_params(default_array.n_antennas, seed=3)
    with pytest.raises(ValueError, match="one model per polarization"):
        transmitter_simulator(default_array, start_time, instrument=[model], n_blocks=1)
    with pytest.raises(ValueError, match="InstrumentModel"):
        transmitter_simulator(default_array, start_time, instrument="not-a-model", n_blocks=1)


def test_channelizer_colors_each_receptor_independently(default_array, start_time):
    """Both receptors pick up the filterbank's temporal memory, separately."""
    pfb = PFBChannelizer()
    phase_center = zenith_phase_center(default_array, start_time, duration_s=1.0)
    sim = VoltageSimulator(
        default_array,
        phase_center,
        start_time,
        [],
        n_chan=32,
        n_time_per_block=2000,
        n_blocks=2,
        noise_std=1.0,
        n_pol=2,
        channelizer=pfb,
        rng=np.random.default_rng(7),
    )
    data = np.concatenate([block.data for block in sim.blocks()], axis=3)
    predicted = pfb.temporal_autocorrelation(32)[1]
    for i_pol in range(2):
        stream = data[:, i_pol]
        measured = np.mean(stream[:, :, 1:] * np.conj(stream[:, :, :-1])) / np.mean(
            np.abs(stream) ** 2
        )
        assert measured.real == pytest.approx(predicted.real, abs=0.02)
    # Filtering does not couple the receptors: they stay incoherent.
    assert abs(pol_coherence(data)) < 0.02


def test_quantization_rails_the_receptor_the_transmitter_shouts_into(default_array, start_time):
    """Per-receptor clip fractions record a polarized overload honestly."""
    sim = transmitter_simulator(
        default_array,
        start_time,
        polarization={"type": "linear", "angle_deg": 0.0},
        noise_std=1.0,
        n_time_per_block=256,
        n_blocks=1,
        quantization="int4",
        quant_scale=1.0,
    )
    block = sim.block(0)
    assert block.clip_fraction.shape == (default_array.n_antennas, 2)
    assert block.clip_fraction[:, 0].min() > 0.0
    assert block.clip_fraction[:, 1].max() == 0.0
    assert block.quant_scale == 1.0


def test_packed_round_trip_of_a_genuine_dual_pol_block(default_array, start_time):
    """`pol_mode="block"` writes two different receptors, not one twice."""
    sim = transmitter_simulator(
        default_array,
        start_time,
        polarization={"type": "linear", "angle_deg": POLARIZATION_ANGLE_DEG},
        noise_std=1.0,
        n_chan=8,
        n_time_per_block=64,
        n_blocks=1,
        quantization="int4",
        quant_scale=0.5,
    )
    block = sim.block(0)
    layout = PackedVoltageLayout(
        n_packets=8,
        n_antennas=default_array.n_antennas,
        n_channels=8,
        n_times_per_packet=8,
        n_pols=2,
    )
    raw = pack_from_voltage_block(block.data, layout, 0.5, pol_mode="block")
    assert len(raw) == layout.bytes_per_block
    recovered = unpack_block(raw, layout, 0.5)
    # On-disk order is (n_ant, n_chan, n_time, n_pol); the block's is
    # (n_ant, n_pol, n_chan, n_time).
    assert np.array_equal(recovered, np.transpose(block.data, (0, 2, 3, 1)))
    assert not np.array_equal(recovered[..., 0], recovered[..., 1])


def test_packed_block_mode_accepts_a_single_pol_block(default_array, start_time):
    """One writer path for both shapes: 3-D data still duplicates."""
    sim = transmitter_simulator(
        default_array,
        start_time,
        n_pol=1,
        n_chan=8,
        n_time_per_block=64,
        n_blocks=1,
        quantization="int4",
        quant_scale=0.5,
    )
    block = sim.block(0)
    layout = PackedVoltageLayout(
        n_packets=8,
        n_antennas=default_array.n_antennas,
        n_channels=8,
        n_times_per_packet=8,
        n_pols=2,
    )
    raw = pack_from_voltage_block(block.data, layout, 0.5, pol_mode="block")
    recovered = unpack_block(raw, layout, 0.5)
    assert np.array_equal(recovered[..., 0], recovered[..., 1])
    assert np.array_equal(recovered[..., 0], block.data)


def test_packed_block_mode_rejects_a_stray_shape(default_array):
    """A 2-D array is not a block, and says so."""
    layout = PackedVoltageLayout(
        n_packets=1, n_antennas=2, n_channels=2, n_times_per_packet=2, n_pols=2
    )
    with pytest.raises(ValueError, match="pol_mode='block'"):
        pack_from_voltage_block(np.zeros((2, 2), dtype=np.complex64), layout, pol_mode="block")


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"type": "elliptical"}, "polarization type"),
        ({"type": "linear", "angle_deg": np.nan}, "angle_deg must be finite"),
        ({"type": "linear", "angle_deg": np.inf}, "angle_deg must be finite"),
        ({"type": "linear", "angle_deg": 0.0, "tilt": 1.0}, "unexpected keys"),
        ({"type": "linear", "angle_deg": 0.0, "fraction": 1.5}, "fraction"),
        ({"type": "linear", "angle_deg": 0.0, "fraction": np.nan}, "fraction"),
        ({"type": "full", "jones": [1.0, 0.0, 0.0]}, "length-2"),
        ({"type": "full", "jones": [1.0]}, "length-2"),
        ({"type": "full", "jones": [1.0, np.nan]}, "non-finite"),
        ({"type": "full", "jones": [1.0, np.inf * 1j]}, "non-finite"),
        ({"type": "full", "jones": [0.0, 0.0]}, "all zero"),
        ({"type": "unpolarized", "angle_deg": 3.0}, "unexpected keys"),
    ],
)
def test_bad_polarization_specifications_raise(spec, match):
    """Every malformed state is rejected at construction, loudly."""
    with pytest.raises(ValueError, match=match):
        NarrowbandTransmitter(
            enu_from_horizontal(70.0, 1.0, 4000.0), 1.405e9, 1.0e5, 1.0, polarization=spec
        )


def test_polarization_must_be_a_mapping():
    """A bare string is a common slip and gets a pointed message."""
    with pytest.raises(ValueError, match="must be None or a mapping"):
        NarrowbandTransmitter(
            enu_from_horizontal(70.0, 1.0, 4000.0), 1.405e9, 1.0e5, 1.0, polarization="linear"
        )


def test_explicit_unpolarized_normalizes_to_the_default():
    """One internal representation for "no polarization state"."""
    source = NarrowbandTransmitter(
        enu_from_horizontal(70.0, 1.0, 4000.0),
        1.405e9,
        1.0e5,
        1.0,
        polarization={"type": "unpolarized"},
    )
    assert source.polarization is None
    assert np.array_equal(source.pol_mixing(2), np.eye(2))
    assert source.polarization_amplitudes(2) == pytest.approx(np.ones(2))
    assert source.polarization_amplitudes(1) == pytest.approx(np.ones(1))


def test_resolve_polarization_rejects_unsupported_receptor_counts():
    """Only 1 or 2 receptors are modelled."""
    with pytest.raises(ValueError, match="n_pol must be 1 or 2"):
        resolve_polarization(None, 3)
    with pytest.raises(ValueError, match="n_pol must be 1 or 2"):
        resolve_polarization(None, 0)


def test_simulator_rejects_unsupported_receptor_counts(default_array, start_time):
    """...and so does the simulator that would have to synthesize them."""
    with pytest.raises(ValueError, match="n_pol must be 1 or 2"):
        point_source_simulator(default_array, start_time, n_pol=3)


def test_a_source_that_ignores_n_pol_is_caught(default_array, start_time):
    """A source emitting the old shape into a dual-pol run fails loudly."""

    class SinglePolOnly(RFISource):
        def contribution(self, ctx):
            return (
                np.zeros((ctx.n_antennas, ctx.n_chan, ctx.n_time), dtype=np.complex64),
                np.zeros((ctx.n_chan, ctx.n_time), dtype=bool),
            )

    phase_center = zenith_phase_center(default_array, start_time, duration_s=1.0)
    sim = VoltageSimulator(
        default_array,
        phase_center,
        start_time,
        [],
        rfi_sources=[SinglePolOnly("legacy")],
        n_chan=4,
        n_time_per_block=8,
        n_blocks=1,
        n_pol=2,
        rng=np.random.default_rng(0),
    )
    with pytest.raises(ValueError, match="returned voltages of shape"):
        sim.block(0)


def test_voltage_block_validates_its_shapes(default_array, start_time):
    """The container rejects data and ground truth it cannot describe."""
    sim = transmitter_simulator(default_array, start_time, n_blocks=1)
    block = sim.block(0)
    fields = {
        "time": block.time,
        "center_time": block.center_time,
        "freq_hz": block.freq_hz,
        "sample_period_s": block.sample_period_s,
        "phase_center_delays_s": block.phase_center_delays_s,
        "antenna_positions_enu_m": block.antenna_positions_enu_m,
        "e_l_enu": block.e_l_enu,
        "e_m_enu": block.e_m_enu,
        "s0_enu": block.s0_enu,
    }
    with pytest.raises(ValueError, match="data must have shape"):
        VoltageBlock(data=block.data[:, 0, 0], **fields)
    with pytest.raises(ValueError, match="rfi_polarization must have shape"):
        VoltageBlock(
            data=block.data,
            rfi_source_names=("tx",),
            rfi_mask=np.zeros((1, block.n_chan, block.n_time), dtype=bool),
            rfi_polarization=np.ones((1, 3), dtype=np.complex128),
            **fields,
        )


def test_dual_pol_visibilities_must_be_labelled(default_array, start_time):
    """A polarization axis without names is not a schema anyone can read."""
    sim = transmitter_simulator(default_array, start_time, n_blocks=1)
    vis = correlate(sim.blocks())
    fields = {
        "ant_1": vis.ant_1,
        "ant_2": vis.ant_2,
        "freq_hz": vis.freq_hz,
        "time_mjd": vis.time_mjd,
        "integration_time_s": vis.integration_time_s,
        "n_samples": vis.n_samples,
        "baseline_vectors_enu_m": vis.baseline_vectors_enu_m,
        "e_l_enu": vis.e_l_enu,
        "e_m_enu": vis.e_m_enu,
        "s0_enu": vis.s0_enu,
    }
    with pytest.raises(ValueError, match="must be labelled"):
        Visibilities(data=vis.data, **fields)
    with pytest.raises(ValueError, match="pol_names has"):
        Visibilities(data=vis.data, pol_names=("XX", "YY", "XY"), **fields)
    with pytest.raises(ValueError, match="data must have shape"):
        Visibilities(data=vis.data[:, :, :, 0, np.newaxis, np.newaxis], **fields)

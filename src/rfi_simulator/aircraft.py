r"""Airborne transponders as interference: short, strong, moving pulses.

Aircraft surveillance transponders broadcast at 1090 MHz in bursts about
a microsecond long, a few times a second, from a platform a few hundred
kilometres away moving at a few hundred metres per second. Three of those
numbers matter for a voltage-level simulator:

* **The pulses are shorter than one spectrum.** At this package's default
  32.768 us post-channelization sample period, a microsecond burst lands
  inside a single time sample, spread across the channels its ~2 MHz
  spectrum covers. So a transponder is modelled as one-sample bursts, not
  as a continuous emitter with a duty cycle -- it is the extreme end of
  the impulsive family.
* **They are strong.** An aircraft in line of sight with a kilowatt-class
  transponder can dominate the total power of the samples it lands in,
  which is exactly the regime where a naive excision algorithm starts
  eating real signal along with the interference.
* **The platform moves, but not fast enough to Doppler.** At 250 m/s the
  fractional shift is :math:`8 \times 10^{-7}`, i.e. under 900 Hz at
  1090 MHz -- well within one 30.5 kHz channel. Motion is therefore
  carried entirely by the per-block geometry (exact path delays, which
  visibly change the fringe across a pass) and no frequency shift is
  applied. Contrast `rfi_simulator.satellites`, where the Doppler moves
  the signal by several channels and cannot be ignored.

Band placement
--------------
1090 MHz is far outside this package's default 11.7 MHz band around
1.405 GHz, so a default-band simulation with an `ADSBTransponder` raises
`ValueError` rather than silently contributing nothing. Re-center the
simulated band on the transponder, or set `carrier_freq_hz` to an in-band
value to study a hypothetical in-band airborne emitter -- the pulse
statistics and the fast-moving geometry are what make this source
distinctive, not the particular carrier.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u

from rfi_simulator.array_config import _to_value
from rfi_simulator.rfi import (
    BlockContext,
    RFISource,
    band_overlaps,
    channels_within,
    circular_normal,
    elevation_deg,
    occupancy_mask,
    out_of_band_message,
)

__all__ = ["ADSB_FREQ_HZ", "ADSBTransponder"]

ADSB_FREQ_HZ = 1.09e9
"""float: Standard airborne surveillance transponder frequency, Hz."""


class ADSBTransponder(RFISource):
    """An aircraft transponder on a straight-line course.

    The aircraft flies a constant-velocity track through the array's
    local ENU frame and emits one-sample bursts at a Poisson message
    rate. Position -- and therefore the per-antenna path delays -- is
    re-evaluated every block, so a pass across the sky produces visibly
    evolving fringes.

    Parameters
    ----------
    position_enu_m : array_like
        Position at the **start of the observation**, shape ``(3,)``,
        local ENU meters. The third component is the altitude above the
        array origin, e.g. ``11000.0`` for cruise altitude.
    velocity_enu_m_s : array_like, optional
        Constant velocity, shape ``(3,)``, ENU meters per second. Default
        ``(250.0, 0.0, 0.0)``, i.e. due east at a typical cruise speed.
    carrier_freq_hz : float or astropy.units.Quantity, optional
        Center frequency of the burst spectrum, Hz. Default
        `ADSB_FREQ_HZ` (1090 MHz), which is outside the package's default
        band -- see the module docstring.
    bandwidth_hz : float or astropy.units.Quantity, optional
        Full width of the burst spectrum, Hz. Default 2.0e6, roughly the
        inverse of the sub-microsecond pulse rise time.
    received_power_jy : float or astropy.units.Quantity, optional
        Power received at the array origin during a burst, summed over
        the occupied channels, in janskys. Default 5.0e4 -- deliberately
        loud, because that is the interesting case.
    message_rate_hz : float or astropy.units.Quantity, optional
        Mean burst rate, messages per second. Default 6.2, the nominal
        extended-squitter rate. Raise it to get useful statistics out of
        a short snippet.
    pulse_width_samples : int, optional
        Burst length in post-channelization samples. Default 1: at the
        package defaults a real burst is ~30 times shorter than one
        sample, so one sample is already an over-estimate.
    coupling : None, array_like or mapping, optional
        Per-antenna linear amplitude coupling; see
        `rfi_simulator.rfi.resolve_coupling`. Default ``None`` (uniform).
    min_elevation_deg : float or astropy.units.Quantity, optional
        Elevation below which the aircraft is treated as over the
        horizon: blocks whose mid-point elevation is under this
        contribute exactly zero voltages and an all-False mask. Default
        0.0, the geometric horizon -- which matters here, because a
        long track at cruise altitude flies out of view during a single
        simulated pass. Set to ``-90`` to disable the cut.
    name : str, optional
        Label for the source. Default ``"transponder"``.

    Raises
    ------
    ValueError
        If any rate, power, bandwidth or width is invalid, if the
        position/velocity vectors are not 3-vectors, or if
        `min_elevation_deg` is outside ``[-90, 90]``.

    Notes
    -----
    The horizon test is a **sharp geometric cut** in the array's local
    tangent plane: the aircraft is either fully visible or fully absent,
    switching between one block and the next. No atmospheric refraction,
    knife-edge diffraction, terrain or antenna response is modelled, and
    Earth curvature is not applied to the tangent plane -- so for an
    aircraft hundreds of kilometres away the cut is approximate. Real
    horizon crossings fade; this one does not.

    Time is measured from the start of the observation, taken as
    ``block index * block duration`` -- blocks tile the observation
    contiguously, so the aircraft's position depends only on the block
    index and the source stays a pure function of its inputs. There is no
    hidden state carried between blocks and none is needed.

    The emission within a burst is drawn as band-limited noise rather
    than a deterministic pulse shape. What survives channelization at
    30.5 kHz resolution is the burst's power spectral density, not its
    microsecond-scale pulse-position modulation, so modelling the
    modulation would add cost without changing anything measurable in
    this data product.

    Examples
    --------
    >>> aircraft = ADSBTransponder(
    ...     position_enu_m=(-40000.0, 15000.0, 11000.0),
    ...     velocity_enu_m_s=(250.0, 0.0, 0.0),
    ...     carrier_freq_hz=1.405e9,
    ...     message_rate_hz=50.0,
    ... )
    >>> aircraft.name
    'transponder'
    """

    def __init__(
        self,
        position_enu_m,
        velocity_enu_m_s=(250.0, 0.0, 0.0),
        *,
        carrier_freq_hz=ADSB_FREQ_HZ,
        bandwidth_hz=2.0e6,
        received_power_jy=5.0e4,
        message_rate_hz=6.2,
        pulse_width_samples: int = 1,
        min_elevation_deg=0.0,
        coupling=None,
        name: str = "transponder",
    ) -> None:
        super().__init__(name, coupling=coupling)
        self.position_enu_m = np.asarray(_to_value(position_enu_m, u.m), dtype=np.float64).reshape(
            3
        )
        self.velocity_enu_m_s = np.asarray(
            _to_value(velocity_enu_m_s, u.m / u.s), dtype=np.float64
        ).reshape(3)
        self.carrier_freq_hz = float(_to_value(carrier_freq_hz, u.Hz))
        self.bandwidth_hz = float(_to_value(bandwidth_hz, u.Hz))
        self.received_power_jy = float(_to_value(received_power_jy, u.Jy))
        self.message_rate_hz = float(_to_value(message_rate_hz, u.Hz))
        self.pulse_width_samples = int(pulse_width_samples)
        self.min_elevation_deg = float(_to_value(min_elevation_deg, u.deg))

        if not -90.0 <= self.min_elevation_deg <= 90.0:
            raise ValueError(
                f"min_elevation_deg must be in [-90, 90], got {self.min_elevation_deg}"
            )
        if self.bandwidth_hz < 0.0:
            raise ValueError(f"bandwidth_hz must be >= 0, got {self.bandwidth_hz}")
        if self.received_power_jy < 0.0:
            raise ValueError(f"received_power_jy must be >= 0, got {self.received_power_jy}")
        if self.message_rate_hz < 0.0:
            raise ValueError(f"message_rate_hz must be >= 0, got {self.message_rate_hz}")
        if self.pulse_width_samples < 1:
            raise ValueError(f"pulse_width_samples must be >= 1, got {self.pulse_width_samples}")

    def position_at(self, elapsed_s: float) -> np.ndarray:
        """Aircraft position a given time after the start of the observation.

        Parameters
        ----------
        elapsed_s : float
            Seconds since the start of the observation.

        Returns
        -------
        numpy.ndarray
            Shape ``(3,)`` float64 ENU position in meters.
        """
        return self.position_enu_m + self.velocity_enu_m_s * float(elapsed_s)

    def block_position_enu_m(self, ctx: BlockContext) -> np.ndarray:
        """Aircraft position at the mid-point of a block.

        Parameters
        ----------
        ctx : BlockContext
            Block context; `BlockContext.index` and
            `BlockContext.duration_s` set the elapsed time.

        Returns
        -------
        numpy.ndarray
            Shape ``(3,)`` float64 ENU position in meters.
        """
        return self.position_at((ctx.index + 0.5) * ctx.duration_s)

    def draw_burst_samples(self, ctx: BlockContext) -> np.ndarray:
        """Draw the first sample index of each burst in a block.

        Parameters
        ----------
        ctx : BlockContext
            Block context; its `rng` supplies the Poisson count and the
            burst placements.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_bursts,)`` int array of first-sample indices,
            uniform over the block.
        """
        expected = self.message_rate_hz * ctx.duration_s
        n_bursts = int(ctx.rng.poisson(expected))
        return ctx.rng.integers(0, ctx.n_time, size=n_bursts)

    def contribution(self, ctx: BlockContext) -> tuple[np.ndarray, np.ndarray]:
        """Voltages and occupancy mask for one block.

        Parameters
        ----------
        ctx : BlockContext
            Block geometry, timing and generator.

        Returns
        -------
        voltages : numpy.ndarray
            Complex64 ``(n_antennas, n_chan, n_time)`` contribution in
            root-Jy.
        mask : numpy.ndarray
            Boolean ``(n_chan, n_time)`` occupancy labels: the occupied
            channels, on the samples a burst landed in. All False for a
            block in which the aircraft is below `min_elevation_deg`.

        Raises
        ------
        ValueError
            If the burst spectrum lies wholly outside the simulated band.
            At the package defaults that is the *expected* outcome for a
            1090 MHz transponder, and it is an error rather than silence
            so the band mismatch is impossible to overlook. The band check
            runs before the horizon test, so a misconfigured band is
            reported even on blocks where the aircraft is out of view.
        """
        in_band = band_overlaps(
            self.carrier_freq_hz, self.bandwidth_hz, ctx.freq_hz, ctx.chan_width_hz
        )
        occupied = channels_within(ctx.freq_hz, self.carrier_freq_hz, self.bandwidth_hz)
        n_occupied = int(occupied.sum())
        if not in_band or n_occupied == 0:
            raise ValueError(
                out_of_band_message(self.name, self.carrier_freq_hz, self.bandwidth_hz, ctx.freq_hz)
            )

        position_enu_m = self.block_position_enu_m(ctx)
        if elevation_deg(position_enu_m) < self.min_elevation_deg:
            # Over the horizon: silent, and labelled silent. Note this
            # returns before drawing any bursts, so an out-of-view block
            # costs nothing and consumes no randomness it does not need.
            return (
                np.zeros((ctx.n_antennas, ctx.n_chan, ctx.n_time), dtype=np.complex64),
                np.zeros((ctx.n_chan, ctx.n_time), dtype=bool),
            )

        starts = self.draw_burst_samples(ctx)
        active = np.zeros(ctx.n_time, dtype=bool)
        for start in starts:
            stop = min(int(start) + self.pulse_width_samples, ctx.n_time)
            active[int(start) : stop] = True

        envelope = np.zeros((ctx.n_chan, ctx.n_time), dtype=np.float64)
        envelope[np.ix_(occupied, active)] = self.received_power_jy / n_occupied
        mask = occupancy_mask(envelope)

        voltages = np.zeros((ctx.n_antennas, ctx.n_chan, ctx.n_time), dtype=np.complex64)
        active_samples = np.flatnonzero(active)
        if active_samples.size == 0:
            return voltages, mask

        waveform = circular_normal(ctx.rng, (n_occupied, active_samples.size))
        waveform *= np.float32(np.sqrt(self.received_power_jy / n_occupied))

        phasors = self.coupled_phasors(
            position_enu_m, ctx.antenna_positions_enu_m, ctx.freq_hz[occupied]
        )
        voltages[np.ix_(np.arange(ctx.n_antennas), occupied, active_samples)] = (
            phasors[:, :, np.newaxis] * waveform[np.newaxis, :, :]
        )
        return voltages, mask

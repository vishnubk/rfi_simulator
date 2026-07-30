"""Antenna array configuration: geometry, loading, and validation.

Conventions (see ``docs/design_stage2.md``): antenna positions are stored
internally as plain ``float64`` numpy arrays in local ENU (East-North-Up)
meters relative to the array origin. Astropy `~astropy.units.Quantity`
inputs are accepted at the public API boundary (the ``ArrayConfig``
constructor) and converted to meters immediately; nothing downstream of
``__init__`` deals with units.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from astropy import units as u


@dataclass
class ArrayConfig:
    """Geometry and location of an interferometric antenna array.

    Parameters
    ----------
    antenna_positions_enu_m : numpy.ndarray or astropy.units.Quantity
        Antenna positions in local East-North-Up (ENU) coordinates,
        shape ``(n_antennas, 3)``, relative to the array origin given by
        `latitude_deg`, `longitude_deg`, `height_m`. If a `Quantity` is
        given it must carry a unit convertible to meters; it is converted
        to plain meters on construction. Columns are ``(east, north, up)``
        in meters.
    latitude_deg : float or astropy.units.Quantity
        Geodetic latitude of the array origin, in degrees (or an angular
        `Quantity` convertible to degrees).
    longitude_deg : float or astropy.units.Quantity
        Geodetic longitude of the array origin, in degrees (or an angular
        `Quantity` convertible to degrees). East-positive.
    height_m : float or astropy.units.Quantity
        Height of the array origin above the WGS84 ellipsoid, in meters
        (or a length `Quantity` convertible to meters).
    name : str, optional
        Human-readable array name, e.g. ``"array_default"``. Defaults to
        ``""``.

    Attributes
    ----------
    antenna_positions_enu_m : numpy.ndarray
        Validated ``(n_antennas, 3)`` float64 array of ENU positions in
        meters.
    latitude_deg : float
        Array origin latitude in degrees.
    longitude_deg : float
        Array origin longitude in degrees.
    height_m : float
        Array origin height in meters.
    name : str
        Array name.

    Raises
    ------
    ValueError
        If fewer than 2 antennas are given, the position array is not
        shape ``(n_antennas, 3)``, or any position value is non-finite
        (NaN or inf).

    Notes
    -----
    Duplicate antenna positions (within floating-point tolerance) are
    permitted -- they are useful for zero-baseline sanity tests (see
    acceptance criterion 5 in ``docs/design_stage2.md``) -- but trigger a
    `UserWarning` since they are usually unintentional in a real array.
    """

    antenna_positions_enu_m: np.ndarray
    latitude_deg: float
    longitude_deg: float
    height_m: float
    name: str = ""

    def __post_init__(self) -> None:
        self.antenna_positions_enu_m = _to_value(self.antenna_positions_enu_m, u.m)
        self.latitude_deg = _to_value(self.latitude_deg, u.deg)
        self.longitude_deg = _to_value(self.longitude_deg, u.deg)
        self.height_m = _to_value(self.height_m, u.m)

        positions = np.asarray(self.antenna_positions_enu_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                "antenna_positions_enu_m must have shape (n_antennas, 3), "
                f"got shape {positions.shape}"
            )
        if positions.shape[0] < 2:
            raise ValueError(f"ArrayConfig requires at least 2 antennas, got {positions.shape[0]}")
        if not np.all(np.isfinite(positions)):
            raise ValueError("antenna_positions_enu_m contains non-finite values")

        self.antenna_positions_enu_m = positions

        _warn_on_duplicate_positions(positions)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ArrayConfig":
        """Load an `ArrayConfig` from a YAML file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a YAML file with keys ``name``, ``latitude_deg``,
            ``longitude_deg``, ``height_m``, and ``antennas`` (a list of
            ``[east_m, north_m, up_m]`` triples). See
            ``configs/array_default.yaml`` for the schema and an example.

        Returns
        -------
        ArrayConfig
            The loaded and validated array configuration.

        Raises
        ------
        ValueError
            If the YAML is missing required keys or the geometry fails
            validation (see `ArrayConfig.__post_init__`).
        """
        path = Path(path)
        with path.open("r") as f:
            data = yaml.safe_load(f)

        required_keys = {"latitude_deg", "longitude_deg", "height_m", "antennas"}
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(f"array YAML {path} is missing required keys: {sorted(missing)}")

        positions = np.asarray(data["antennas"], dtype=np.float64)

        return cls(
            antenna_positions_enu_m=positions,
            latitude_deg=data["latitude_deg"],
            longitude_deg=data["longitude_deg"],
            height_m=data["height_m"],
            name=data.get("name", path.stem),
        )

    @property
    def n_antennas(self) -> int:
        """int: Number of antennas in the array."""
        return self.antenna_positions_enu_m.shape[0]

    def baselines_enu_m(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute all antenna-pair baseline vectors, i < j.

        Returns
        -------
        baseline_vectors_enu_m : numpy.ndarray
            Shape ``(n_baselines, 3)`` float64 array of baseline vectors
            ``r_i - r_j`` in ENU meters, for each pair with ``i < j``, in
            the same convention as the visibility definition in
            ``docs/design_stage2.md`` (conjugate on the second antenna).
        index_pairs : numpy.ndarray
            Shape ``(n_baselines, 2)`` int array of the ``(i, j)`` antenna
            index pairs corresponding to each row of
            `baseline_vectors_enu_m`, with ``i < j``.

        Notes
        -----
        ``n_baselines = n_antennas * (n_antennas - 1) / 2``; autos are not
        included.
        """
        n = self.n_antennas
        i_idx, j_idx = np.triu_indices(n, k=1)
        vectors = self.antenna_positions_enu_m[i_idx] - self.antenna_positions_enu_m[j_idx]
        index_pairs = np.stack([i_idx, j_idx], axis=1)
        return vectors, index_pairs


def _to_value(x, unit: u.UnitBase):
    """Convert `x` to a plain value in `unit` if it is a Quantity, else pass through."""
    if isinstance(x, u.Quantity):
        return x.to_value(unit)
    return x


def _warn_on_duplicate_positions(positions: np.ndarray, tol_m: float = 1e-6) -> None:
    """Warn if any two antennas share (near-)identical ENU positions."""
    n = positions.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if np.allclose(positions[i], positions[j], atol=tol_m, rtol=0.0):
                warnings.warn(
                    f"antennas {i} and {j} have duplicate (or near-duplicate) ENU positions",
                    UserWarning,
                    stacklevel=3,
                )

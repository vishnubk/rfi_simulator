# rfi_simulator

Tools for simulating radio-frequency interference (RFI) in radio-interferometric data.

Simulates a small, configurable antenna array observing celestial sources, with RFI injected at the voltage level and propagated through channelization, correlation, and imaging. Currently implemented: point-source sky model, geometric delay tracking, FX correlation, dirty imaging, and a library of RFI sources — narrowband stationary transmitters, broadband impulsive events, TLE-driven satellites (with Doppler and near-field geometry), and ADS-B aircraft — each carrying ground-truth time–frequency contamination labels through every stage. Also included are classical flagging baselines (spectral kurtosis, robust sigma clipping, SumThreshold) and the metrics to score any predicted flag mask against those labels.

## Quickstart

```python
import numpy as np
from astropy.time import Time

from rfi_simulator import (
    ArrayConfig, PointSource, VoltageSimulator, correlate, dirty_image, earth_location, zenith_coord,
)

array = ArrayConfig.from_yaml("configs/array_default.yaml")
t0 = Time("2026-07-30T06:00:00")
phase_center = zenith_coord(earth_location(array), t0)  # phase up on the zenith
source = PointSource.from_lm(phase_center, lm=(0.0087, -0.0052), flux_jy=5.0)

sim = VoltageSimulator(array, phase_center, t0, [source], rng=np.random.default_rng(42))
vis = correlate(sim.blocks())            # ~2 s of data, 45 baselines, 384 channels
image, l_grid, m_grid = dirty_image(vis)

peak = np.unravel_index(np.argmax(image), image.shape)
print(f"peak {image[peak]:.2f} Jy at l={l_grid[peak[1]]:+.4f}, m={m_grid[peak[0]]:+.4f}")
# -> peak 4.82 Jy at l=+0.0086, m=-0.0054
```

## Flagging baselines and scoring

Three classical, non-learned flaggers ship with the library, together with the scoring needed to compare any predicted mask against the simulator's ground truth. Flaggers take plain arrays and never see the labels; masks are boolean with `True` = contaminated.

```python
import numpy as np
from rfi_simulator import (
    flag_scores, mad_clip_mask, pool_truth_accumulations,
    spectral_kurtosis_mask, sumthreshold_mask,
)

block = next(sim.blocks())
voltages = block.data[0]                      # one antenna, (n_chan, n_time)
truth = block.rfi_mask.any(axis=0)            # union over interference sources

# Spectral kurtosis works pre-detection, on accumulations of M time samples,
# so its mask is coarser in time than the labels. `pool_truth_accumulations`
# puts the truth on exactly that grid — including dropping the tail of fewer
# than M samples, about which the flagger reached no decision.
mask = spectral_kurtosis_mask(voltages, m=256)
print(flag_scores(mask, pool_truth_accumulations(truth, m=256)))

# Robust per-channel clipping of the detected power, and the run-finding
# SumThreshold algorithm on its noise-normalized residual. Both decide at
# full resolution, so they score against the labels directly.
power = np.abs(voltages) ** 2
clipped, deviation = mad_clip_mask(power, n_sigma=5.0, return_statistic=True)
runs = sumthreshold_mask(deviation, chi_1=6.0)
print(flag_scores(runs, truth))   # precision, recall, f1, mcc, false-positive rate, ...
```

## Web interface

An interactive browser UI for exploring the simulator — edit the antenna layout on a site plan, add sky and RFI sources, and view per-antenna waterfalls (with ground-truth RFI mask overlays), the dirty image, and uv coverage. It is a thin layer over the library and runs fully offline.

```bash
pip install -e '.[webui]'
rfi-simulator-ui --port 8765   # then open http://127.0.0.1:8765
```

Under active development; documentation and datasets will be published as the project matures.

Licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE).

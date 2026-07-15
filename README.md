# Tidal response method

A self-contained implementation of the **Munk–Cartwright tidal response method**:
estimate the amplitudes and phases of tidal constituents from a sea-level (or
current) time series by regressing the observations onto the astronomical
tide-generating forcing.

Unlike classical harmonic analysis — which fits an independent sinusoid to every
constituent — the response method fits a small set of complex **lag weights**
that define a smooth *admittance* `H(f)` across each tidal band. A constituent's
amplitude and phase are then read off by applying `H` to that constituent's
astronomical forcing line. Because a handful of weights pin down a whole band,
the method is robust on short or gappy records and extends naturally to
**time-varying** admittances (secular trends, or dependence on physical
covariates such as sea level or stratification).

## Install

```bash
pip install -r requirements.txt
```

Required: `numpy`, `pandas`, `scipy`, `skyfield` (the ephemeris `de421.bsp`,
~17 MB, is downloaded automatically on first use). `matplotlib` + `jupyter` are
needed only for the demo; `scikit-learn` and `utide` are optional.

## Quick start

```python
import pandas as pd
from response_method import TidalResponseLS

# ts: DataFrame with a DatetimeIndex and a column named "observations"
model = TidalResponseLS(lat=41.0, lon=-71.0)
model.fit(ts, uniform_lags=(3, 12), symmetrical=True, estimator="ols")

fdf = model.forcing_at(ts.index)                       # astronomical forcing
_, hc = model.harmonic_from_admittance(
            ts.index, fdf, constituents=["M2", "S2", "N2", "K1", "O1"])

print(hc[["constituent", "amp_observations", "phase_observations_deg"]])
```

`hc` has one row per requested constituent with `amp_<col>` and
`phase_<col>_deg`. The constituents may span diurnal and semidiurnal bands.

## The four-step API

| Step | Call | What it does |
|------|------|--------------|
| 1 | `TidalResponseLS(lat, lon)` | pick a station |
| 2 | `model.fit(ts, uniform_lags=(n, dt_h), symmetrical=True, estimator=...)` | fit the admittance |
| 3 | `model.forcing_at(index)` | astronomical forcing at the obs times |
| 4 | `model.harmonic_from_admittance(index, fdf, constituents=[...])` | amplitudes & phases |

Key `fit` options:

- `uniform_lags=(n, dt_hours)` — forcing lags at `0, ±dt, …, ±n·dt` hours
  (`symmetrical=True` adds the negative side). More lags ⇒ more admittance
  freedom per band.
- `estimator` — `"ols"` (least squares) or `"vbayes"` (variational-Bayes ARD:
  automatic shrinkage plus a posterior covariance for uncertainty).
- `trend=True` — let the admittance vary linearly in time; step 4 then also
  returns `amp_rate_<col>_per_year` and `phase_rate_<col>_deg_per_year`.
- forcing caching — `forcing_use_cache=True` with `forcing_cache_dir=...`
  stores the (expensive) forcing on disk so re-fits at a station are fast.

## Demo

[`demo_synthetic.ipynb`](demo_synthetic.ipynb) walks through the whole workflow
on synthetic data: build a sea-level record from known constituents, recover
their amplitudes and phases, plot the fitted band admittance, and (bonus) fit a
linear amplitude **trend**. Regenerate it with `python build_demo.py`.

A note the demo makes concrete: the response method assumes a **smooth
admittance across each band**, so dominant lines (M2, S2, K1, O1) recover to a
few percent while minor satellite lines (e.g. N2) that depart from the band
shape are less tightly constrained — the price of borrowing strength across
constituents.

## Beyond the basics

The admittance can be made to depend on arbitrary **physical covariates**
instead of (or in addition to) time — pass `covariates={name: series}` to `fit`
and query `admittance(..., covariate=name)` — to attribute tidal change to
drivers such as sea-level rise or stratification. See the docstrings in
[`response_method.py`](response_method.py).

## License

MIT — see [LICENSE](LICENSE).

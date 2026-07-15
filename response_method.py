"""
response_method.py
------------------

Refactor: use complex astronomic forcing functions (Munk & Cartwright style), while keeping
a *real* least-squares system by splitting complex inputs into Re/Im regressors.

NEW IN THIS DROP-IN REPLACEMENT
-------------------------------
ForcingGenerator now supports an optional on-disk cache of computed forcing on a regular grid
(default chunking = per-year, grid freq = "6min"). You can:

- Opt in to cache (speed):
      forcing_use_cache=True
- Force recomputation (no cache):
      forcing_use_cache=False   or   forcing_force_recompute=True
- Validate that cached forcing matches uncached forcing:
      forcing_validate_cache=True

Caching is transparent to the rest of the model: build_design_matrix() still receives a base_df
on a regular grid and interpolates to lagged times.

Key points
----------
- ForcingGenerator.compute_base(..., complex_inputs=True) returns complex-valued columns.
- build_design_matrix(..., complex_inputs=True, split_complex=True) builds real regressors:
      y(t) ≈ Σ_l [ a_l Re{x(t+lag_l)} + b_l Im{x(t+lag_l)} ] + ...
  which is equivalent to:
      y(t) ≈ Re{ Σ_l w_l x(t+lag_l) }   with   w_l = a_l - i b_l
- admittance() reconstructs complex weights w_l and returns complex H(f).
- harmonic_from_admittance() supports complex forcing_df directly.

Dependencies
------------
Required:
  numpy, pandas
Optional (recommended):
  scipy, skyfield
Optional:
  sklearn (StandardScaler; fallback provided)
"""

from __future__ import annotations

import math
import os
import json
import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Optional sklearn scaler
try:
    from sklearn.preprocessing import StandardScaler as _SkStandardScaler
except Exception:  # pragma: no cover
    _SkStandardScaler = None

# Optional SciPy
try:
    from scipy.special import lpmv, eval_legendre
    from scipy.special import gammaln as _gammaln
except Exception as _scipy_err:  # pragma: no cover
    lpmv = None
    eval_legendre = None
    _gammaln = None
    _SCIPY_IMPORT_ERROR = _scipy_err
else:
    _SCIPY_IMPORT_ERROR = None

# Optional Skyfield
try:
    from skyfield.api import load, wgs84
except Exception as _skyfield_err:  # pragma: no cover
    load = None
    wgs84 = None
    _SKYFIELD_IMPORT_ERROR = _skyfield_err
else:
    _SKYFIELD_IMPORT_ERROR = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def cosd(x_deg):
    return np.cos(np.deg2rad(x_deg))


def _to_datetime_index(index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index)


class _StandardScalerFallback:
    """Minimal StandardScaler-like behavior (mean/std), used if sklearn isn't available."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        mu = np.nanmean(X, axis=0)
        sig = np.nanstd(X, axis=0)
        sig = np.where(sig < 1e-12, 1.0, sig)
        self.mean_ = mu
        self.scale_ = sig
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _get_scaler():
    return _SkStandardScaler() if _SkStandardScaler is not None else _StandardScalerFallback()


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _as_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return a tz-aware UTC index representing the same instants."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _maybe_back_to_tz(idx_utc: pd.DatetimeIndex, tz) -> pd.DatetimeIndex:
    """Convert UTC tz-aware index back to tz (or drop tz if tz is None)."""
    if tz is None:
        # drop tz info (keep same instants expressed as naive UTC timestamps)
        return idx_utc.tz_convert("UTC").tz_localize(None)
    return idx_utc.tz_convert(tz)


def _date_range_inclusive_left(start, end, freq: str, tz="UTC") -> pd.DatetimeIndex:
    """
    Make a date_range [start, end) with step=freq.
    Safe for both tz-aware and naive inputs.
    """

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Normalize timezone correctly
    if start.tz is None:
        start = start.tz_localize(tz)
    else:
        start = start.tz_convert(tz)

    if end.tz is None:
        end = end.tz_localize(tz)
    else:
        end = end.tz_convert(tz)

    try:
        return pd.date_range(start=start, end=end, freq=freq, inclusive="left")
    except TypeError:  # older pandas
        return pd.date_range(start=start, end=end - pd.Timedelta(nanoseconds=1), freq=freq)



def _interp_df_time(base_df: pd.DataFrame, target_times: pd.DatetimeIndex) -> np.ndarray:
    """
    Time interpolation for real or complex DataFrame columns (vectorized over cols).
    Returns ndarray shape (len(target_times), n_cols) with dtype float64 or complex128.
    """
    base_index = pd.DatetimeIndex(base_df.index)
    t0 = base_index.view("int64").astype(np.float64) / 1e9
    t1 = pd.DatetimeIndex(target_times).view("int64").astype(np.float64) / 1e9

    X0 = base_df.to_numpy()
    is_complex = np.iscomplexobj(X0)
    out = np.empty((len(t1), X0.shape[1]), dtype=np.complex128 if is_complex else np.float64)

    order = np.argsort(t0)
    t0s = t0[order]
    X0s = X0[order, :]

    for j in range(X0s.shape[1]):
        col = X0s[:, j]
        if is_complex:
            out[:, j] = np.interp(t1, t0s, np.real(col)) + 1j * np.interp(t1, t0s, np.imag(col))
        else:
            out[:, j] = np.interp(t1, t0s, col.astype(np.float64))
    return out


# ---------------------------------------------------------------------
# Default input configuration
# ---------------------------------------------------------------------

DEFAULT_INPUT_CONFIG: Dict[str, Dict] = {
    "Radiational": {"degrees": [1, 2], "orders": {1: [1], 2: [1, 2]}},
    "Gravitational": {"degrees": [2, 3], "orders": {2: [1, 2], 3: [1, 2, 3]}},
}


# ---------------------------------------------------------------------
# Forcing generation (COMPLEX) + OPTIONAL DISK CACHE
# ---------------------------------------------------------------------

class ForcingGenerator:
    """
    Generate complex-valued gravitational/radiational input functions at a station.

    Caching
    -------
    If use_cache=True, forcing is computed on a canonical UTC grid at cache_freq (default "6min"),
    stored per-year, and later re-used and time-interpolated to the requested index.

    Cache safety:
    - Cache files are keyed by (lat, lon, ephemeris, input_config, complex_inputs, cache_freq).
    - Changing any of these automatically uses a different cache key (no stale mixing).
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        *,
        input_config: Optional[dict] = None,
        ephemeris: str = "de421.bsp",
        skyfield_cache: Optional[dict] = None,
        # cache controls
        use_cache: bool = False,
        cache_dir: Optional[str] = None,
        cache_freq: str = "6min",
        cache_dtype: Union[str, np.dtype] = "float64",
        cache_chunk: str = "year",  # currently only "year" supported
    ):
        if _SCIPY_IMPORT_ERROR is not None:
            raise ImportError(
                "ForcingGenerator requires SciPy (scipy.special.lpmv, scipy.special.eval_legendre). "
                "Install with: pip install scipy"
            ) from _SCIPY_IMPORT_ERROR

        self.lat = float(lat)
        self.lon = float(lon)
        self.input_config = dict(input_config) if input_config is not None else dict(DEFAULT_INPUT_CONFIG)
        self.ephemeris = ephemeris
        self._skyfield_cache = skyfield_cache if skyfield_cache is not None else {}

        # Physical-ish constants (scaling consistency matters more than absolute)
        self.M_E = 5.9722e24
        self.M_M = 7.3e22
        self.M_S = 1.989e30
        self.E_r = 6371.01e3
        self.solar_constant = 1.946 / 100

        # cache configuration
        self.use_cache_default = bool(use_cache)
        self.cache_freq_default = str(cache_freq)
        self.cache_chunk_default = str(cache_chunk)
        self.cache_dtype_default = np.dtype(cache_dtype).name
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "response_method_forcing")
        self.cache_dir_default = str(cache_dir)
        _ensure_dir(self.cache_dir_default)

        # diagnostic report from last compute_base() call
        self.last_cache_report_: Optional[dict] = None

    # ---------- public cache helpers ----------

    def set_cache(
        self,
        *,
        use_cache: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        cache_freq: Optional[str] = None,
        cache_dtype: Optional[Union[str, np.dtype]] = None,
        cache_chunk: Optional[str] = None,
    ):
        if use_cache is not None:
            self.use_cache_default = bool(use_cache)
        if cache_dir is not None:
            self.cache_dir_default = str(cache_dir)
            _ensure_dir(self.cache_dir_default)
        if cache_freq is not None:
            self.cache_freq_default = str(cache_freq)
        if cache_dtype is not None:
            self.cache_dtype_default = np.dtype(cache_dtype).name
        if cache_chunk is not None:
            self.cache_chunk_default = str(cache_chunk)

    def clear_cache(self, *, years: Optional[Sequence[int]] = None, complex_inputs: Optional[bool] = None):
        """
        Delete cache files for this generator configuration. If years is None, deletes all years.
        If complex_inputs is None, deletes both complex and real cache variants.
        """
        if years is not None:
            years = [int(y) for y in years]

        variants = []
        if complex_inputs is None:
            variants = [True, False]
        else:
            variants = [bool(complex_inputs)]

        removed = 0
        for cx in variants:
            key = self._cache_key(complex_inputs=cx, cache_freq=self.cache_freq_default)
            if years is None:
                # delete all matching
                for fn in os.listdir(self.cache_dir_default):
                    if fn.startswith(f"forcing_{key}_") and fn.endswith(".npz"):
                        try:
                            os.remove(os.path.join(self.cache_dir_default, fn))
                            removed += 1
                        except OSError:
                            pass
            else:
                for y in years:
                    path = self._cache_path(year=y, key=key)
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                            removed += 1
                        except OSError:
                            pass
        return removed

    # ---------- skyfield ----------

    def _get_skyfield(self):
        if _SKYFIELD_IMPORT_ERROR is not None:
            raise ImportError(
                "Skyfield is required to generate forcing inputs. Install with: pip install skyfield"
            ) from _SKYFIELD_IMPORT_ERROR

        if "tscale" not in self._skyfield_cache:
            self._skyfield_cache["tscale"] = load.timescale()
        if "planets" not in self._skyfield_cache:
            self._skyfield_cache["planets"] = load(self.ephemeris)
        return self._skyfield_cache["tscale"], self._skyfield_cache["planets"]

    # ---------- spherical harmonics ----------

    @staticmethod
    def _norm_factor(degree: int, order: int) -> float:
        from math import factorial, sqrt, pi
        return sqrt(((2 * degree + 1) / (4 * pi)) * (factorial(degree - order) / factorial(degree + order)))

    @classmethod
    def spherical_harmonic_dir(cls, degree: int, order: int, theta_rad: np.ndarray, phi_rad: np.ndarray) -> np.ndarray:
        """
        Complex spherical harmonic Y_degree^order(θ, φ) for directional angles:
          - θ = colatitude (zenith angle)
          - φ = azimuth (or longitude-like angle)
        """
        degree = int(degree)
        order = int(order)
        if order < 0 or degree < 0 or order > degree:
            raise ValueError("Require 0 <= order <= degree.")

        mu = np.cos(theta_rad)
        Plm = lpmv(order, degree, mu)
        norm = cls._norm_factor(degree, order)
        return norm * Plm * np.exp(1j * order * phi_rad)

    # ---------- astro compute ----------

    def _compute_astro(self, index: pd.DatetimeIndex) -> dict:
        index = _to_datetime_index(index)
        tscale, planets = self._get_skyfield()
        times = tscale.from_datetimes(index.to_pydatetime())

        earth = planets["earth"]
        moon = planets["moon"]
        sun = planets["sun"]

        earth2moon = (moon.at(times) - earth.at(times)).distance().m
        earth2sun = (sun.at(times) - earth.at(times)).distance().m

        observer = earth + wgs84.latlon(self.lat, self.lon)

        app_moon = observer.at(times).observe(moon).apparent()
        alt_moon, az_moon, dist_moon = app_moon.altaz()
        zenith_moon = 90.0 - alt_moon.degrees

        app_sun = observer.at(times).observe(sun).apparent()
        alt_sun, az_sun, dist_sun = app_sun.altaz()
        zenith_sun = 90.0 - alt_sun.degrees

        max_deg = 0
        for it in ("Radiational", "Gravitational"):
            max_deg = max(max_deg, max(self.input_config.get(it, {}).get("degrees", [0])))

        mu_moon = cosd(zenith_moon)
        mu_sun = cosd(zenith_sun)
        solar_leg = {n: eval_legendre(n, mu_sun) for n in range(max_deg + 1)}
        lunar_leg = {n: eval_legendre(n, mu_moon) for n in range(max_deg + 1)}

        return {
            "station2moon": dist_moon.m,
            "station2sun": dist_sun.m,
            "earth2moon": earth2moon,
            "earth2sun": earth2sun,
            "zenith_moon_deg": np.asarray(zenith_moon, dtype=float),
            "zenith_sun_deg": np.asarray(zenith_sun, dtype=float),
            "az_moon_deg": np.asarray(az_moon.degrees, dtype=float),
            "az_sun_deg": np.asarray(az_sun.degrees, dtype=float),
            "mu_moon": np.asarray(mu_moon, dtype=float),
            "mu_sun": np.asarray(mu_sun, dtype=float),
            "mean_r_moon": float(np.mean(earth2moon)),
            "mean_r_sun": float(np.mean(earth2sun)),
            "Solar_Legendre": solar_leg,
            "Lunar_Legendre": lunar_leg,
        }

    # ---------- forcing pieces ----------

    def radiational(self, degree: int, order: int, astro: dict) -> np.ndarray:
        degree = int(degree)
        order = int(order)
        if degree > 2:
            raise ValueError("Radiational currently implemented for degree <= 2.")

        parallax = 1.0 / 23455.0
        k_n = [1 / 4 + (1 / 6) * parallax, (1 / 2) + (3 / 8) * parallax, (5 / 16) + (1 / 3) * parallax]
        k = k_n[degree]

        zenith_sun = astro["zenith_sun_deg"]
        az_sun = np.deg2rad(astro["az_sun_deg"])
        theta_sun = np.deg2rad(zenith_sun)

        station2sun = astro["station2sun"]
        mean_r_sun = astro["mean_r_sun"]

        Ydir = self.spherical_harmonic_dir(degree, order, theta_sun, az_sun)
        amp = self.solar_constant * (mean_r_sun / station2sun) * k
        rad = amp * Ydir

        rad = np.asarray(rad, dtype=np.complex128)
        rad[(zenith_sun >= 90.0) & (zenith_sun <= 180.0)] = 0.0
        return rad

    def gravitational(self, degree: int, order: int, astro: dict) -> np.ndarray:
        degree = int(degree)
        order = int(order)
        if degree > 3:
            raise ValueError("Gravitational currently implemented for degree <= 3.")

        earth2moon = astro["earth2moon"]
        earth2sun = astro["earth2sun"]
        mean_r_moon = astro["mean_r_moon"]
        mean_r_sun = astro["mean_r_sun"]

        theta_m = np.deg2rad(astro["zenith_moon_deg"])
        phi_m = np.deg2rad(astro["az_moon_deg"])
        theta_s = np.deg2rad(astro["zenith_sun_deg"])
        phi_s = np.deg2rad(astro["az_sun_deg"])

        Y_moon = self.spherical_harmonic_dir(degree, order, theta_m, phi_m)
        Y_sun = self.spherical_harmonic_dir(degree, order, theta_s, phi_s)

        K_n_Moon = self.E_r * (self.M_M / self.M_E) * (self.E_r / earth2moon) ** (degree + 1)
        K_n_Sun = self.E_r * (self.M_S / self.M_E) * (self.E_r / earth2sun) ** (degree + 1)

        grav_Moon = K_n_Moon * (mean_r_moon / earth2moon) ** (degree + 1) * Y_moon
        grav_Sun = K_n_Sun * (mean_r_sun / earth2sun) ** (degree + 1) * Y_sun

        return np.asarray(grav_Moon + grav_Sun, dtype=np.complex128)

    # ---------- caching internals ----------

    def _cache_key(self, *, complex_inputs: bool, cache_freq: str) -> str:
        payload = {
            "v": 1,
            "lat": float(self.lat),
            "lon": float(self.lon),
            "ephemeris": str(self.ephemeris),
            "input_config": self.input_config,
            "complex_inputs": bool(complex_inputs),
            "cache_freq": str(cache_freq),
        }
        s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(s).hexdigest()[:12]

    def _cache_path(self, *, year: int, key: str) -> str:
        return os.path.join(self.cache_dir_default, f"forcing_{key}_{int(year)}.npz")

    def _load_year_cache(self, *, year: int, key: str, complex_inputs: bool) -> Optional[pd.DataFrame]:
        path = self._cache_path(year=year, key=key)
        if not os.path.exists(path):
            return None
        try:
            z = np.load(path, allow_pickle=False)
            cols = z["cols"].astype(str).tolist()
            t_ns = z["t_ns"].astype(np.int64)
            idx_utc = pd.to_datetime(t_ns, utc=True)

            if complex_inputs:
                xr = z["X_real"]
                xi = z["X_imag"]
                X = xr.astype(np.float64) + 1j * xi.astype(np.float64)
            else:
                X = z["X_real"].astype(np.float64)

            return pd.DataFrame(X, index=idx_utc, columns=cols)
        except Exception as e:
            warnings.warn(f"Failed to load cache file {path!r} (will recompute): {e}")
            return None

    def _save_year_cache(
        self,
        *,
        year: int,
        key: str,
        df_utc: pd.DataFrame,
        complex_inputs: bool,
        cache_dtype: np.dtype,
    ):
        path = self._cache_path(year=year, key=key)
        idx = pd.DatetimeIndex(df_utc.index)
        if idx.tz is None:
            raise ValueError("Internal error: cache DataFrame must be UTC tz-aware.")
        if str(idx.tz) != "UTC":
            idx = idx.tz_convert("UTC")

        cols = np.asarray(df_utc.columns, dtype="U")
        t_ns = idx.view("int64").astype(np.int64)

        X = df_utc.to_numpy()
        if complex_inputs:
            xr = np.asarray(np.real(X), dtype=cache_dtype)
            xi = np.asarray(np.imag(X), dtype=cache_dtype)
            np.savez_compressed(path, t_ns=t_ns, cols=cols, X_real=xr, X_imag=xi)
        else:
            xr = np.asarray(X, dtype=cache_dtype)
            np.savez_compressed(path, t_ns=t_ns, cols=cols, X_real=xr)

    def _compute_base_uncached(self, index: pd.DatetimeIndex, *, complex_inputs: bool = True) -> pd.DataFrame:
        """Original forcing computation (no disk cache). Expects index tz-aware UTC (recommended)."""
        index = _to_datetime_index(index)
        astro = self._compute_astro(index)

        cols: Dict[str, np.ndarray] = {}
        for input_type, cfg in self.input_config.items():
            degrees = cfg.get("degrees", [])
            orders_map = cfg.get("orders", {})
            if input_type not in ("Radiational", "Gravitational"):
                continue
            for deg in degrees:
                for ord_ in orders_map.get(int(deg), []):
                    if input_type == "Radiational":
                        vals = self.radiational(deg, ord_, astro)
                    else:
                        vals = self.gravitational(deg, ord_, astro)

                    name = f"{input_type}_{int(deg)}_{int(ord_)}"
                    vals = np.asarray(vals)
                    cols[name] = vals.astype(np.complex128) if complex_inputs else np.real(vals).astype(np.float64)

        return pd.DataFrame(cols, index=index)

    # ---------- public forcing API ----------

    def compute_base(
        self,
        index: pd.DatetimeIndex,
        *,
        complex_inputs: bool = True,
        # cache controls (overrides defaults if not None)
        use_cache: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        cache_freq: Optional[str] = None,
        cache_dtype: Optional[Union[str, np.dtype]] = None,
        cache_chunk: Optional[str] = None,
        force_recompute: bool = False,
        validate_cache: bool = False,
        validate_atol: float = 0.0,
        validate_rtol: float = 0.0,
    ) -> pd.DataFrame:
        """
        Return base forcing inputs on `index`.

        If use_cache=True, uses disk cache (computes & stores missing years).
        If force_recompute=True, recomputes even if cache exists (but can still write cache).
        If validate_cache=True and use_cache=True, recomputes uncached and compares.
        """
        idx_req = _to_datetime_index(index)
        tz_in = idx_req.tz
        idx_req_utc = _as_utc_index(idx_req)

        # resolve cache options
        if use_cache is None:
            use_cache = self.use_cache_default
        if cache_dir is not None:
            self.cache_dir_default = str(cache_dir)
            _ensure_dir(self.cache_dir_default)
        if cache_freq is None:
            cache_freq = self.cache_freq_default
        if cache_dtype is None:
            cache_dtype = np.dtype(self.cache_dtype_default)
        else:
            cache_dtype = np.dtype(cache_dtype)
        if cache_chunk is None:
            cache_chunk = self.cache_chunk_default
        cache_chunk = str(cache_chunk).lower().strip()
        if cache_chunk != "year":
            raise ValueError("Only cache_chunk='year' is supported in this implementation.")

        report = {
            "use_cache": bool(use_cache),
            "force_recompute": bool(force_recompute),
            "cache_freq": str(cache_freq),
            "cache_dtype": cache_dtype.name,
            "cache_dir": str(self.cache_dir_default),
            "complex_inputs": bool(complex_inputs),
            "loaded_years": [],
            "computed_years": [],
            "validated": False,
            "max_abs_diff": None,
        }

        if not use_cache:
            df = self._compute_base_uncached(idx_req_utc, complex_inputs=complex_inputs)
            out = df.copy()
            out.index = _maybe_back_to_tz(pd.DatetimeIndex(out.index), tz_in)
            self.last_cache_report_ = report
            return out

        key = self._cache_key(complex_inputs=complex_inputs, cache_freq=cache_freq)

        years = np.unique(idx_req_utc.year)
        pieces = []
        for y in years:
            df_y = None
            if (not force_recompute):
                df_y = self._load_year_cache(year=int(y), key=key, complex_inputs=complex_inputs)
                if df_y is not None:
                    report["loaded_years"].append(int(y))

            if df_y is None:
                # compute this year on canonical UTC grid
                start = pd.Timestamp(f"{int(y)}-01-01", tz="UTC")
                end = pd.Timestamp(f"{int(y)+1}-01-01", tz="UTC")
                grid_y = _date_range_inclusive_left(start, end, freq=str(cache_freq), tz="UTC")
                df_y = self._compute_base_uncached(grid_y, complex_inputs=complex_inputs)
                report["computed_years"].append(int(y))
                # save cache
                try:
                    self._save_year_cache(year=int(y), key=key, df_utc=df_y, complex_inputs=complex_inputs, cache_dtype=cache_dtype)
                except Exception as e:
                    warnings.warn(f"Failed to write cache for year={y}: {e}")

            pieces.append(df_y)

        base_utc = pd.concat(pieces).sort_index()

        # Return at requested times (interpolate if needed)
        # We interpolate even if exact alignment holds; cheap and robust.
        vals = _interp_df_time(base_utc, idx_req_utc)
        out = pd.DataFrame(vals, index=idx_req_utc, columns=base_utc.columns)
        out.index = _maybe_back_to_tz(pd.DatetimeIndex(out.index), tz_in)

        if validate_cache:
            df_unc = self._compute_base_uncached(idx_req_utc, complex_inputs=complex_inputs)
            vals_unc = df_unc.to_numpy()
            vals_cache = out.copy()
            vals_cache.index = idx_req_utc  # align instants for numeric compare
            vals_cache = vals_cache.to_numpy()

            diff = np.abs(vals_unc - vals_cache)
            max_abs = float(np.nanmax(diff)) if diff.size else 0.0
            report["validated"] = True
            report["max_abs_diff"] = max_abs

            if (validate_atol > 0) or (validate_rtol > 0):
                denom = np.maximum(np.abs(vals_unc), 1.0)
                ok = np.all(diff <= (validate_atol + validate_rtol * denom))
                if not ok:
                    raise AssertionError(
                        f"Forcing cache validation failed: max_abs_diff={max_abs:.3e} "
                        f"(atol={validate_atol}, rtol={validate_rtol}). "
                        f"Try cache_dtype='float64' or set force_recompute=True to rebuild cache."
                    )

        self.last_cache_report_ = report
        return out


# ---------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------

@dataclass
class DesignMatrix:
    X: np.ndarray
    Y: np.ndarray
    feature_names: List[str]
    y_names: List[str]
    index: pd.DatetimeIndex
    linear_groups: Dict[str, List[int]]
    x_scaler_mean: Optional[np.ndarray] = None
    x_scaler_scale: Optional[np.ndarray] = None
    # Trend embedding (linear-in-time admittance)
    trend: bool = False
    trend_t0: Optional[pd.Timestamp] = None
    # Physical-covariate embedding (admittance varies with named covariates z_k(t))
    covariate_names: Optional[List[str]] = None
    covariate_means: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------
# Trend / time helpers
# ---------------------------------------------------------------------

#: Marker appended to a feature name to flag its time-trend (tau-interaction) copy.
TREND_SUFFIX = "::trend"

#: Prefix for markers that flag a physical-covariate interaction copy of a weight.
#: The companion column for covariate ``z_k`` is named ``<base>::cov:<name>`` and
#: carries ``z_k(t) * regressor``, so the response weight (and hence the admittance)
#: becomes ``w(t) = w0 + tau*w_trend + sum_k z_k(t)*w_k``.
COV_PREFIX = "::cov:"


def cov_suffix(name: str) -> str:
    """Feature-name suffix carrying the interaction with covariate ``name``."""
    return COV_PREFIX + str(name)

#: Days in a (Julian) year; matches VTide's convention for cross-comparison.
DAYS_PER_YEAR = 365.2425


def _midpoint_timestamp(index: pd.DatetimeIndex) -> pd.Timestamp:
    """Midpoint instant of a DatetimeIndex (used as the trend reference epoch)."""
    index = pd.DatetimeIndex(index)
    return index[0] + (index[-1] - index[0]) / 2


def tau_years(index, t0) -> np.ndarray:
    """
    Centered time coordinate in years relative to epoch ``t0``.

    Uses Julian dates and a 365.2425-day year so that the response-method trend
    is expressed on exactly the same time axis as VTide's linear amplitude model.
    """
    index = pd.DatetimeIndex(index)
    jd = index.to_julian_date().to_numpy().astype(np.float64)
    jd0 = float(pd.Timestamp(t0).to_julian_date())
    return (jd - jd0) / DAYS_PER_YEAR


def make_uniform_lags(steps: int, tau_hours: float, *, symmetrical: bool = False) -> List[float]:
    steps = int(steps)
    tau_hours = float(tau_hours)
    lags = [-(s * tau_hours) for s in range(steps, 0, -1)] + [0.0]
    if symmetrical:
        lags += [(s * tau_hours) for s in range(1, steps + 1)]
    return lags


def _interp_base_to_times(base_df: pd.DataFrame, target_times: pd.DatetimeIndex) -> np.ndarray:
    """Time interpolation for real or complex base_df columns."""
    return _interp_df_time(base_df, target_times)


def build_design_matrix(
    ts: pd.DataFrame,
    *,
    forcing_generator: Optional[ForcingGenerator] = None,
    base_df: Optional[pd.DataFrame] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    input_config: Optional[dict] = None,
    ephemeris: str = "de421.bsp",
    y_cols: Optional[List[str]] = None,
    lags_hours: Optional[Sequence[float]] = None,
    uniform_lags: Optional[Tuple[int, float]] = None,
    symmetrical: bool = False,
    base_freq: str = "6min",
    linear_inputs: Optional[List[str]] = None,
    include_radiational: bool = True,
    include_gravitational: bool = True,
    complex_inputs: bool = True,
    split_complex: bool = True,
    bilinear: bool = False,
    bilinear_pairs: Optional[List[Tuple[str, str]]] = None,
    bilinear_lag_pairs: Optional[List[Tuple[float, float]]] = None,
    bilinear_same_lag: bool = True,
    add_constant: bool = True,
    standardize_X: bool = True,
    sample_weight: Union[str, np.ndarray, None] = None,
    drop_constant_imag: bool = True,
    imag_tol: float = 1e-12,
    # linear-in-time trend embedding
    trend: bool = False,
    trend_t0=None,
    # physical-covariate embedding: {name: per-timestamp series aligned to ts.index}
    covariates: Optional[Dict[str, np.ndarray]] = None,
    # forcing cache passthrough (all optional)
    forcing_use_cache: Optional[bool] = None,
    forcing_force_recompute: bool = False,
    forcing_validate_cache: bool = False,
    forcing_cache_dir: Optional[str] = None,
    forcing_cache_freq: Optional[str] = None,
    forcing_cache_dtype: Optional[Union[str, np.dtype]] = None,
    forcing_validate_atol: float = 0.0,
    forcing_validate_rtol: float = 0.0,
) -> Tuple[DesignMatrix, Optional[np.ndarray]]:

    if not isinstance(ts.index, pd.DatetimeIndex):
        raise ValueError("ts must have a DatetimeIndex")
    index = pd.DatetimeIndex(ts.index)

    # Linear-in-time trend basis: centered time in years (tau) at the reference epoch.
    # Each lag-weight gets a companion regressor (tau * regressor) so the response
    # weights -- and hence the admittance H(f) -- become w(t) = w0 + w1 * tau.
    if trend:
        if trend_t0 is None:
            trend_t0 = _midpoint_timestamp(index)
        trend_t0 = pd.Timestamp(trend_t0)
        tau_full = tau_years(index, trend_t0)
    else:
        trend_t0 = None
        tau_full = None

    # Physical-covariate embedding. Each named covariate z_k(t) gets companion
    # regressors z_k * regressor, so the response weights -- and hence the
    # admittance H(f) -- become w(t) = w0 + sum_k z_k(t) * w_k. Covariates are
    # centered to their record mean so H0(f) is the mean-state admittance and the
    # recovered w_k are sensitivities per physical unit of z_k.
    cov_centered: Dict[str, np.ndarray] = {}
    cov_means: Dict[str, float] = {}
    if covariates:
        for name, arr in covariates.items():
            a = np.asarray(arr, dtype=np.float64).ravel()
            if a.shape[0] != len(index):
                raise ValueError(
                    f"covariate {name!r} has length {a.shape[0]}, expected len(index)={len(index)}"
                )
            m = float(np.nanmean(a))
            cov_centered[name] = a - m
            cov_means[name] = m
    cov_names = list(cov_centered.keys())

    if y_cols is None:
        if "observations" in ts.columns:
            y_cols = ["observations"]
        elif all(c in ts.columns for c in ["u", "v"]):
            y_cols = ["u", "v"]
        else:
            raise ValueError("Could not infer y_cols; provide y_cols explicitly.")
    y_cols = list(y_cols)

    Y_raw = ts[y_cols].to_numpy()
    if Y_raw.ndim == 1:
        Y_raw = Y_raw.reshape(-1, 1)

    if lags_hours is None:
        if uniform_lags is None:
            raise ValueError("Provide either lags_hours or uniform_lags=(steps,tau_hours)")
        lags_hours = make_uniform_lags(uniform_lags[0], uniform_lags[1], symmetrical=symmetrical)
    lags_hours = [float(x) for x in lags_hours]

    # Build base_df on a regular grid for interpolation
    if base_df is None:
        if forcing_generator is None:
            if lat is None or lon is None:
                raise ValueError("Provide lat/lon or forcing_generator or base_df")
            forcing_generator = ForcingGenerator(lat=lat, lon=lon, input_config=input_config, ephemeris=ephemeris)

        min_lag = float(np.min(lags_hours))
        max_lag = float(np.max(lags_hours))
        start = (index.min() + pd.Timedelta(hours=min_lag)).floor(base_freq)
        end = (index.max() + pd.Timedelta(hours=max_lag)).ceil(base_freq)
        grid = pd.date_range(start=start, end=end, freq=base_freq, tz=index.tz)

        base_df = forcing_generator.compute_base(
            grid,
            complex_inputs=complex_inputs,
            use_cache=forcing_use_cache,
            cache_dir=forcing_cache_dir,
            cache_freq=forcing_cache_freq,
            cache_dtype=forcing_cache_dtype,
            force_recompute=forcing_force_recompute,
            validate_cache=forcing_validate_cache,
            validate_atol=forcing_validate_atol,
            validate_rtol=forcing_validate_rtol,
        )

    # filter columns by type
    keep_cols = []
    for c in base_df.columns:
        if (not include_radiational) and c.startswith("Radiational_"):
            continue
        if (not include_gravitational) and c.startswith("Gravitational_"):
            continue
        keep_cols.append(c)
    base_df = base_df[keep_cols]

    if linear_inputs is None:
        linear_inputs = list(base_df.columns)
    else:
        missing = [c for c in linear_inputs if c not in base_df.columns]
        if missing:
            raise ValueError(f"linear_inputs missing from base_df: {missing}")
        linear_inputs = list(linear_inputs)

    base_col_to_j = {c: j for j, c in enumerate(base_df.columns)}
    base_is_complex = np.iscomplexobj(base_df.to_numpy())

    if complex_inputs and not base_is_complex:
        raise ValueError("complex_inputs=True requires complex base_df (compute_base(complex_inputs=True)).")
    if complex_inputs and not split_complex:
        raise ValueError("This refactor expects split_complex=True when complex_inputs=True (real LS system).")

    # interpolate all base columns for each lag
    lag_to_vals: Dict[float, np.ndarray] = {}
    for lag in lags_hours:
        target_times = index + pd.Timedelta(hours=lag)
        lag_to_vals[lag] = _interp_base_to_times(base_df, target_times)

    feature_blocks: List[np.ndarray] = []
    feature_names: List[str] = []
    linear_groups: Dict[str, List[int]] = {}

    def _append_real_col(col: np.ndarray, name: str, group_key: Optional[str] = None,
                         trendable: bool = True):
        col = np.asarray(col, dtype=np.float64)
        feature_blocks.append(col.reshape(-1, 1))
        feature_names.append(name)
        if group_key is not None:
            linear_groups.setdefault(group_key, []).append(len(feature_names) - 1)
        # Companion tau-interaction column => time-rate of this lag weight.
        if trend and trendable:
            feature_blocks.append((tau_full * col).reshape(-1, 1))
            feature_names.append(name + TREND_SUFFIX)
            if group_key is not None:
                linear_groups.setdefault(group_key + TREND_SUFFIX, []).append(len(feature_names) - 1)
        # Companion covariate-interaction columns => sensitivity of this lag weight
        # to each physical covariate z_k.
        if trendable and cov_names:
            for cname in cov_names:
                feature_blocks.append((cov_centered[cname] * col).reshape(-1, 1))
                feature_names.append(name + cov_suffix(cname))
                if group_key is not None:
                    linear_groups.setdefault(group_key + cov_suffix(cname), []).append(len(feature_names) - 1)

    # linear terms
    for c in linear_inputs:
        j = base_col_to_j[c]
        for lag in lags_hours:
            col = lag_to_vals[lag][:, j]
            if complex_inputs:
                re = np.real(col)
                im = np.imag(col)
                _append_real_col(re, f"{c}_Re@{lag:.6g}h", group_key=f"{c}_Re")
                if (not drop_constant_imag) or (np.nanmax(np.abs(im)) > imag_tol):
                    _append_real_col(im, f"{c}_Im@{lag:.6g}h", group_key=f"{c}_Im")
            else:
                _append_real_col(col, f"{c}@{lag:.6g}h", group_key=c)

    # bilinear terms (optional)
    if bilinear:
        if bilinear_pairs is None:
            bilinear_pairs = []
            for i in range(len(linear_inputs)):
                for j2 in range(i, len(linear_inputs)):
                    bilinear_pairs.append((linear_inputs[i], linear_inputs[j2]))
        else:
            bilinear_pairs = list(bilinear_pairs)

        if bilinear_lag_pairs is None:
            if bilinear_same_lag:
                bilinear_lag_pairs = [(lag, lag) for lag in lags_hours]
            else:
                bilinear_lag_pairs = [(0.0, 0.0)]
        else:
            bilinear_lag_pairs = [(float(a), float(b)) for (a, b) in bilinear_lag_pairs]

        for (c1, c2) in bilinear_pairs:
            j1 = base_col_to_j[c1]
            j2 = base_col_to_j[c2]
            for (lag1, lag2) in bilinear_lag_pairs:
                x1 = lag_to_vals[lag1][:, j1]
                x2 = lag_to_vals[lag2][:, j2]
                prod = x1 * x2
                if complex_inputs:
                    _append_real_col(np.real(prod), f"{c1}*{c2}_Re@{lag1:.6g}h,{lag2:.6g}h", trendable=False)
                    _append_real_col(np.imag(prod), f"{c1}*{c2}_Im@{lag1:.6g}h,{lag2:.6g}h", trendable=False)
                else:
                    _append_real_col(prod, f"{c1}*{c2}@{lag1:.6g}h,{lag2:.6g}h", trendable=False)

    # constant (kept at index 0). With trend, a companion 'const::trend' column
    # absorbs a linear mean-sea-level drift, analogous to VTide's trend term.
    if add_constant:
        front_blocks = [np.ones((len(index), 1), dtype=np.float64)]
        front_names = ["const"]
        if trend:
            front_blocks.append(tau_full.reshape(-1, 1))
            front_names.append("const" + TREND_SUFFIX)
        # Covariate companions of the constant absorb any covariate-correlated drift
        # in the non-tidal mean, so it does not leak into the tidal sensitivities.
        for cname in cov_names:
            front_blocks.append(cov_centered[cname].reshape(-1, 1))
            front_names.append("const" + cov_suffix(cname))
        feature_blocks = front_blocks + feature_blocks
        feature_names = front_names + feature_names
        shift = len(front_names)
        for k in list(linear_groups.keys()):
            linear_groups[k] = [i + shift for i in linear_groups[k]]

    X = np.hstack(feature_blocks) if feature_blocks else np.zeros((len(index), 0), dtype=np.float64)

    # drop NaN rows
    bad_y = np.any(~np.isfinite(Y_raw), axis=1)
    bad_x = np.any(~np.isfinite(X), axis=1) if X.size else np.zeros(len(index), dtype=bool)
    keep = ~(bad_y | bad_x)

    X = X[keep, :]
    Y = Y_raw[keep, :]
    idx_keep = index[keep]

    # weights
    w = None
    if sample_weight is None:
        w = None
    elif isinstance(sample_weight, str) and sample_weight.lower() == "dt":
        dt = np.diff(idx_keep.view("int64")) / 1e9
        if len(dt) == 0:
            w = np.ones(len(idx_keep), dtype=np.float64)
        else:
            dt0 = float(np.median(dt))
            dt_pad = np.concatenate([[dt[0]], dt])
            w = dt_pad / max(dt0, 1e-12)
            w = np.clip(w, 1e-6, np.inf)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)
        if w.shape[0] != len(idx_keep):
            raise ValueError("sample_weight array must match the number of valid samples")

    # standardize
    x_mean = None
    x_scale = None
    if standardize_X and X.size:
        if feature_names and feature_names[0] == "const":
            X0 = X[:, 1:]
            scaler = _get_scaler()
            Xs = scaler.fit_transform(X0)
            X = np.hstack([X[:, [0]], Xs])
            x_mean = getattr(scaler, "mean_", None)
            x_scale = getattr(scaler, "scale_", None)
        else:
            scaler = _get_scaler()
            X = scaler.fit_transform(X)
            x_mean = getattr(scaler, "mean_", None)
            x_scale = getattr(scaler, "scale_", None)

    dm = DesignMatrix(
        X=X,
        Y=Y,
        feature_names=feature_names,
        y_names=y_cols,
        index=idx_keep,
        linear_groups=linear_groups,
        x_scaler_mean=x_mean,
        x_scaler_scale=x_scale,
        trend=bool(trend),
        trend_t0=trend_t0,
        covariate_names=cov_names if cov_names else None,
        covariate_means=cov_means if cov_names else None,
    )
    return dm, w


# ---------------------------------------------------------------------
# Harmonic utilities (complex forcing)
# ---------------------------------------------------------------------

def _to_days(idx: pd.DatetimeIndex, t0=None) -> np.ndarray:
    idx = pd.DatetimeIndex(idx)
    if t0 is None:
        t0 = idx[0]
    return (idx - pd.Timestamp(t0)).total_seconds().to_numpy() / 86400.0


def _multi_constituent_complex_amps_complex_x(
    t_days: np.ndarray, x: np.ndarray, freqs_cpd: np.ndarray, *, include_intercept: bool = True
) -> Tuple[np.ndarray, complex]:
    """
    Joint complex LS:
        x(t) ≈ Σ_k Ck exp(i 2π f_k t) + c0
    Returns (Ck, c0). Works for complex x.
    """
    x = np.asarray(x, dtype=np.complex128)
    freqs_cpd = np.asarray(freqs_cpd, dtype=float)
    cols = [np.exp(1j * 2.0 * np.pi * f * t_days) for f in freqs_cpd]
    if include_intercept:
        cols.append(np.ones_like(t_days, dtype=np.complex128))
    A = np.column_stack(cols).astype(np.complex128)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    if include_intercept:
        return coef[:-1], complex(coef[-1])
    return coef, 0.0 + 0.0j


# ---------------------------------------------------------------------
# Variational Bayes ARD linear solver (self-contained port of VTide's
# bayes_linear_fit_ard; ndarray-based, no np.matrix / external deps).
# ---------------------------------------------------------------------

def _bayes_linear_fit_ard(
    X: np.ndarray,
    y: np.ndarray,
    *,
    a0: float = 1e-2,
    b0: float = 1e-4,
    c0: float = 1e-2,
    d0: float = 1e-4,
    max_iter: int = 500,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Variational Bayesian linear regression with Automatic Relevance Determination.

    Fits ``y ~ X w`` with a per-coefficient Gaussian prior whose precision is learned
    (ARD shrinkage) and a Gamma noise model. This mirrors the estimator used inside
    VTide, so the response method can be solved with the *same* inference engine.

    Parameters
    ----------
    X : (N, D) design matrix (no implicit bias column is added).
    y : (N,) or (N, 1) targets.

    Returns
    -------
    w : (D,) posterior mean weights.
    V : (D, D) posterior covariance of the weights.
    s2 : scalar expected noise variance (1 / E[tau]).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    N, D = X.shape

    X_corr = X.T @ X
    Xy_corr = X.T @ y

    an = a0 + N / 2.0
    gammaln_an = float(_gammaln(an))
    cn = c0 + 0.5
    D_gammaln_cn = D * float(_gammaln(cn))

    E_a = np.full(D, c0 / d0, dtype=np.float64)
    L_last = -np.inf
    V = np.eye(D)
    w = np.zeros((D, 1))

    for _ in range(max_iter):
        invV = np.diag(E_a) + X_corr
        V = np.linalg.inv(invV)
        # log|V| = -log|invV|
        sign, logdet_invV = np.linalg.slogdet(invV)
        logdetV = -logdet_invV
        w = V @ Xy_corr

        resid = X @ w - y
        sse = float(np.sum(resid ** 2))

        w1 = w[:, 0]
        bn = b0 + 0.5 * (sse + np.sum((w1 ** 2) * E_a))
        E_t = an / bn
        dn = d0 + 0.5 * (E_t * (w1 ** 2) + np.diag(V))
        E_a = cn / dn

        L = (
            -0.5 * (E_t * sse + np.sum(X * (X @ V)))
            + 0.5 * logdetV
            - b0 * E_t
            + gammaln_an
            - an * np.log(bn)
            + an
            + D_gammaln_cn
            - cn * np.sum(np.log(dn))
        )
        if L_last > L:
            break
        if abs(L_last - L) < abs(tol * L):
            break
        L_last = L

    s2 = float(bn / an)
    return w[:, 0], V, s2


# ---------------------------------------------------------------------
# Least squares model
# ---------------------------------------------------------------------

class TidalResponseLS:
    """Least-squares tidal response model (linear or bilinear)."""

    def __init__(
        self,
        lat: float,
        lon: float,
        *,
        input_config: Optional[dict] = None,
        ephemeris: str = "de421.bsp",
        # forcing cache defaults (optional)
        forcing_use_cache: bool = False,
        forcing_cache_dir: Optional[str] = None,
        forcing_cache_freq: str = "6min",
        forcing_cache_dtype: Union[str, np.dtype] = "float64",
    ):
        self.lat = float(lat)
        self.lon = float(lon)
        self.input_config = dict(input_config) if input_config is not None else dict(DEFAULT_INPUT_CONFIG)
        self.ephemeris = ephemeris

        self.forcing_generator = ForcingGenerator(
            lat=self.lat,
            lon=self.lon,
            input_config=self.input_config,
            ephemeris=self.ephemeris,
            use_cache=forcing_use_cache,
            cache_dir=forcing_cache_dir,
            cache_freq=forcing_cache_freq,
            cache_dtype=forcing_cache_dtype,
        )

        self.feature_names_: Optional[List[str]] = None
        self.y_names_: Optional[List[str]] = None
        self.coef_: Optional[np.ndarray] = None  # real regressor coefficients
        self.resid_stats_: Optional[dict] = None
        self.n_outputs_: int = 0
        self.estimator_: str = "ols"
        self.coef_cov_: Optional[List[np.ndarray]] = None
        self.coef_noise_var_: Optional[List[float]] = None
        self.trend_: bool = False
        self.trend_t0_: Optional[pd.Timestamp] = None
        self.covariate_names_: Optional[List[str]] = None
        self.covariate_means_: Optional[Dict[str, float]] = None

        self._train_dm: Optional[DesignMatrix] = None
        self._fit_cfg: dict = {}

    def _build_smoothness_penalty(self, dm: DesignMatrix, *, smoothness_order: int = 2) -> np.ndarray:
        """Penalize roughness of lag weights per linear input (applied per Re/Im group)."""
        n_features = dm.X.shape[1]
        rows = []

        def diff_mat(n: int, order: int) -> np.ndarray:
            if order == 1:
                D = np.zeros((n - 1, n))
                for i in range(n - 1):
                    D[i, i] = -1
                    D[i, i + 1] = 1
                return D
            if order == 2:
                if n < 3:
                    return np.zeros((0, n))
                D = np.zeros((n - 2, n))
                for i in range(n - 2):
                    D[i, i] = 1
                    D[i, i + 1] = -2
                    D[i, i + 2] = 1
                return D
            raise ValueError("smoothness_order must be 1 or 2")

        for group_name, idxs in dm.linear_groups.items():
            idxs = [i for i in idxs if dm.feature_names[i] != "const"]
            n = len(idxs)
            D = diff_mat(n, smoothness_order)
            if D.shape[0] == 0:
                continue
            for r in range(D.shape[0]):
                row = np.zeros((n_features,), dtype=np.float64)
                row[idxs] = D[r, :]
                rows.append(row)

        if not rows:
            return np.zeros((0, n_features), dtype=np.float64)
        return np.vstack(rows)

    def fit(
        self,
        ts: pd.DataFrame,
        *,
        y_cols: Optional[List[str]] = None,
        lags_hours: Optional[Sequence[float]] = None,
        uniform_lags: Optional[Tuple[int, float]] = None,
        symmetrical: bool = False,
        base_freq: str = "6min",
        base_df: Optional[pd.DataFrame] = None,
        linear_inputs: Optional[List[str]] = None,
        include_radiational: bool = True,
        include_gravitational: bool = True,
        complex_inputs: bool = True,
        split_complex: bool = True,
        interaction: str = "linear",  # "linear" or "bilinear"
        bilinear_pairs: Optional[List[Tuple[str, str]]] = None,
        bilinear_lag_pairs: Optional[List[Tuple[float, float]]] = None,
        bilinear_same_lag: bool = True,
        add_constant: bool = True,
        standardize_X: bool = True,
        sample_weight: Union[str, np.ndarray, None] = None,
        ridge_lambda: float = 0.0,
        smoothness_lambda: float = 0.0,
        smoothness_order: int = 2,
        # linear-in-time trend embedding
        trend: bool = False,
        trend_t0=None,
        # physical-covariate embedding: {name: per-timestamp series aligned to ts.index}
        covariates: Optional[Dict[str, np.ndarray]] = None,
        # estimator: "ols" (least squares) or "vbayes" (variational Bayes ARD)
        estimator: str = "ols",
        # forcing cache controls (opt-in per fit)
        forcing_use_cache: Optional[bool] = None,
        forcing_force_recompute: bool = False,
        forcing_validate_cache: bool = False,
        forcing_cache_dir: Optional[str] = None,
        forcing_cache_freq: Optional[str] = None,
        forcing_cache_dtype: Optional[Union[str, np.dtype]] = None,
        forcing_validate_atol: float = 0.0,
        forcing_validate_rtol: float = 0.0,
    ) -> "TidalResponseLS":

        if interaction not in ("linear", "bilinear"):
            raise ValueError("interaction must be 'linear' or 'bilinear'")
        estimator = str(estimator).lower().strip()
        if estimator not in ("ols", "vbayes"):
            raise ValueError("estimator must be 'ols' or 'vbayes'")

        dm, w = build_design_matrix(
            ts,
            forcing_generator=self.forcing_generator,
            base_df=base_df,
            y_cols=y_cols,
            lags_hours=lags_hours,
            uniform_lags=uniform_lags,
            symmetrical=symmetrical,
            base_freq=base_freq,
            linear_inputs=linear_inputs,
            include_radiational=include_radiational,
            include_gravitational=include_gravitational,
            complex_inputs=complex_inputs,
            split_complex=split_complex,
            bilinear=(interaction == "bilinear"),
            bilinear_pairs=bilinear_pairs,
            bilinear_lag_pairs=bilinear_lag_pairs,
            bilinear_same_lag=bilinear_same_lag,
            add_constant=add_constant,
            standardize_X=standardize_X,
            sample_weight=sample_weight,
            trend=trend,
            trend_t0=trend_t0,
            covariates=covariates,
            forcing_use_cache=forcing_use_cache,
            forcing_force_recompute=forcing_force_recompute,
            forcing_validate_cache=forcing_validate_cache,
            forcing_cache_dir=forcing_cache_dir,
            forcing_cache_freq=forcing_cache_freq,
            forcing_cache_dtype=forcing_cache_dtype,
            forcing_validate_atol=forcing_validate_atol,
            forcing_validate_rtol=forcing_validate_rtol,
        )

        X = dm.X
        Y = dm.Y
        n, p = X.shape
        k = Y.shape[1]

        if w is not None:
            sw = np.sqrt(w).reshape(-1, 1)
            Xw = X * sw
            Yw = Y * sw
        else:
            Xw = X
            Yw = Y

        # Posterior covariance of the (standardized) coefficients, per output, when
        # available (VBayes). Used to propagate uncertainty onto admittance/trends.
        coef_cov: Optional[List[np.ndarray]] = None
        coef_noise_var: Optional[List[float]] = None

        if estimator == "vbayes":
            if _SCIPY_IMPORT_ERROR is not None:
                raise ImportError("estimator='vbayes' requires scipy.special.gammaln.")
            if (ridge_lambda and ridge_lambda > 0) or (smoothness_lambda and smoothness_lambda > 0):
                warnings.warn(
                    "ridge_lambda/smoothness_lambda are ignored when estimator='vbayes' "
                    "(ARD provides its own shrinkage)."
                )
            coef = np.zeros((p, k), dtype=np.float64)
            coef_cov = []
            coef_noise_var = []
            for j in range(k):
                wj, Vj, s2j = _bayes_linear_fit_ard(Xw, Yw[:, j])
                coef[:, j] = wj
                coef_cov.append(Vj)
                coef_noise_var.append(s2j)
        else:
            blocks_A = [Xw]
            blocks_B = [Yw]

            if ridge_lambda and ridge_lambda > 0:
                A_ridge = math.sqrt(ridge_lambda) * np.eye(p, dtype=np.float64)
                B_ridge = np.zeros((p, k), dtype=np.float64)
                blocks_A.append(A_ridge)
                blocks_B.append(B_ridge)

            if smoothness_lambda and smoothness_lambda > 0:
                L = self._build_smoothness_penalty(dm, smoothness_order=smoothness_order)
                if L.shape[0] > 0:
                    A_smooth = math.sqrt(smoothness_lambda) * L
                    B_smooth = np.zeros((L.shape[0], k), dtype=np.float64)
                    blocks_A.append(A_smooth)
                    blocks_B.append(B_smooth)

            A = np.vstack(blocks_A)
            B = np.vstack(blocks_B)

            coef, *_ = np.linalg.lstsq(A, B, rcond=None)

        Yhat = X @ coef
        resid = Y - Yhat
        ss_res = np.sum(resid**2, axis=0)
        ss_tot = np.sum((Y - np.mean(Y, axis=0, keepdims=True)) ** 2, axis=0)
        r2 = 1.0 - (ss_res / np.maximum(ss_tot, 1e-12))

        self.feature_names_ = dm.feature_names
        self.y_names_ = dm.y_names
        self.coef_ = coef
        self.n_outputs_ = int(k)
        self.estimator_ = estimator
        self.coef_cov_ = coef_cov          # list of (p,p) per output, or None (OLS)
        self.coef_noise_var_ = coef_noise_var
        self.trend_ = bool(dm.trend)
        self.trend_t0_ = dm.trend_t0
        self.covariate_names_ = list(dm.covariate_names) if dm.covariate_names else None
        self.covariate_means_ = dict(dm.covariate_means) if dm.covariate_means else None
        self.resid_stats_ = {
            "n_samples": int(n),
            "n_features": int(p),
            "r2": r2.tolist(),
            "rmse": np.sqrt(np.mean(resid**2, axis=0)).tolist(),
        }
        self._train_dm = dm

        # store fit cfg for predict()
        self._fit_cfg = dict(
            y_cols=list(dm.y_names),
            lags_hours=list(map(float, lags_hours)) if lags_hours is not None else None,
            uniform_lags=tuple(uniform_lags) if uniform_lags is not None else None,
            symmetrical=bool(symmetrical),
            base_freq=str(base_freq),
            linear_inputs=list(linear_inputs) if linear_inputs is not None else None,
            include_radiational=bool(include_radiational),
            include_gravitational=bool(include_gravitational),
            complex_inputs=bool(complex_inputs),
            split_complex=bool(split_complex),
            interaction=str(interaction),
            bilinear_pairs=list(bilinear_pairs) if bilinear_pairs is not None else None,
            bilinear_lag_pairs=list(bilinear_lag_pairs) if bilinear_lag_pairs is not None else None,
            bilinear_same_lag=bool(bilinear_same_lag),
            add_constant=bool(add_constant),
            standardize_X=bool(standardize_X),
            trend=bool(trend),
            trend_t0=self.trend_t0_,
            estimator=str(estimator),
            forcing_use_cache=forcing_use_cache,
            forcing_cache_dir=forcing_cache_dir,
            forcing_cache_freq=forcing_cache_freq,
            forcing_cache_dtype=forcing_cache_dtype,
        )

        return self

    def _apply_training_standardization(self, X: np.ndarray) -> np.ndarray:
        dm = self._train_dm
        if dm is None:
            return X
        if dm.x_scaler_mean is None or dm.x_scaler_scale is None:
            return X
        if self.feature_names_ and len(self.feature_names_) > 0 and self.feature_names_[0] == "const":
            X0 = X[:, 1:]
            Xs = (X0 - dm.x_scaler_mean) / dm.x_scaler_scale
            return np.hstack([X[:, [0]], Xs])
        return (X - dm.x_scaler_mean) / dm.x_scaler_scale

    def predict(self, ts: pd.DataFrame) -> pd.DataFrame:
        if self.coef_ is None or self.feature_names_ is None:
            raise RuntimeError("Model not fit yet")

        cfg = dict(self._fit_cfg)
        dm, _ = build_design_matrix(
            ts,
            forcing_generator=self.forcing_generator,
            y_cols=cfg.get("y_cols", None),
            lags_hours=cfg.get("lags_hours", None),
            uniform_lags=cfg.get("uniform_lags", None),
            symmetrical=cfg.get("symmetrical", False),
            base_freq=cfg.get("base_freq", "6min"),
            linear_inputs=cfg.get("linear_inputs", None),
            include_radiational=cfg.get("include_radiational", True),
            include_gravitational=cfg.get("include_gravitational", True),
            complex_inputs=cfg.get("complex_inputs", True),
            split_complex=cfg.get("split_complex", True),
            bilinear=(cfg.get("interaction", "linear") == "bilinear"),
            bilinear_pairs=cfg.get("bilinear_pairs", None),
            bilinear_lag_pairs=cfg.get("bilinear_lag_pairs", None),
            bilinear_same_lag=cfg.get("bilinear_same_lag", True),
            add_constant=cfg.get("add_constant", True),
            standardize_X=False,
            sample_weight=None,
            trend=cfg.get("trend", False),
            trend_t0=cfg.get("trend_t0", None),
            forcing_use_cache=cfg.get("forcing_use_cache", None),
            forcing_cache_dir=cfg.get("forcing_cache_dir", None),
            forcing_cache_freq=cfg.get("forcing_cache_freq", None),
            forcing_cache_dtype=cfg.get("forcing_cache_dtype", None),
        )

        X = self._apply_training_standardization(dm.X)
        Yhat = X @ self.coef_
        return pd.DataFrame(Yhat, index=dm.index, columns=self.y_names_)

    # --------------------
    # Forcing access helper
    # --------------------

    def forcing_at(
        self,
        index,
        *,
        base_freq: Optional[str] = None,
        linear_inputs: Optional[List[str]] = None,
        include_radiational: Optional[bool] = None,
        include_gravitational: Optional[bool] = None,
        complex_inputs: bool = True,
        forcing_use_cache: Optional[bool] = None,
        forcing_cache_dir: Optional[str] = None,
        forcing_cache_freq: Optional[str] = None,
        forcing_cache_dtype: Optional[Union[str, np.dtype]] = None,
        forcing_force_recompute: bool = False,
        forcing_validate_cache: bool = False,
        forcing_validate_atol: float = 0.0,
        forcing_validate_rtol: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute (and time-interpolate) the complex forcing at the given times.
        Useful for building forcing_df for harmonic_from_admittance().

        This respects the same cache controls as fit().
        """
        idx = pd.DatetimeIndex(index)

        if base_freq is None:
            base_freq = self._fit_cfg.get("base_freq", "6min")
        if linear_inputs is None:
            linear_inputs = self._fit_cfg.get("linear_inputs", None)
        if include_radiational is None:
            include_radiational = bool(self._fit_cfg.get("include_radiational", True))
        if include_gravitational is None:
            include_gravitational = bool(self._fit_cfg.get("include_gravitational", True))

        start = idx.min().floor(base_freq)
        end = idx.max().ceil(base_freq)
        grid = pd.date_range(start=start, end=end, freq=base_freq, tz=idx.tz)

        base = self.forcing_generator.compute_base(
            grid,
            complex_inputs=complex_inputs,
            use_cache=forcing_use_cache,
            cache_dir=forcing_cache_dir,
            cache_freq=forcing_cache_freq,
            cache_dtype=forcing_cache_dtype,
            force_recompute=forcing_force_recompute,
            validate_cache=forcing_validate_cache,
            validate_atol=forcing_validate_atol,
            validate_rtol=forcing_validate_rtol,
        )

        keep = []
        for c in base.columns:
            if (not include_radiational) and c.startswith("Radiational_"):
                continue
            if (not include_gravitational) and c.startswith("Gravitational_"):
                continue
            keep.append(c)
        base = base[keep]

        if linear_inputs is not None:
            base = base[linear_inputs]

        vals = _interp_base_to_times(base, idx)
        return pd.DataFrame(vals, index=idx, columns=base.columns)

    # --------------------
    # Coeff helper
    # --------------------

    def _coef_unstandardized(self) -> np.ndarray:
        """Undo StandardScaler scaling (does not adjust intercept shift; fine for admittance)."""
        if self.coef_ is None:
            raise RuntimeError("Model not fit yet")
        dm = self._train_dm
        if dm is None or dm.x_scaler_scale is None:
            return self.coef_

        coef = np.array(self.coef_, dtype=float, copy=True)
        has_const = (self.feature_names_ is not None and len(self.feature_names_) > 0 and self.feature_names_[0] == "const")
        offset = 1 if has_const else 0

        scales = np.asarray(dm.x_scaler_scale, dtype=float)
        if scales.ndim != 1 or scales.shape[0] != coef.shape[0] - offset:
            raise ValueError("Scaler scale shape does not match coefficient shape")
        coef[offset:, :] = coef[offset:, :] / scales[:, None]
        return coef

    def _resolve_output_index(self, output=0) -> int:
        n_out = int(getattr(self, "n_outputs_", 1))
        if isinstance(output, int):
            if output < 0 or output >= n_out:
                raise IndexError(f"output={output} out of range")
            return output
        y_names = list(getattr(self, "y_names_", []))
        if not y_names:
            raise ValueError("Model has no y_names_")
        return y_names.index(output)

    # --------------------
    # Admittance
    # --------------------

    def admittance(
        self,
        freqs_cpd: np.ndarray,
        *,
        feature_prefix: str,
        lags_hours: Optional[Sequence[float]] = None,
        output_index: int = 0,
        use_unstandardized: bool = True,
        sign: str = "design",
        component: str = "base",
        covariate: Optional[str] = None,
    ) -> np.ndarray:
        """
        Admittance H(f) for ONE forcing prefix.

        With split complex inputs, coefficients live on:
            prefix_Re@lag   and  prefix_Im@lag
        and complex lag weights are reconstructed as:
            w(lag) = a(lag) - i b(lag)

        Parameters
        ----------
        component : {'base', 'rate'}
            'base' -> H0(f), the admittance at the reference state (all covariates
            and tau at their centered zero).
            'rate' -> H1(f), d H / d tau, the admittance trend per year (requires a
            model fit with trend=True). At time tau (years from the reference epoch)
            the admittance is  H(f, tau) = H0(f) + tau * H1(f); see admittance_at_time.
        covariate : str, optional
            If given, return the admittance sensitivity dH/dz to the named physical
            covariate (requires a model fit with that covariate). This overrides
            ``component``; the admittance at covariate value z is H0(f) + z * dH/dz.
        """
        if self.coef_ is None or self.feature_names_ is None:
            raise RuntimeError("Model not fit yet")
        component = str(component).lower().strip()

        # Resolve which family of companion columns to read: base (no suffix),
        # the time-trend companions (TREND_SUFFIX), or a covariate's companions.
        if covariate is not None:
            cov_list = getattr(self, "covariate_names_", None) or []
            if covariate not in cov_list:
                raise RuntimeError(
                    f"covariate={covariate!r} not among fitted covariates {cov_list}"
                )
            want_suffix = cov_suffix(covariate)
        elif component == "rate":
            if not getattr(self, "trend_", False):
                raise RuntimeError("component='rate' requires a model fit with trend=True")
            want_suffix = TREND_SUFFIX
        elif component == "base":
            want_suffix = ""
        else:
            raise ValueError("component must be 'base' or 'rate'")

        coef = self._coef_unstandardized() if use_unstandardized else self.coef_
        fnames = list(self.feature_names_)

        # Base weights live on names ending in 'h'; their companions end in a
        # '::...' suffix. Select the requested family and parse the lag accordingly.
        def _match(nm: str, tag: str) -> Optional[float]:
            """Return the lag (hours) if nm is the requested column for `tag` in {_Re,_Im, ''}."""
            if "*" in nm:
                return None
            head = feature_prefix + tag + "@"
            if not nm.startswith(head):
                return None
            rest = nm[len(head):]
            if want_suffix:
                if not rest.endswith(want_suffix):
                    return None
                rest = rest[: -len(want_suffix)]
            else:
                # base family: exclude every companion ('::trend', '::cov:<name>', ...)
                if "::" in rest:
                    return None
            if not rest.endswith("h"):
                return None
            try:
                return float(rest[:-1])
            except Exception:
                return None

        lag_to_idx_re: Dict[float, int] = {}
        lag_to_idx_im: Dict[float, int] = {}
        lag_to_idx_real: Dict[float, int] = {}
        for i, nm in enumerate(fnames):
            lag = _match(nm, "_Re")
            if lag is not None:
                lag_to_idx_re[lag] = i
                continue
            lag = _match(nm, "_Im")
            if lag is not None:
                lag_to_idx_im[lag] = i
                continue
            lag = _match(nm, "")
            if lag is not None:
                lag_to_idx_real[lag] = i

        if lags_hours is None:
            lags = sorted(set(list(lag_to_idx_re.keys()) + list(lag_to_idx_im.keys()) + list(lag_to_idx_real.keys())))
            if not lags:
                fam = (f"covariate {covariate!r} " if covariate is not None
                       else ("trend " if component == "rate" else ""))
                raise ValueError(
                    f"No {fam}lagged weights found for feature_prefix={feature_prefix!r}"
                )
            lags_hours = lags
        else:
            lags_hours = [float(l) for l in lags_hours]

        w_lag = np.zeros((len(lags_hours),), dtype=np.complex128)
        for k, lag in enumerate(lags_hours):
            a = 0.0
            b = 0.0
            if lag in lag_to_idx_re:
                a = float(coef[lag_to_idx_re[lag], output_index])
            elif lag in lag_to_idx_real:
                a = float(coef[lag_to_idx_real[lag], output_index])
            if lag in lag_to_idx_im:
                b = float(coef[lag_to_idx_im[lag], output_index])
            w_lag[k] = a - 1j * b

        lag_days = np.asarray(lags_hours, dtype=float) / 24.0
        freqs_cpd = np.asarray(freqs_cpd, dtype=float)

        if sign == "design":
            ex = np.exp(+1j * 2.0 * np.pi * freqs_cpd[:, None] * lag_days[None, :])
        elif sign == "convolution":
            ex = np.exp(-1j * 2.0 * np.pi * freqs_cpd[:, None] * lag_days[None, :])
        else:
            raise ValueError("sign must be 'design' or 'convolution'")

        return ex @ w_lag

    def admittance_at_time(
        self,
        freqs_cpd: np.ndarray,
        tau_years: float,
        *,
        feature_prefix: str,
        output_index: int = 0,
        use_unstandardized: bool = True,
        sign: str = "design",
    ) -> np.ndarray:
        """
        Admittance evaluated at a given time offset (years from the trend reference
        epoch): H(f, tau) = H0(f) + tau * H1(f). For a non-trend model this is just H0.
        """
        H0 = self.admittance(
            freqs_cpd, feature_prefix=feature_prefix, output_index=output_index,
            use_unstandardized=use_unstandardized, sign=sign, component="base",
        )
        if not getattr(self, "trend_", False) or tau_years == 0:
            return H0
        H1 = self.admittance(
            freqs_cpd, feature_prefix=feature_prefix, output_index=output_index,
            use_unstandardized=use_unstandardized, sign=sign, component="rate",
        )
        return H0 + float(tau_years) * H1

    # --------------------
    # Plotting
    # --------------------

    def plot_admittance_band(
        self,
        feature_prefix,
        band="semidiurnal",
        output=0,
        n_grid=1200,
        constituents=None,
        show_constituents=True,
        unwrap_phase=True,
        figsize=(10, 7),
        dpi=150,
    ):
        import matplotlib.pyplot as plt

        BANDS = {
            "diurnal": (0.80, 1.20, ["Q1", "O1", "P1", "K1"]),
            "semidiurnal": (1.70, 2.20, ["N2", "M2", "S2", "K2"]),
            "terdiurnal": (2.70, 3.30, ["M3"]),
            "quarterdiurnal": (3.70, 4.30, ["M4"]),
        }

        FREQS = {
            "Q1": 0.893244, "O1": 0.929536, "P1": 0.997262, "K1": 1.002738,
            "N2": 1.895982, "M2": 1.932274, "S2": 2.0, "K2": 2.005476,
            "M3": 2.898411, "M4": 3.864548,
        }

        def get_freqs(names):
            freq_map = {}
            try:
                from utide import ut_constants
                const_names = [str(n).strip() for n in ut_constants.const.name]
                const_freq = np.asarray(ut_constants.const.freq, dtype=float) * 24.0  # cph -> cpd
                freq_map = dict(zip(const_names, const_freq))
            except Exception:
                freq_map = {}
            return {n: float(freq_map[n]) if n in freq_map else float(FREQS[n]) for n in names}

        out_i = self._resolve_output_index(output)
        band = str(band).lower().strip()
        if band not in BANDS:
            raise ValueError(f"Unknown band {band!r}. Options: {list(BANDS.keys())}")

        fmin, fmax, default_cons = BANDS[band]
        if constituents is None:
            constituents = default_cons

        freqs = np.linspace(fmin, fmax, int(n_grid))
        Hout = self.admittance(freqs, feature_prefix=feature_prefix, output_index=out_i)

        mag = np.abs(Hout)
        ph = np.angle(Hout)
        if unwrap_phase:
            ph = np.unwrap(ph)
        ph_deg = np.degrees(ph)

        fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=figsize, dpi=dpi)

        ax_mag.plot(freqs, mag, linewidth=1.5)
        ax_mag.set_ylabel("|H(f)|")
        ax_mag.grid(True, alpha=0.3)
        ax_mag.set_title(f"Admittance: '{feature_prefix}' ({band}, output={out_i})", fontweight="bold")

        ax_ph.plot(freqs, ph_deg, linewidth=1.5)
        ax_ph.set_ylabel("Phase [deg]")
        ax_ph.set_xlabel("Frequency [cycles/day]")
        ax_ph.grid(True, alpha=0.3)

        if show_constituents and constituents:
            c_freqs = get_freqs(constituents)
            for name, f_cpd in c_freqs.items():
                ax_mag.axvline(f_cpd, color="gray", ls="--", lw=1, alpha=0.6)
                ax_ph.axvline(f_cpd, color="gray", ls="--", lw=1, alpha=0.6)
                y_pos = ax_ph.get_ylim()[0] + 0.02 * np.diff(ax_ph.get_ylim())[0]
                ax_ph.text(f_cpd, y_pos, f" {name}", rotation=90, va="bottom", ha="left", fontsize=9, color="gray")

        fig.tight_layout()
        return fig

    def plot_admittance_all_bands(self, feature_prefix, output=0, bands=None, figsize=(12, 10), dpi=150):
        import matplotlib.pyplot as plt

        BANDS = {
            "diurnal": (0.80, 1.20, ["Q1", "O1", "P1", "K1"]),
            "semidiurnal": (1.70, 2.20, ["N2", "M2", "S2", "K2"]),
            "terdiurnal": (2.70, 3.30, ["M3"]),
            "quarterdiurnal": (3.70, 4.30, ["M4"]),
        }

        if bands is None:
            bands = ["diurnal", "semidiurnal", "terdiurnal"]

        out_i = self._resolve_output_index(output)

        n_bands = len(bands)
        fig, axes = plt.subplots(n_bands, 2, figsize=figsize, dpi=dpi)
        if n_bands == 1:
            axes = axes.reshape(1, -1)

        for i, band in enumerate(bands):
            band = str(band).lower().strip()
            if band not in BANDS:
                raise ValueError(f"Unknown band {band!r}. Options: {list(BANDS.keys())}")
            fmin, fmax, _ = BANDS[band]
            freqs = np.linspace(fmin, fmax, 800)

            H = self.admittance(freqs, feature_prefix=feature_prefix, output_index=out_i)
            mag = np.abs(H)
            ph_deg = np.degrees(np.unwrap(np.angle(H)))

            axes[i, 0].plot(freqs, mag, linewidth=1.5)
            axes[i, 0].set_title(band.capitalize(), fontweight="bold")
            axes[i, 0].set_ylabel("|H(f)|")
            axes[i, 0].grid(True, alpha=0.3)

            axes[i, 1].plot(freqs, ph_deg, linewidth=1.5)
            axes[i, 1].set_ylabel("Phase [deg]")
            axes[i, 1].grid(True, alpha=0.3)

            if i == n_bands - 1:
                axes[i, 0].set_xlabel("Frequency [cycles/day]")
                axes[i, 1].set_xlabel("Frequency [cycles/day]")

        fig.suptitle(f"Admittance: '{feature_prefix}' (output={out_i})", fontweight="bold", y=0.995)
        fig.tight_layout()
        return fig

    # --------------------
    # Harmonic from admittance (complex forcing supported)
    # --------------------

    def harmonic_from_admittance(
        self,
        index,
        forcing_df,
        band="semidiurnal",
        constituents=None,
        feature_prefixes=None,
        include_intercept=False,
        t0=None,
        covariates=None,
    ):
        import numpy as np
        import pandas as pd

        BANDS = {
            "diurnal": (0.80, 1.20, ["Q1", "O1", "P1", "K1"]),
            "semidiurnal": (1.70, 2.20, ["N2", "M2", "S2", "K2"]),
            "terdiurnal": (2.70, 3.30, ["M3"]),
            "quarterdiurnal": (3.70, 4.30, ["M4"]),
        }

        FREQS = {
            "Q1": 0.893244, "O1": 0.929536, "P1": 0.997262, "K1": 1.002738,
            "N2": 1.895982, "M2": 1.932274, "S2": 2.0, "K2": 2.005476,
            "M3": 2.898411, "M4": 3.864548,
        }

        def get_freqs(names):
            freq_map = {}
            try:
                from utide import ut_constants
                const_names = [str(n).strip() for n in ut_constants.const.name]
                const_freq = np.asarray(ut_constants.const.freq, dtype=float) * 24.0  # cph -> cpd
                freq_map = dict(zip(const_names, const_freq))
            except Exception:
                freq_map = {}
            out = {}
            for n in names:
                if n in freq_map:
                    out[n] = float(freq_map[n])
                elif n in FREQS:
                    out[n] = float(FREQS[n])
                else:
                    raise KeyError(
                        f"Unknown tidal constituent {n!r}: not found in utide.ut_constants "
                        f"or the built-in frequency table."
                    )
            return out

        index = pd.DatetimeIndex(index)
        band = str(band).lower().strip()
        if band not in BANDS:
            raise ValueError(f"Unknown band {band!r}. Options: {list(BANDS.keys())}")

        if constituents is None:
            constituents = BANDS[band][2]

        freq_map = get_freqs(constituents)
        names = list(freq_map.keys())
        freqs = np.array([freq_map[n] for n in names], dtype=float)

        y_names = list(getattr(self, "y_names_", [])) or ["y0"]
        n_out = len(y_names)

        if feature_prefixes is None:
            model_prefixes = set()
            for nm in (getattr(self, "feature_names_", []) or []):
                if "*" in nm:
                    continue
                if nm.endswith("h") and "@" in nm:
                    prefix = nm.split("@", 1)[0]
                    if prefix.endswith("_Re"):
                        prefix = prefix[:-3]
                    if prefix.endswith("_Im"):
                        prefix = prefix[:-3]
                    if prefix != "const":
                        model_prefixes.add(prefix)
            feature_prefixes = [c for c in forcing_df.columns if c in model_prefixes]
            if not feature_prefixes:
                raise ValueError("No matching forcing features between forcing_df and fitted model.")

        forcing_df = forcing_df.sort_index()
        forcing_at_index = forcing_df.reindex(forcing_df.index.union(index)).interpolate(method="time").reindex(index)
        if forcing_at_index.isna().any().any():
            bad_cols = forcing_at_index.columns[forcing_at_index.isna().any(axis=0)].tolist()
            raise ValueError(f"forcing_df could not be interpolated onto all requested times. Bad cols: {bad_cols}")

        t_days = _to_days(index, t0)
        trend = bool(getattr(self, "trend_", False))

        # Resolve which physical covariates to report sensitivities for.
        fitted_covs = list(getattr(self, "covariate_names_", None) or [])
        if covariates is None:
            cov_list = fitted_covs
        else:
            cov_list = [c for c in covariates]
            missing = [c for c in cov_list if c not in fitted_covs]
            if missing:
                raise ValueError(
                    f"requested covariates {missing} were not in the fit ({fitted_covs})"
                )

        # Yk0 = complex constituent amplitude at the reference state (tau=0, covariates=0).
        # Yk1 = its rate of change per year (trend). Ykc[name] = its sensitivity to the
        #       physical covariate `name`, obtained by applying the corresponding
        #       admittance component to the (time-invariant) forcing constituents Xk.
        Yk0 = np.zeros((len(freqs), n_out), dtype=np.complex128)
        Yk1 = np.zeros((len(freqs), n_out), dtype=np.complex128)
        Ykc = {c: np.zeros((len(freqs), n_out), dtype=np.complex128) for c in cov_list}

        has_bilinear = any("*" in nm for nm in (getattr(self, "feature_names_", []) or []))
        if has_bilinear:
            print("[harmonic_from_admittance] WARNING: fitted model contains bilinear terms; ignoring them here.")

        # The model output is REAL: y = Re{ sum_l w_l x(t+lag) }. For COMPLEX (Munk-Cartwright)
        # forcing the +f and -f spectral lines are independent, and Re{} folds both into the
        # real constituent line. The analytic amplitude A_k (so that the contribution equals
        # Re{A_k e^{i 2pi f_k t}}) is therefore
        #     A_k = sum_p [ X^+_{p,k} H_p(+f_k)  +  conj( X^-_{p,k} H_p(-f_k) ) ],
        # where X^{+/-} are the forcing amplitudes at +/-f_k. (For real forcing
        # X^- = conj(X^+) and the two terms coincide, recovering the usual formula.)
        freqs_pm = np.concatenate([freqs, -freqs])
        nfk = len(freqs)
        for prefix in feature_prefixes:
            x = np.asarray(forcing_at_index[prefix].to_numpy(), dtype=np.complex128)
            Xpm, _ = _multi_constituent_complex_amps_complex_x(t_days, x, freqs_pm, include_intercept=True)
            Xp, Xm = Xpm[:nfk], Xpm[nfk:]

            for oi in range(n_out):
                H0p = self.admittance(freqs, feature_prefix=prefix, output_index=oi,
                                      use_unstandardized=True, sign="design", component="base")
                H0m = self.admittance(-freqs, feature_prefix=prefix, output_index=oi,
                                      use_unstandardized=True, sign="design", component="base")
                Yk0[:, oi] += Xp * H0p + np.conj(Xm * H0m)
                if trend:
                    H1p = self.admittance(freqs, feature_prefix=prefix, output_index=oi,
                                          use_unstandardized=True, sign="design", component="rate")
                    H1m = self.admittance(-freqs, feature_prefix=prefix, output_index=oi,
                                          use_unstandardized=True, sign="design", component="rate")
                    Yk1[:, oi] += Xp * H1p + np.conj(Xm * H1m)
                for cname in cov_list:
                    Hcp = self.admittance(freqs, feature_prefix=prefix, output_index=oi,
                                          use_unstandardized=True, sign="design", covariate=cname)
                    Hcm = self.admittance(-freqs, feature_prefix=prefix, output_index=oi,
                                          use_unstandardized=True, sign="design", covariate=cname)
                    Ykc[cname][:, oi] += Xp * Hcp + np.conj(Xm * Hcm)

        # Reference (tau=0) reconstruction of the band.
        y = np.zeros((len(index), n_out), dtype=float)
        for k, f in enumerate(freqs):
            exp_term = np.exp(1j * 2.0 * np.pi * float(f) * t_days)
            y += np.real(exp_term[:, None] * Yk0[k, :][None, :])

        if include_intercept and "const" in (getattr(self, "feature_names_", []) or []):
            const_idx = list(self.feature_names_).index("const")
            y += np.asarray(self.coef_[const_idx, :], dtype=float)[None, :]

        y_harm = pd.DataFrame(y, index=index, columns=y_names)

        # amplitude/phase and their per-year rates from the linearly-evolving complex
        # amplitude Y(tau) = Yk0 + tau * Yk1.  With Yk0 = a0 + i b0, Yk1 = a1 + i b1:
        #   amp        = |Yk0|
        #   d amp/dtau = (a0 a1 + b0 b1) / amp                 (matches VTide)
        #   d phi/dtau = (a0 b1 - b0 a1) / (a0^2 + b0^2)       [rad/yr -> deg/yr]
        cols = {"constituent": names, "freq_cpd": freqs}
        for i, nm in enumerate(y_names):
            a0 = np.real(Yk0[:, i]); b0 = np.imag(Yk0[:, i])
            amp = np.abs(Yk0[:, i])
            cols[f"amp_{nm}"] = amp
            cols[f"phase_{nm}_deg"] = np.degrees(np.angle(Yk0[:, i]))
            denom = np.where(amp > 1e-12, amp, 1e-12)
            if trend:
                a1 = np.real(Yk1[:, i]); b1 = np.imag(Yk1[:, i])
                cols[f"amp_rate_{nm}_per_year"] = (a0 * a1 + b0 * b1) / denom
                cols[f"phase_rate_{nm}_deg_per_year"] = np.degrees((a0 * b1 - b0 * a1) / (denom ** 2))
            # Sensitivities to each physical covariate: with Y(z) = Yk0 + z * Ykc,
            #   d amp / dz   = (a0 ac + b0 bc) / |Yk0|     (length units per unit z)
            #   d phase / dz = (a0 bc - b0 ac) / |Yk0|^2   (rad/z -> deg/z)
            for cname in cov_list:
                ac = np.real(Ykc[cname][:, i]); bc = np.imag(Ykc[cname][:, i])
                cols[f"amp_sens_{nm}_d_{cname}"] = (a0 * ac + b0 * bc) / denom
                cols[f"phase_sens_{nm}_d_{cname}_deg"] = np.degrees((a0 * bc - b0 * ac) / (denom ** 2))

        hc = pd.DataFrame(cols)
        return y_harm, hc

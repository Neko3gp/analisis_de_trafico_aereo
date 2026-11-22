"""Kernels de Numba (CPU, paralelos con prange; aprovechan todos los núcleos del M1).

Si Numba no está instalado, el mismo código corre como Python puro (lento, pero con
la misma lógica), lo que permite probarlo sin Numba. `HAVE_NUMBA` dice si hay aceleración real.
"""
import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - depende del entorno
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

EARTH_RADIUS_KM = 6371.0088
_DEG = np.pi / 180.0


@njit(cache=True)
def _hav(la1, lo1, la2, lo2):
    dlat = (la2 - la1) * _DEG
    dlon = (lo2 - lo1) * _DEG
    a = np.sin(dlat / 2.0) ** 2 + np.cos(la1 * _DEG) * np.cos(la2 * _DEG) * np.sin(dlon / 2.0) ** 2
    if a > 1.0:
        a = 1.0
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


@njit(parallel=True, cache=True)
def mean_std(x):
    """Media y desviación estándar (poblacional) ignorando NaN."""
    n = 0
    s = 0.0
    for i in prange(x.shape[0]):
        v = x[i]
        if not np.isnan(v):
            s += v
            n += 1
    if n == 0:
        return np.nan, np.nan
    m = s / n
    ss = 0.0
    for i in prange(x.shape[0]):
        v = x[i]
        if not np.isnan(v):
            ss += (v - m) * (v - m)
    return m, np.sqrt(ss / n)


@njit(parallel=True, cache=True)
def pearson(x, y):
    """Correlación de Pearson sobre pares sin NaN; NaN si n<2 o varianza cero."""
    n = 0
    sx = 0.0
    sy = 0.0
    for i in prange(x.shape[0]):
        if not (np.isnan(x[i]) or np.isnan(y[i])):
            n += 1
            sx += x[i]
            sy += y[i]
    if n < 2:
        return np.nan
    mx = sx / n
    my = sy / n
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for i in prange(x.shape[0]):
        if not (np.isnan(x[i]) or np.isnan(y[i])):
            dx = x[i] - mx
            dy = y[i] - my
            sxx += dx * dx
            syy += dy * dy
            sxy += dx * dy
    if sxx <= 0.0 or syy <= 0.0:
        return np.nan
    r = sxy / np.sqrt(sxx * syy)
    return min(1.0, max(-1.0, r))


@njit(parallel=True, cache=True)
def haversine_km(lat1, lon1, lat2, lon2):
    out = np.empty(lat1.shape[0])
    for i in prange(lat1.shape[0]):
        out[i] = _hav(lat1[i], lon1[i], lat2[i], lon2[i])
    return out


@njit(parallel=True, cache=True)
def simulate_routes(lat1, lon1, lat2, lon2, speed_kmh, n_steps):
    """Tiempo estimado (horas): longitud de la ruta partida en n_steps segmentos / velocidad."""
    n = lat1.shape[0]
    out = np.empty(n)
    for i in prange(n):
        sp = speed_kmh[i]
        if np.isnan(sp) or sp <= 0.0:
            out[i] = np.nan
            continue
        total = 0.0
        plat = lat1[i]
        plon = lon1[i]
        for k in range(1, n_steps + 1):
            t = k / n_steps
            clat = lat1[i] + (lat2[i] - lat1[i]) * t
            clon = lon1[i] + (lon2[i] - lon1[i]) * t
            total += _hav(plat, plon, clat, clon)
            plat = clat
            plon = clon
        out[i] = total / sp
    return out

"""Cálculos numéricos con dos opciones: numpy (base) y numba (compilado y en paralelo).

get_backend("numba") devuelve Numba o, si no está instalado, numpy con un aviso.
"""
import warnings
from functools import lru_cache
from time import perf_counter

import numpy as np

from src.compute.numba_kernels import EARTH_RADIUS_KM, HAVE_NUMBA

_DEG = np.pi / 180.0
_CHUNK = 20_000  # filas por bloque en simulate_routes (acota la memoria)


def _f64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _hav(la1, lo1, la2, lo2):
    dlat, dlon = (la2 - la1) * _DEG, (lo2 - lo1) * _DEG
    a = np.sin(dlat / 2) ** 2 + np.cos(la1 * _DEG) * np.cos(la2 * _DEG) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


class Backend:
    """Versión con numpy. NumbaBackend reemplaza los métodos pesados por kernels compilados."""
    name = "numpy"

    def mean_std(self, x):
        x = _f64(x)
        x = x[~np.isnan(x)]
        if x.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(x)), float(np.std(x))

    def outlier_mask(self, x, z: float = 3.0) -> np.ndarray:
        """True donde el valor está a más de z desviaciones estándar de la media."""
        x = _f64(x)
        m, s = self.mean_std(x)
        if not np.isfinite(s) or s == 0:
            return np.zeros(x.shape, dtype=bool)
        with np.errstate(invalid="ignore"):
            return np.abs(x - m) / s > z  # NaN compara False

    def pearson(self, x, y) -> float:
        """Correlación de Pearson ignorando los pares con NaN."""
        x, y = _f64(x), _f64(y)
        keep = ~(np.isnan(x) | np.isnan(y))
        x, y = x[keep], y[keep]
        if x.size < 2:
            return float("nan")
        da, db = x - x.mean(), y - y.mean()
        sxx, syy = float((da * da).sum()), float((db * db).sum())
        if sxx <= 0 or syy <= 0:
            return float("nan")
        r = float((da * db).sum()) / np.sqrt(sxx * syy)
        return float(min(1.0, max(-1.0, r)))

    def haversine_km(self, lat1, lon1, lat2, lon2) -> np.ndarray:
        arrs = np.broadcast_arrays(*[_f64(a) for a in (lat1, lon1, lat2, lon2)])
        return _hav(*arrs)

    def _route_km(self, lat1, lon1, lat2, lon2, n_steps: int) -> np.ndarray:
        t = (np.arange(n_steps + 1) / n_steps)[None, :]
        a1, o1, a2, o2 = (v[:, None] for v in (lat1, lon1, lat2, lon2))
        lat, lon = a1 + (a2 - a1) * t, o1 + (o2 - o1) * t
        seg = _hav(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
        return seg.sum(axis=1)

    def simulate_routes(self, lat1, lon1, lat2, lon2, speed_kmh, n_steps: int) -> np.ndarray:
        """Horas estimadas por vuelo: ruta partida en n_steps segmentos / velocidad (NaN si no hay)."""
        if n_steps < 1:
            raise ValueError("n_steps debe ser >= 1")
        la1, lo1, la2, lo2, sp = (_f64(a) for a in (lat1, lon1, lat2, lon2, speed_kmh))
        km = np.empty(la1.shape[0])
        for s in range(0, km.size, _CHUNK):
            e = s + _CHUNK
            km[s:e] = self._route_km(la1[s:e], lo1[s:e], la2[s:e], lo2[s:e], n_steps)
        with np.errstate(invalid="ignore", divide="ignore"):
            hours = km / sp
        hours[~(sp > 0)] = np.nan
        return hours


class NumbaBackend(Backend):
    name = "numba"

    def __init__(self):
        from src.compute import numba_kernels as nk
        if not nk.HAVE_NUMBA:
            raise ImportError("numba no está instalado")
        self.nk = nk

    @staticmethod
    def _c(a):
        return np.ascontiguousarray(a, dtype=np.float64)

    def mean_std(self, x):
        m, s = self.nk.mean_std(self._c(x))
        return float(m), float(s)

    def pearson(self, x, y) -> float:
        return float(self.nk.pearson(self._c(x), self._c(y)))

    def haversine_km(self, lat1, lon1, lat2, lon2):
        arrs = np.broadcast_arrays(*[_f64(a) for a in (lat1, lon1, lat2, lon2)])
        return self.nk.haversine_km(*[self._c(a) for a in arrs])

    def simulate_routes(self, lat1, lon1, lat2, lon2, speed_kmh, n_steps: int):
        if n_steps < 1:
            raise ValueError("n_steps debe ser >= 1")
        return self.nk.simulate_routes(*[self._c(a) for a in (lat1, lon1, lat2, lon2, speed_kmh)],
                                       int(n_steps))


@lru_cache(maxsize=None)
def available_backends() -> list:
    return ["numpy", "numba"] if HAVE_NUMBA else ["numpy"]


def get_backend(name: str = "numba") -> Backend:
    if (name or "").lower() == "numba":
        try:
            return NumbaBackend()
        except ImportError:
            pass
    if (name or "").lower() != "numpy":
        warnings.warn(f"Backend '{name}' no disponible; usando 'numpy'", RuntimeWarning, stacklevel=2)
    return Backend()


def benchmark(n: int = 100_000, n_steps: int = 64, seed: int = 0) -> dict:
    """Segundos que tarda cada backend en simular n rutas."""
    rng = np.random.default_rng(seed)
    lat1, lat2 = rng.uniform(15, 32, n), rng.uniform(15, 32, n)
    lon1, lon2 = rng.uniform(-117, -86, n), rng.uniform(-117, -86, n)
    sp = rng.uniform(600, 900, n)
    times = {}
    for name in available_backends():
        b = get_backend(name)
        warm = slice(0, min(n, 16))  # la primera llamada de numba compila: se excluye del tiempo
        b.simulate_routes(lat1[warm], lon1[warm], lat2[warm], lon2[warm], sp[warm], n_steps)
        t0 = perf_counter()
        b.simulate_routes(lat1, lon1, lat2, lon2, sp, n_steps)
        times[name] = perf_counter() - t0
    return times

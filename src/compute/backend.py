"""Capa de abstracción para motores de cálculo numérico (NumPy y Numba).

Ofrece una interfaz uniforme (`Backend`) para dos implementaciones:
- Backend base: Operaciones vectorizadas puras sobre NumPy.
- NumbaBackend: Rutinas compiladas JIT con ejecución multinúcleo en paralelo.

La función fábrica `get_backend("numba")` devuelve la implementación acelerada
o realiza un fallback automático a NumPy con una advertencia si Numba no está instalado.
"""

import warnings
from functools import lru_cache
from time import perf_counter

import numpy as np

from src.compute.numba_kernels import EARTH_RADIUS_KM, HAVE_NUMBA

# Constante de conversión a radianes
_DEG = np.pi / 180.0

# Tamaño de bloque para procesar rutas en NumPy sin agotar memoria RAM
_CHUNK = 20_000


def _f64(a) -> np.ndarray:
    """Garantiza que el arreglo de entrada tenga tipo float64."""
    return np.asarray(a, dtype=np.float64)


def _hav(la1, lo1, la2, lo2):
    """Cálculo vectorizado de la fórmula de Haversine con NumPy."""
    dlat, dlon = (la2 - la1) * _DEG, (lo2 - lo1) * _DEG
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(la1 * _DEG) * np.cos(la2 * _DEG) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


class Backend:
    """Motor numérico base implementado con NumPy.

    Sirve como referencia estándar de exactitud numérica y como fallback
    cuando Numba no está disponible en la plataforma.
    """

    name = "numpy"

    def mean_std(self, x) -> tuple:
        """Calcula la media y la desviación estándar ignorando valores NaN.

        Args:
            x: Arreglo de números.

        Returns:
            tuple[float, float]: (media, desviación_estándar).
        """
        x = _f64(x)
        x = x[~np.isnan(x)]
        if x.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(x)), float(np.std(x))

    def outlier_mask(self, x, z: float = 3.0) -> np.ndarray:
        """Genera una máscara booleana para identificar valores atípicos mediante Z-score (|z| > 3.0).

        Args:
            x: Arreglo de datos.
            z (float): Umbral de desviación estándar (por defecto: 3.0).

        Returns:
            np.ndarray: Arreglo booleano con True en las posiciones de valores atípicos.
        """
        x = _f64(x)
        m, s = self.mean_std(x)
        if not np.isfinite(s) or s == 0:
            return np.zeros(x.shape, dtype=bool)
        with np.errstate(invalid="ignore"):
            return np.abs(x - m) / s > z  # Los valores NaN evalúan a False automáticamente

    def pearson(self, x, y) -> float:
        """Calcula el coeficiente de correlación lineal de Pearson ignorando pares con NaN.

        Args:
            x: Primer vector de datos.
            y: Segundo vector de datos.

        Returns:
            float: Coeficiente de correlación en el rango [-1, 1], o NaN.
        """
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
        """Calcula distancias geodésicas en km mediante broadcasting de NumPy.

        Args:
            lat1, lon1: Coordenadas de origen en grados decimales.
            lat2, lon2: Coordenadas de destino en grados decimales.

        Returns:
            np.ndarray: Distancias calculadas en kilómetros.
        """
        arrs = np.broadcast_arrays(*[_f64(a) for a in (lat1, lon1, lat2, lon2)])
        return _hav(*arrs)

    def _route_km(self, lat1, lon1, lat2, lon2, n_steps: int) -> np.ndarray:
        """Calcula la distancia acumulada de trayectorias subdivididas en n_steps tramos."""
        t = (np.arange(n_steps + 1) / n_steps)[None, :]
        a1, o1, a2, o2 = (v[:, None] for v in (lat1, lon1, lat2, lon2))
        lat, lon = a1 + (a2 - a1) * t, o1 + (o2 - o1) * t
        seg = _hav(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
        return seg.sum(axis=1)

    def simulate_routes(
        self,
        lat1,
        lon1,
        lat2,
        lon2,
        speed_kmh,
        n_steps: int,
    ) -> np.ndarray:
        """Estima las horas de vuelo dividiendo la ruta en segmentos de interpolación lineal.

        Procesa los datos en bloques (`_CHUNK`) para evitar el consumo desmedido de memoria.

        Args:
            lat1, lon1: Coordenadas de origen.
            lat2, lon2: Coordenadas de destino.
            speed_kmh: Velocidades de vuelo en km/h.
            n_steps (int): Cantidad de divisiones por ruta.

        Returns:
            np.ndarray: Vector con las horas de vuelo estimadas.
        """
        if n_steps < 1:
            raise ValueError("n_steps debe ser >= 1")
        la1, lo1, la2, lo2, sp = (_f64(a) for a in (lat1, lon1, lat2, lon2, speed_kmh))
        km = np.empty(la1.shape[0])

        # Procesar en bloques para acotar el uso de memoria RAM
        for s in range(0, km.size, _CHUNK):
            e = s + _CHUNK
            km[s:e] = self._route_km(la1[s:e], lo1[s:e], la2[s:e], lo2[s:e], n_steps)

        with np.errstate(invalid="ignore", divide="ignore"):
            hours = km / sp
        hours[~(sp > 0)] = np.nan
        return hours


class NumbaBackend(Backend):
    """Motor numérico de alto rendimiento acelerado con Numba.

    Reemplaza las funciones críticas por kernels compilados JIT con paralelismo
    en todos los núcleos del CPU disponibles.
    """

    name = "numba"

    def __init__(self):
        """Inicializa el backend verificando que Numba esté instalado."""
        from src.compute import numba_kernels as nk
        if not nk.HAVE_NUMBA:
            raise ImportError("numba no está instalado")
        self.nk = nk

    @staticmethod
    def _c(a):
        """Garantiza arreglos contiguos en memoria C con tipo float64 para los kernels."""
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
        return self.nk.simulate_routes(
            *[self._c(a) for a in (lat1, lon1, lat2, lon2, speed_kmh)],
            int(n_steps),
        )


@lru_cache(maxsize=None)
def available_backends() -> list:
    """Retorna la lista de motores de cálculo disponibles en el entorno actual.

    Returns:
        list[str]: ['numpy', 'numba'] si Numba está disponible, o ['numpy'].
    """
    return ["numpy", "numba"] if HAVE_NUMBA else ["numpy"]


def get_backend(name: str = "numba") -> Backend:
    """Fábrica para instanciar el motor de cálculo solicitado.

    Si se solicita 'numba' pero no está instalado, retorna transparentemente
    una instancia de `Backend` (NumPy) emitiendo una advertencia.

    Args:
        name (str): Nombre del motor deseado ('numba' o 'numpy').

    Returns:
        Backend: Instancia del backend listo para operar.
    """
    if (name or "").lower() == "numba":
        try:
            return NumbaBackend()
        except ImportError:
            pass

    if (name or "").lower() != "numpy":
        warnings.warn(
            f"Backend '{name}' no disponible; usando 'numpy'",
            RuntimeWarning,
            stacklevel=2,
        )
    return Backend()


def benchmark(n: int = 100_000, n_steps: int = 64, seed: int = 0) -> dict:
    """Compara el tiempo de ejecución entre los motores disponibles al simular n rutas.

    Incluye una fase previa de calentamiento (warm-up) para excluir el tiempo de
    compilación JIT de Numba de la medición final.

    Args:
        n (int): Cantidad de rutas sintéticas a simular (por defecto: 100,000).
        n_steps (int): Cantidad de tramos por ruta (por defecto: 64).
        seed (int): Semilla para datos aleatorios reproducibles.

    Returns:
        dict: Diccionario {nombre_backend: segundos_de_ejecución}.
    """
    rng = np.random.default_rng(seed)
    lat1, lat2 = rng.uniform(15, 32, n), rng.uniform(15, 32, n)
    lon1, lon2 = rng.uniform(-117, -86, n), rng.uniform(-117, -86, n)
    sp = rng.uniform(600, 900, n)
    times = {}

    for name in available_backends():
        b = get_backend(name)
        # Calentamiento (warm-up): fuerza la compilación JIT previa de Numba
        warm = slice(0, min(n, 16))
        b.simulate_routes(lat1[warm], lon1[warm], lat2[warm], lon2[warm], sp[warm], n_steps)

        # Medición precisa del tiempo de ejecución
        t0 = perf_counter()
        b.simulate_routes(lat1, lon1, lat2, lon2, sp, n_steps)
        times[name] = perf_counter() - t0

    return times

"""Kernels compilados en código máquina con Numba para cálculo numérico de alto rendimiento.

Aprovecha el paralelismo a nivel de hilos (OpenMP / TBB según la plataforma) mediante `prange`,
optimizando el rendimiento en procesadores multinúcleo (incluyendo Apple Silicon M1/M2/M3 y x86_64).

Mecanismo de tolerancia y fallback:
Si Numba no está disponible en el entorno de ejecución, se provee un decorador ficticio (`njit`)
y un alias `prange = range`, permitiendo que el mismo código se ejecute como Python estándar
sin alterar los resultados numéricos. La constante `HAVE_NUMBA` refleja si existe aceleración nativa.
"""

import numpy as np

# Intentar importar Numba; si no está disponible, definir reemplazos transparentes en Python puro
try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - depende de las dependencias instaladas en el entorno
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        """Decorador passthrough cuando Numba no está instalado."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

# Radio medio volumétrico de la Tierra según el estándar geodésico IUGG (en kilómetros)
EARTH_RADIUS_KM = 6371.0088

# Factor de conversión de grados sexagesimales a radianes
_DEG = np.pi / 180.0


@njit(cache=True)
def _hav(la1: float, lo1: float, la2: float, lo2: float) -> float:
    """Calcula la distancia geodésica ortodrómica entre dos coordenadas mediante la fórmula de Haversine.

    Aplica la aproximación esférica de la Tierra. Incluye una guarda numérica
    para asegurar que el argumento de `arcsin(sqrt(a))` no supere 1.0 debido
    a imprecisiones de redondeo en punto flotante.

    Args:
        la1 (float): Latitud del punto 1 en grados.
        lo1 (float): Longitud del punto 1 en grados.
        la2 (float): Latitud del punto 2 en grados.
        lo2 (float): Longitud del punto 2 en grados.

    Returns:
        float: Distancia en kilómetros sobre la superficie terrestre.
    """
    dlat = (la2 - la1) * _DEG
    dlon = (lo2 - lo1) * _DEG
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(la1 * _DEG) * np.cos(la2 * _DEG) * np.sin(dlon / 2.0) ** 2
    )
    # Acotación para prevenir errores de dominio matemático en arcsin
    if a > 1.0:
        a = 1.0
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


@njit(parallel=True, cache=True)
def mean_std(x: np.ndarray):
    r"""Calcula la media y desviación estándar poblacional en paralelo ignorando valores NaN.

    Implementa un algoritmo de dos pasadas:
    1. Primera pasada con `prange`: acumula la suma de valores finitos y el conteo $n$.
    2. Segunda pasada con `prange`: acumula las diferencias cuadráticas $(x_i - \mu)^2$
       para obtener una varianza numéricamente estable.

    Args:
        x (np.ndarray): Arreglo unidimensional de números en punto flotante (float64).

    Returns:
        tuple[float, float]: Tupla (media, desviación_estándar). Si no hay valores válidos,
                             retorna (np.nan, np.nan).
    """
    n = 0
    s = 0.0
    # 1. Acumular suma y conteo de valores no NaN
    for i in prange(x.shape[0]):
        v = x[i]
        if not np.isnan(v):
            s += v
            n += 1

    if n == 0:
        return np.nan, np.nan

    m = s / n
    ss = 0.0
    # 2. Acumular suma de cuadrados de desvíos respecto a la media
    for i in prange(x.shape[0]):
        v = x[i]
        if not np.isnan(v):
            ss += (v - m) * (v - m)

    return m, np.sqrt(ss / n)


@njit(parallel=True, cache=True)
def pearson(x: np.ndarray, y: np.ndarray) -> float:
    r"""Calcula el coeficiente de correlación lineal de Pearson entre dos arreglos en paralelo.

    Descarta cualquier par $(x_i, y_i)$ donde al menos uno de los dos valores sea NaN.
    Si el número de pares válidos es menor a 2, o si la varianza de alguna de las series es cero,
    devuelve `np.nan`. El resultado final se acota al intervalo teórico [-1.0, 1.0].

    Args:
        x (np.ndarray): Primer vector de observaciones (float64).
        y (np.ndarray): Segundo vector de observaciones (float64).

    Returns:
        float: Coeficiente $r \in [-1, 1]$, o np.nan si no es computable.
    """
    n = 0
    sx = 0.0
    sy = 0.0
    # 1. Conteo y sumas para medias
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

    # 2. Acumular covarianzas y varianzas
    for i in prange(x.shape[0]):
        if not (np.isnan(x[i]) or np.isnan(y[i])):
            dx = x[i] - mx
            dy = y[i] - my
            sxx += dx * dx
            syy += dy * dy
            sxy += dx * dy

    # Si alguna variable es constante, la correlación no está definida
    if sxx <= 0.0 or syy <= 0.0:
        return np.nan

    r = sxy / np.sqrt(sxx * syy)
    # Acotar para evitar desbordes leves por precisión de flotantes (ej. 1.0000000000000002)
    return min(1.0, max(-1.0, r))


@njit(parallel=True, cache=True)
def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calcula distancias geodésicas punto a punto entre vectores de coordenadas en paralelo.

    Args:
        lat1 (np.ndarray): Latitudes iniciales en grados decimales.
        lon1 (np.ndarray): Longitudes iniciales en grados decimales.
        lat2 (np.ndarray): Latitudes finales en grados decimales.
        lon2 (np.ndarray): Longitudes finales en grados decimales.

    Returns:
        np.ndarray: Vector de distancias correspondientes en kilómetros.
    """
    out = np.empty(lat1.shape[0])
    for i in prange(lat1.shape[0]):
        out[i] = _hav(lat1[i], lon1[i], lat2[i], lon2[i])
    return out


@njit(parallel=True, cache=True)
def simulate_routes(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    speed_kmh: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Simula trayectorias de vuelo subdivididas en múltiples pasos y calcula el tiempo estimado.

    Para cada par de coordenadas (origen -> destino), discretiza la ruta en `n_steps`
    segmentos geodésicos lineales, sumando la distancia de cada segmento y dividiéndola
    por la velocidad de crucero para obtener el tiempo de vuelo estimado en horas.

    Args:
        lat1 (np.ndarray): Latitud del punto de partida.
        lon1 (np.ndarray): Longitud del punto de partida.
        lat2 (np.ndarray): Latitud del punto de llegada.
        lon2 (np.ndarray): Longitud del punto de llegada.
        speed_kmh (np.ndarray): Velocidad horizontal de la aeronave en km/h.
        n_steps (int): Número de subdivisiones para interpolar la trayectoria.

    Returns:
        np.ndarray: Arreglo con las horas de vuelo estimadas para cada aeronave
                    (NaN si la velocidad no es válida o es <= 0).
    """
    n = lat1.shape[0]
    out = np.empty(n)

    for i in prange(n):
        sp = speed_kmh[i]
        # Si la velocidad es nula, negativa o NaN, el cálculo no es posible
        if np.isnan(sp) or sp <= 0.0:
            out[i] = np.nan
            continue

        total = 0.0
        plat = lat1[i]
        plon = lon1[i]

        # Interpolar n_steps segmentos a lo largo del trayecto
        for k in range(1, n_steps + 1):
            t = k / n_steps
            clat = lat1[i] + (lat2[i] - lat1[i]) * t
            clon = lon1[i] + (lon2[i] - lon1[i]) * t
            total += _hav(plat, plon, clat, clon)
            plat = clat
            plon = clon

        # Tiempo = distancia total / velocidad media
        out[i] = total / sp

    return out

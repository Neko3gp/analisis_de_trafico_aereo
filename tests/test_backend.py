import numpy as np
import pytest

from src.compute import numba_kernels as nk
from src.compute.backend import available_backends, benchmark, get_backend

RNG = np.random.default_rng(0)
NAMES = available_backends()


def tol(name):
    return dict(rtol=1e-9, atol=1e-9)


def test_numpy_always_available():
    assert "numpy" in NAMES


@pytest.mark.parametrize("name", NAMES)
def test_mean_std_ignores_nan(name):
    x = np.array([1.0, 2.0, np.nan, 4.0, 7.0])
    m, s = get_backend(name).mean_std(x)
    ref = x[~np.isnan(x)]
    assert np.isclose(m, ref.mean(), **tol(name)) and np.isclose(s, ref.std(), **tol(name))


@pytest.mark.parametrize("name", NAMES)
def test_mean_std_empty_or_all_nan(name):
    b = get_backend(name)
    for x in (np.array([]), np.array([np.nan, np.nan])):
        m, s = b.mean_std(x)
        assert np.isnan(m) and np.isnan(s)


@pytest.mark.parametrize("name", NAMES)
def test_outlier_mask_flags_extreme_and_ignores_nan(name):
    x = np.concatenate([RNG.normal(0, 1, 200), [100.0, np.nan]])
    mask = get_backend(name).outlier_mask(x, z=3.0)
    assert mask.dtype == bool and mask[200] and not mask[201]


@pytest.mark.parametrize("name", NAMES)
def test_outlier_mask_constant_array_flags_nothing(name):
    assert not get_backend(name).outlier_mask(np.full(10, 5.0)).any()


@pytest.mark.parametrize("name", NAMES)
def test_pearson_matches_numpy(name):
    x = RNG.normal(size=500)
    y = 2 * x + RNG.normal(size=500)
    assert np.isclose(get_backend(name).pearson(x, y), np.corrcoef(x, y)[0, 1], **tol(name))


@pytest.mark.parametrize("name", NAMES)
def test_pearson_drops_nan_pairs(name):
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, np.nan])
    assert np.isclose(get_backend(name).pearson(x, y), 1.0, **tol(name))


@pytest.mark.parametrize("name", NAMES)
def test_pearson_zero_variance_or_tiny_is_nan(name):
    b = get_backend(name)
    assert np.isnan(b.pearson(np.ones(5), np.arange(5.0)))
    assert np.isnan(b.pearson(np.array([1.0]), np.array([2.0])))
    assert np.isnan(b.pearson(np.array([]), np.array([])))


@pytest.mark.parametrize("name", NAMES)
def test_haversine_quarter_equator(name):
    d = get_backend(name).haversine_km(np.array([0.0]), np.array([0.0]),
                                       np.array([0.0]), np.array([90.0]))
    assert np.isclose(d[0], 10007.54, rtol=1e-3)


@pytest.mark.parametrize("name", NAMES)
def test_simulate_routes_equator_ten_degrees(name):
    h = get_backend(name).simulate_routes(
        np.array([0.0, 0.0]), np.array([0.0, 0.0]),
        np.array([0.0, 0.0]), np.array([10.0, 10.0]),
        np.array([800.0, np.nan]), 32)
    assert np.isclose(h[0], 1111.95 / 800.0, rtol=1e-3)
    assert np.isnan(h[1])


@pytest.mark.parametrize("name", NAMES)
def test_backends_agree_on_routes(name):
    n = 300
    lat1, lon1 = RNG.uniform(15, 32, n), RNG.uniform(-117, -86, n)
    lat2, lon2 = RNG.uniform(15, 32, n), RNG.uniform(-117, -86, n)
    sp = RNG.uniform(600, 900, n)
    ref = get_backend("numpy").simulate_routes(lat1, lon1, lat2, lon2, sp, 16)
    got = get_backend(name).simulate_routes(lat1, lon1, lat2, lon2, sp, 16)
    assert np.allclose(got, ref, **tol(name))


def test_unknown_backend_falls_back_to_numpy():
    with pytest.warns(RuntimeWarning):
        assert get_backend("no-existe").name == "numpy"


# La lógica de los kernels de Numba se valida también sin Numba (corre como Python puro)
def test_numba_kernel_logic_matches_numpy():
    x = np.array([1.0, np.nan, 3.0, 8.0])
    m, s = nk.mean_std(x)
    ref = x[~np.isnan(x)]
    assert np.isclose(m, ref.mean()) and np.isclose(s, ref.std())
    a = RNG.normal(size=50)
    b = a + RNG.normal(size=50)
    assert np.isclose(nk.pearson(a, b), np.corrcoef(a, b)[0, 1])
    h = nk.simulate_routes(np.array([0.0]), np.array([0.0]), np.array([0.0]),
                           np.array([10.0]), np.array([800.0]), 8)
    assert np.isclose(h[0], 1111.95 / 800.0, rtol=1e-3)


def test_benchmark_returns_timings_for_available_backends():
    t = benchmark(n=500, n_steps=8)
    assert "numpy" in t and all(v > 0 for v in t.values())
    assert set(t) <= set(NAMES)

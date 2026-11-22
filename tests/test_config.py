from src.config import load_config


def test_demo_when_no_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AIRLABS_API_KEY=\nOPENWEATHER_API_KEY=\n")
    cfg = load_config(str(env))
    assert cfg.demo_mode is True


def test_defaults_airports(tmp_path):
    cfg = load_config(str(tmp_path / "missing.env"))
    assert cfg.airports == ["CUU", "MEX", "GDL", "MTY", "TIJ"]


def test_backend_default_numba(tmp_path):
    cfg = load_config(str(tmp_path / "missing.env"))
    assert cfg.backend == "numba"


def test_keys_disable_demo(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AIRLABS_API_KEY=abc\nOPENWEATHER_API_KEY=def\nDEMO_MODE=0\n")
    cfg = load_config(str(env))
    assert cfg.demo_mode is False
    assert cfg.airlabs_key == "abc"


def test_explicit_demo_overrides_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AIRLABS_API_KEY=abc\nOPENWEATHER_API_KEY=def\nDEMO_MODE=1\n")
    assert load_config(str(env)).demo_mode is True


def test_airports_from_env_and_comments(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comentario\nAIRPORTS=CUU, mex\n")
    assert load_config(str(env)).airports == ["CUU", "MEX"]

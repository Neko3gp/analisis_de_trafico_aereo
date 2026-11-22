# SDD ledger — plan: /home/claude/proyecto-trafico-aereo/docs/plans/2026-11-20-trafico-aereo.md

Pre-flight: Task 3 produce `generate_demo` (llaves flights/weather/air_quality/sensors); Task 4 (`fetch_all`) y Task 5 (`build_hourly`) lo consumen con las mismas columnas. Task 6 produce `Backend`; Task 7 lo consume. Task 5 produce `build_hourly`; Tasks 7 y 8 lo consumen. Sin conflictos.

Ruling: no hay repositorio git ni hay que hacer commits (el plan ya lo indica) y no se usan los scripts sdd (task-start/task-done/review-package) porque dependen de git — el avance se registra en este ledger y las pruebas se corren con pytest directamente — si falla: se pierde el rango de commits por tarea, sin costo para el usuario.

Task 1: complete (tests: pytest tests/test_config.py -> 6 passed)
Task 1: Ruling: run_pipeline.sh llama a src.ingest.run_ingest (punto de entrada de ingesta que el plan no listaba) — se crea en la Task 4 junto a clientes.py — si falla: la ingesta no se puede lanzar desde el script, solo desde Streamlit.
Task 2: complete (tests: pytest tests/test_storage.py -> 8 passed)
Task 3: complete (tests: pytest tests/test_demo.py -> 6 passed)
Task 4: complete (tests: pytest tests/test_clients.py tests/test_run_ingest.py -> 16 passed)
Task 4: Ruling: OpenWeather air_pollution es la fuente base de pm25/pm10 y OpenAQ se suma solo si hay clave (endpoint v3 sin verificar sin red) — el PDF pide ambas pero OpenAQ v3 exige ubicar sensores por separado — si falla: sin OpenAQ los PM salen solo de OpenWeather.
Task 4: Ruling: se agregó src/schema.py con las listas de columnas compartidas — evita duplicarlas — costo nulo.
Task 5: complete (tests: pytest tests/test_clean.py -> 9 passed)
Task 6: complete (tests: pytest tests/test_backend.py -> 15 passed; solo numpy probado aquí: numba/mlx/cupy no están instalados, su lógica se validó con los kernels en Python puro)
Task 6: Ruling: kernels de Numba en módulo propio src/compute/numba_kernels.py (cache=True exige funciones a nivel de módulo) y el backend base usa un `xp` genérico para numpy/mlx/cupy — evita tres implementaciones — si falla: mlx/cupy sin probar aquí; el respaldo automático evita que rompan el programa.
Task 7: complete (tests: pytest tests/test_mpi_run.py -> 10 passed; MPI real no probado aquí: se simuló con un comunicador falso y se probó el respaldo multiprocessing)
Task 8: complete (tests: pytest tests/test_analysis.py -> 11 passed; suite 81 passed; app.py solo compilado: streamlit no está instalado aquí)
Task 9: complete (notebook JSON y sintaxis de sus celdas validadas; README escrito)
Final: Ruling: run_ingest acumula historial sin duplicar (no estaba en el plan) — con APIs reales cada corrida solo trae la foto del momento y sin historial no hay correlaciones — si falla: tablas crecen sin límite (volumen pequeño).
Final: Ruling: app.py arma la tabla por hora si aún no se procesó con MPI — evita pestañas vacías — costo: se calcula dos veces si luego se procesa.
Final: self-review (la revisión la hizo el mismo autor, sin revisor independiente); suite 82/82; corrida de punta a punta en modo demo OK (ingesta + mpi_run con respaldo multiprocessing/numpy).

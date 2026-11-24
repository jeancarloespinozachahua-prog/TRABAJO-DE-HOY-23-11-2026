"""Genera los datasets sintéticos en la carpeta data/."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

# ---- 1. Satisfacción del cliente (80 registros) ----
rng = np.random.default_rng(42)
n = 80
d1 = pd.DataFrame({
    "valoracion": rng.integers(1, 6, n),
    "tiempo_espera": rng.integers(1, 46, n),
    "problemas": rng.integers(0, 5, n),
})
s = d1.valoracion - 0.06 * d1.tiempo_espera - 0.9 * d1.problemas + rng.normal(0, 0.3, n)
d1["satisfecho"] = (s > s.quantile(0.5)).astype(int)
d1.to_csv(DATA / "satisfaccion.csv", index=False)

# ---- 2. Inversión en software (100 registros) ----
rng = np.random.default_rng(7)
n = 100
d2 = pd.DataFrame({
    "presupuesto_ti": rng.integers(0, 81, n),
    "procesos_manuales": rng.integers(10, 101, n),
    "empleados": rng.integers(1, 151, n),
    "digitalizacion": rng.integers(1, 11, n),
})
s = (0.05 * d2.presupuesto_ti + 0.03 * d2.procesos_manuales
     + 0.01 * d2.empleados + 0.25 * d2.digitalizacion + rng.normal(0, 0.4, n))
d2["invierte"] = (s > s.quantile(0.5)).astype(int)
d2.to_csv(DATA / "inversion.csv", index=False)

print("Datasets generados en", DATA)

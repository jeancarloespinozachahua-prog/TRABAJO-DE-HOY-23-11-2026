"""SENATI / KNN Lab - Laboratorio de predicción inteligente (Flask + scikit-learn)."""
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
K = 3

PROYECTOS = {
    "satisfaccion": {
        "titulo": "Satisfacción del cliente",
        "archivo": "satisfaccion.csv",
        "objetivo": "satisfecho",
        "etiquetas": {0: "No", 1: "Sí"},
        "mensajes": {0: "Cliente insatisfecho", 1: "Cliente satisfecho"},
        "campos": [
            {"id": "valoracion", "label": "Valoración (1-5)", "min": 1, "max": 5, "default": 5},
            {"id": "tiempo_espera", "label": "Tiempo de espera (mins)", "min": 0, "max": 120, "default": 12},
            {"id": "problemas", "label": "Problemas reportados", "min": 0, "max": 10, "default": 0},
        ],
    },
    "inversion": {
        "titulo": "Inversión en software",
        "archivo": "inversion.csv",
        "objetivo": "invierte",
        "etiquetas": {0: "No", 1: "Sí"},
        "mensajes": {0: "El empresario NO invertirá", 1: "El empresario SÍ invertirá"},
        "campos": [
            {"id": "presupuesto_ti", "label": "Presupuesto TI (miles S/)", "min": 0, "max": 500, "default": 40},
            {"id": "procesos_manuales", "label": "Procesos manuales (%)", "min": 0, "max": 100, "default": 70},
            {"id": "empleados", "label": "N.º de empleados", "min": 1, "max": 1000, "default": 60},
            {"id": "digitalizacion", "label": "Nivel de digitalización (1-10)", "min": 1, "max": 10, "default": 6},
        ],
    },
}


class Modelo:
    def __init__(self, cfg):
        self.cfg = cfg
        df = pd.read_csv(BASE / "data" / cfg["archivo"])
        self.columnas = [c["id"] for c in cfg["campos"]]
        X, y = df[self.columnas], df[cfg["objetivo"]]
        self.n = len(df)
        X_tr, X_te, self.y_train, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)
        self.scaler = StandardScaler().fit(X_tr)
        self.X_train_s = self.scaler.transform(X_tr)
        self.knn = KNeighborsClassifier(n_neighbors=K).fit(self.X_train_s, self.y_train)
        pred = self.knn.predict(self.scaler.transform(X_te))
        self.exactitud = accuracy_score(y_te, pred)
        self.matriz = confusion_matrix(y_te, pred, labels=[0, 1]).tolist()
        self.n_test = len(X_te)
        self.pca = PCA(n_components=2).fit(self.X_train_s)

    def info(self):
        return {"titulo": self.cfg["titulo"], "exactitud": round(self.exactitud * 100, 1),
                "registros": self.n, "k": K, "n_test": self.n_test,
                "matriz": self.matriz, "campos": self.cfg["campos"],
                "etiquetas": self.cfg["etiquetas"]}

    def predecir(self, valores):
        fila = pd.DataFrame([valores], columns=self.columnas)
        fs = self.scaler.transform(fila)
        clase = int(self.knn.predict(fs)[0])
        dist, idx = self.knn.kneighbors(fs)
        vecinos = [{"n": i + 1,
                    "clase": int(self.y_train.iloc[j]),
                    "etiqueta": self.cfg["etiquetas"][int(self.y_train.iloc[j])],
                    "distancia": round(float(d), 3)}
                   for i, (d, j) in enumerate(zip(dist[0], idx[0]))]
        votos_si = sum(v["clase"] for v in vecinos)
        return {"clase": clase, "etiqueta": self.cfg["etiquetas"][clase],
                "mensaje": self.cfg["mensajes"][clase], "vecinos": vecinos,
                "votos_si": votos_si, "votos_no": K - votos_si,
                "grafico": self.grafico(fs, idx[0])}

    def grafico(self, fs, idx):
        p = self.pca.transform(self.X_train_s)
        q = self.pca.transform(fs)
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for c, color in ((0, "#e57373"), (1, "#4a7bd0")):
            m = (self.y_train.values == c)
            ax.scatter(p[m, 0], p[m, 1], s=28, alpha=.55, color=color,
                       label=self.cfg["etiquetas"][c])
        ax.scatter(p[idx, 0], p[idx, 1], s=130, facecolors="none",
                   edgecolors="black", linewidths=1.4, label=f"{K} vecinos")
        ax.scatter(q[0, 0], q[0, 1], marker="*", s=300, color="gold",
                   edgecolors="black", label="Nuevo caso")
        ax.set_title("Espacio de datos (PCA 2D)", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=8); ax.grid(alpha=.25)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


MODELOS = {k: Modelo(v) for k, v in PROYECTOS.items()}
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/modelo/<pid>")
def modelo(pid):
    if pid not in MODELOS:
        return jsonify(error="Proyecto no encontrado"), 404
    return jsonify(MODELOS[pid].info())


@app.route("/api/predecir/<pid>", methods=["POST"])
def predecir(pid):
    if pid not in MODELOS:
        return jsonify(error="Proyecto no encontrado"), 404
    m = MODELOS[pid]
    datos = request.get_json(force=True)
    try:
        valores = [float(datos[c]) for c in m.columnas]
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Completa todos los campos con números válidos"), 400
    return jsonify(m.predecir(valores))


if __name__ == "__main__":
    print("\n  KNN Lab listo ->  http://localhost:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)

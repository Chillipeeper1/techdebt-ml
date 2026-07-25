import csv
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from git import Repo
from sentence_transformers import SentenceTransformer

from feature_lib import (
    features_radon,
    features_ats,
    feature_git_raw,
    embedding_extract_functions,
    normalization_relative_to_repo,
)


MODO_PRUEBA = False
LIMITE_REPOS = 2
LIMITE_ARCHIVOS_POR_REPO = 10

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

RUTA_CANDIDATOS = RAIZ_PROYECTO / "data" / "raw" / "candidate_repos.csv"
RUTA_REPOS_CLONADOS = RAIZ_PROYECTO / "cloned_repos"
RUTA_SALIDA = RAIZ_PROYECTO / "data" / "proccessed"
RUTA_CSV_SALIDA = RUTA_SALIDA / "feature_extraction_V4.csv"
RUTA_EMBEDDINGS_SALIDA = RUTA_SALIDA / "embeddings_por_archivo.npz"

SKIP_DIRS = ["test", ".venv"]


RUTA_REPOS_CLONADOS.mkdir(exist_ok=True)
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

print("Cargando modelo de embeddings (una sola vez)...")
modelo_embed = SentenceTransformer("all-MiniLM-L6-v2")

feat_list = {}
todas_las_funciones = []
indice_por_archivo = []


with open(RUTA_CANDIDATOS, mode="r", encoding="utf-8") as csv_file:
    read_csv = csv.DictReader(csv_file)

    filas = itertools.islice(read_csv, LIMITE_REPOS) if MODO_PRUEBA else read_csv

    for row in filas:
        repo_url = row["clone_url"]
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        dest_route = RUTA_REPOS_CLONADOS / repo_name

        if not dest_route.exists():
            repo_git = Repo.clone_from(repo_url, dest_route)
        else:
            repo_git = Repo(dest_route)

        files = [
            str(f.relative_to(dest_route)) for f in dest_route.rglob("*.py")
            if not any(part in SKIP_DIRS for part in f.parts)
        ]
        if MODO_PRUEBA:
            files = files[:LIMITE_ARCHIVOS_POR_REPO]

        print(f"  {repo_name}: {len(files)} archivos" + (" (modo prueba)" if MODO_PRUEBA else ""))

        for file in files:
            try:
                with open(dest_route / file, "r", encoding="utf-8") as f:
                    feat_file = f.read()

                    features = {}
                    features.update(features_radon(feat_file))
                    features.update(features_ats(feat_file))
                    features.update(feature_git_raw(file, repo_git))
                    features["repo"] = repo_name

                    clave = f"{repo_name}/{file}"

                    funciones_archivo = embedding_extract_functions(feat_file)
                    indice_por_archivo.append((clave, len(funciones_archivo)))
                    todas_las_funciones.extend(funciones_archivo)

                    feat_list[clave] = features

            except Exception as e:
                print(f"    corrupted file: {file}, {e}")
                continue

print(f"\nTotal archivos procesados: {len(feat_list)}")
print(f"Total funciones recolectadas para embeddings: {len(todas_las_funciones)}")


print("\nGenerando embeddings...")
inicio = time.time()

if todas_las_funciones:
    embeddings_funciones = modelo_embed.encode(
        todas_las_funciones,
        batch_size=64,
        show_progress_bar=True,
    )
else:
    embeddings_funciones = np.empty((0, 384))

duracion = time.time() - inicio
print(f"{len(todas_las_funciones)} funciones embebidas en {duracion:.1f}s")

if indice_por_archivo:
    estimado_total = duracion * (2503 / len(indice_por_archivo))
    print(f"Estimado para ~2503 archivos: {estimado_total:.1f}s (~{estimado_total / 60:.1f} min)")


total_funciones_esperadas = sum(cantidad for _, cantidad in indice_por_archivo)
assert total_funciones_esperadas == len(todas_las_funciones), (
    f"Descuadre: se esperaban {total_funciones_esperadas} funciones, "
    f"pero se generaron embeddings para {len(todas_las_funciones)}"
)
print("Verificación de conteo: OK")



embeddings_por_archivo = {}
cursor = 0
for clave, cantidad in indice_por_archivo:
    if cantidad == 0:
        embeddings_por_archivo[clave] = np.zeros(384)
        continue
    bloque = embeddings_funciones[cursor: cursor + cantidad]
    embeddings_por_archivo[clave] = bloque.mean(axis=0)
    cursor += cantidad

print("\nMuestra de embeddings por archivo:")
for nombre, vector in list(embeddings_por_archivo.items())[:3]:
    print(f"  {nombre}: shape={vector.shape}, tiene NaN={np.isnan(vector).any()}")


np.savez(RUTA_EMBEDDINGS_SALIDA, **embeddings_por_archivo)
print(f"\nEmbeddings guardados en: {RUTA_EMBEDDINGS_SALIDA}")


df = pd.DataFrame.from_dict(feat_list, orient="index")
df.index.name = "file name"
df = normalization_relative_to_repo(df, repo_col="repo")
df.to_csv(RUTA_CSV_SALIDA)
print(f"Features guardadas en: {RUTA_CSV_SALIDA}")

if MODO_PRUEBA:
    print("\n*** MODO_PRUEBA está activo — esto NO es el dataset completo. ***")
    print("*** Cambia MODO_PRUEBA a False para procesar los 22 repos completos. ***")
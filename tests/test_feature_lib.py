import pandas as pd
import pytest
from radon.complexity import cc_visit
from git import Repo, Actor

from src.feature_lib import (
    features_radon,
    features_ats,
    feature_git_raw,
    normalization_relative_to_repo,
)


def test_blank_ratio_no_depende_de_tener_comentarios():
    codigo = "x = 1\n\n\ny = 2\n"  # sin comentarios, con líneas en blanco

    resultado = features_radon(codigo)

    assert resultado["comment_ratio"] == 0  # correcto, no hay comentarios
    assert resultado["blank_ratio"] > 0     # el bug original daba 0 aquí


def test_comment_ratio_cuenta_comentarios_reales():
    codigo = "# comentario 1\n# comentario 2\nx = 1\ny = 2\n"

    resultado = features_radon(codigo)

    assert resultado["comment_ratio"] > 0


def test_max_complexity_es_el_maximo_no_la_suma():
    codigo = """
def simple():
    return 1

def con_condicionales(x):
    if x > 0:
        if x > 10:
            return 1
    return 0

def otra_simple():
    return 2
"""
    complejidades_reales = [f.complexity for f in cc_visit(codigo)]
    maximo_real = max(complejidades_reales)
    suma_total = sum(complejidades_reales)

    resultado = features_radon(codigo)

    assert resultado["max_complexity"] == maximo_real
    assert resultado["max_complexity"] != suma_total or maximo_real == suma_total


def test_archivo_vacio_no_truena():
    codigo = "x = 1\n"

    resultado = features_radon(codigo)

    assert resultado["num_functions"] == 0
    assert resultado["max_complexity"] == 0
    assert resultado["avg_function_len"] == 0


def test_dependency_count_cuenta_imports():
    codigo = "import os\nimport sys\nfrom pathlib import Path\n\nx = 1\n"

    resultado = features_ats(codigo)

    assert resultado["dependency_count"] == 3


def test_max_nesting_depth_detecta_anidamiento_real():
    codigo = """
def funcion():
    if True:
        for i in range(10):
            if i > 5:
                pass
"""
    resultado = features_ats(codigo)

    # función(1) -> if(2) -> for(3) -> if(4)
    assert resultado["max_nesting_depth"] == 4


def test_is_tested_detecta_funciones_de_test():
    codigo_con_test = "def test_algo():\n    assert True\n"
    codigo_sin_test = "def funcion_normal():\n    return 1\n"

    assert features_ats(codigo_con_test)["is_tested"] == 1
    assert features_ats(codigo_sin_test)["is_tested"] == 0


@pytest.fixture
def repo_de_prueba(tmp_path):
    repo = Repo.init(tmp_path)

    archivo = tmp_path / "ejemplo.py"
    archivo.write_text("x = 1\n")
    repo.index.add(["ejemplo.py"])
    repo.index.commit(
        "primer commit",
        author=Actor("Autor Uno", "uno@ejemplo.com"),
    )

    archivo.write_text("x = 1\ny = 2\n")
    repo.index.add(["ejemplo.py"])
    repo.index.commit(
        "segundo commit",
        author=Actor("Autor Dos", "dos@ejemplo.com"),
    )

    return repo

def test_feature_git_raw_cuenta_commits_y_autores(repo_de_prueba):
    resultado = feature_git_raw("ejemplo.py", repo_de_prueba)

    assert resultado["commit_count"] == 2
    assert resultado["author_count"] == 2
    assert resultado["churn_rate"] > 0
    assert resultado["days_since_change"] >= 0


def test_feature_git_raw_archivo_nunca_commiteado(repo_de_prueba):
    resultado = feature_git_raw("no_existe.py", repo_de_prueba)

    assert resultado["commit_count"] == 0
    assert resultado["days_since_change"] == 0


def test_normalizacion_relativa_calcula_ratio_correcto():
    df = pd.DataFrame({
        "repo": ["a", "a", "a", "b", "b"],
        "commit_count": [10, 20, 30, 5, 15],
        "author_count": [1, 1, 1, 1, 1],
        "churn_rate": [1, 1, 1, 1, 1],
        "days_since_change": [1, 1, 1, 1, 1],
    })

    resultado = normalization_relative_to_repo(df.copy(), repo_col="repo")

    assert resultado.loc[0, "commit_count"] == pytest.approx(0.5)
    assert resultado.loc[1, "commit_count"] == pytest.approx(1.0)
    assert resultado.loc[2, "commit_count"] == pytest.approx(1.5)


def test_normalizacion_relativa_no_divide_entre_cero():
    df = pd.DataFrame({
        "repo": ["a", "a"],
        "commit_count": [0, 0],
        "author_count": [1, 1],
        "churn_rate": [0, 0],
        "days_since_change": [1, 1],
    })

    resultado = normalization_relative_to_repo(df.copy(), repo_col="repo")

    assert not resultado["commit_count"].isin([float("inf")]).any()
    assert resultado["commit_count"].notna().all()
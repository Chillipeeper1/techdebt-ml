"""
TechDebt ML — Buscador de repositorios
=======================================
Busca repos de Python en GitHub, filtra los que tienen
suficientes bugs documentados, y guarda el resultado en CSV.

Uso:
    pip install requests python-dotenv
    python buscar_repos.py

Requisitos:
    Archivo .env con: GITHUB_TOKEN=ghp_tuTokenAqui
"""

import os
import time
import csv
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")

if not TOKEN:
    raise EnvironmentError(
        "No se encontró GITHUB_TOKEN.\n"
        "Crea un archivo .env con: GITHUB_TOKEN=ghp_tuTokenAqui"
    )

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ─── Configuración ────────────────────────────────────────────────────────────

CONFIG = {
    "min_stars": 5,
    "max_stars": 500,
    "max_size_kb": 8000,       # repos pequeños (~3 MB máximo)
    "min_bugs_cerrados": 5,   # mínimo de issues con label "bug" cerrados
    "paginas": 10,              # páginas de búsqueda (30 repos por página)
    "pausa_entre_requests": 1, # segundos entre requests (respeta rate limit)
    "archivo_salida": "repos_candidatos.csv"
}

# ─── Funciones ────────────────────────────────────────────────────────────────

def get(url, params=None):
    """Request GET con manejo de rate limit."""
    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 403:
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        espera = max(0, reset_time - int(time.time())) + 5
        print(f"  Rate limit alcanzado. Esperando {espera}s...")
        time.sleep(espera)
        response = requests.get(url, headers=HEADERS, params=params)

    return response


def buscar_repos():
    """Busca repos de Python con los filtros de CONFIG."""
    print("Buscando repositorios en GitHub...")
    repos = []

    for pagina in range(1, CONFIG["paginas"] + 1):
        print(f"  Página {pagina}/{CONFIG['paginas']}...", end=" ")

        params = {
            "q": (
                f"language:python "
                f"stars:{CONFIG['min_stars']}..{CONFIG['max_stars']} "
                f"size:<{CONFIG['max_size_kb']} "
                f"is:public"
            ),
            "sort": "updated",
            "per_page": 30,
            "page": pagina
        }

        response = get("https://api.github.com/search/repositories", params)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json().get('message')}")
            break

        items = response.json().get("items", [])
        repos.extend(items)
        print(f"{len(items)} repos encontrados")

        time.sleep(CONFIG["pausa_entre_requests"])

    print(f"\nTotal encontrados: {len(repos)} repos\n")
    return repos


def contar_bugs_cerrados(repo_fullname):
    """Cuenta issues cerrados con label 'bug' (excluye PRs)."""
    params = {
        "state": "closed",
        "labels": "bug",
        "per_page": 100
    }

    response = get(
        f"https://api.github.com/repos/{repo_fullname}/issues",
        params
    )

    if response.status_code != 200:
        return 0

    issues = response.json()
    # GitHub mezcla issues y PRs — filtramos solo issues reales
    solo_issues = [i for i in issues if "pull_request" not in i]
    return len(solo_issues)


def obtener_detalle_repo(repo):
    """Extrae los campos relevantes de un repo para el CSV."""
    return {
        "nombre": repo["full_name"],
        "descripcion": (repo.get("description") or "")[:100],
        "estrellas": repo["stargazers_count"],
        "forks": repo["forks_count"],
        "issues_abiertos": repo["open_issues_count"],
        "tamaño_kb": repo["size"],
        "creado": repo["created_at"][:10],
        "actualizado": repo["updated_at"][:10],
        "url": repo["html_url"],
        "clone_url": repo["clone_url"],
        "bugs_cerrados": 0  # se llena en el filtrado
    }


def filtrar_por_bugs(repos_raw):
    """Revisa bugs cerrados de cada repo y filtra los que cumplen el mínimo."""
    print("Verificando bugs cerrados por repo...")
    candidatos = []
    total = len(repos_raw)

    for i, repo in enumerate(repos_raw):
        nombre = repo["full_name"]
        print(f"  [{i+1}/{total}] {nombre}...", end=" ")

        n_bugs = contar_bugs_cerrados(nombre)
        print(f"{n_bugs} bugs cerrados")

        if n_bugs >= CONFIG["min_bugs_cerrados"]:
            detalle = obtener_detalle_repo(repo)
            detalle["bugs_cerrados"] = n_bugs
            candidatos.append(detalle)

        time.sleep(CONFIG["pausa_entre_requests"])

    return candidatos


def guardar_csv(candidatos):
    """Guarda los repos candidatos en un CSV ordenado por bugs."""
    if not candidatos:
        print("\nNo se encontraron repos con suficientes bugs.")
        return

    candidatos_ordenados = sorted(
        candidatos,
        key=lambda r: r["bugs_cerrados"],
        reverse=True
    )

    archivo = CONFIG["archivo_salida"]
    columnas = [
        "nombre", "bugs_cerrados", "estrellas", "forks",
        "issues_abiertos", "tamaño_kb", "creado",
        "actualizado", "descripcion", "url", "clone_url"
    ]

    with open(archivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(candidatos_ordenados)

    print(f"\nGuardado en: {archivo}")
    return archivo


def imprimir_resumen(candidatos):
    """Muestra un resumen de los repos encontrados."""
    if not candidatos:
        return

    print("\n" + "="*60)
    print(f"  REPOS CANDIDATOS PARA TECHDEBT ML")
    print("="*60)
    print(f"  Total: {len(candidatos)} repos")
    print(f"  Criterio: >= {CONFIG['min_bugs_cerrados']} bugs cerrados")
    print("="*60)

    ordenados = sorted(candidatos, key=lambda r: r["bugs_cerrados"], reverse=True)

    for repo in ordenados[:15]:  # muestra los 15 mejores
        print(
            f"  {repo['bugs_cerrados']:>3} bugs | "
            f"{repo['estrellas']:>3} estrellas | "
            f"{repo['nombre']}"
        )

    if len(ordenados) > 15:
        print(f"  ... y {len(ordenados) - 15} más en el CSV")

    print("="*60)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\nTechDebt ML — Buscador de repositorios")
    print(f"Inicio: {datetime.now().strftime('%H:%M:%S')}\n")

    repos_raw = buscar_repos()

    if not repos_raw:
        print("No se encontraron repos. Revisa tu token o los filtros.")
        return

    candidatos = filtrar_por_bugs(repos_raw)
    guardar_csv(candidatos)
    imprimir_resumen(candidatos)

    print(f"\nFin: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Siguiente paso: clonar los repos del CSV y extraer features con Radon + GitPython\n")


if __name__ == "__main__":
    main()
"""Script do benchmark sintético: proveniência, recusa de sobrescrita, determinismo."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_scenarios.py"
# Um cenário que a main acerta e um que ela erra: resultado ruim não muda o
# código de saída.
SUBCONJUNTO = ["--fps", "3", "--scenarios", "stranger_takes", "owner_pulls_neighbour_closer"]


def roda(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_json_traz_proveniencia_alcance_e_resultados(tmp_path):
    saida = tmp_path / "res" / "cenarios.json"

    execucao = roda("--output", str(saida), *SUBCONJUNTO)

    assert execucao.returncode == 0, execucao.stderr
    doc = json.loads(saida.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert doc["modo"] == "sintetico: posições geradas, sem detector"
    assert "não demonstra detector" in doc["alcance"]
    assert len(doc["commit"]["sha"]) == 40
    assert isinstance(doc["commit"]["sujo"], bool)
    assert len(doc["config_sha256"]) == 64
    assert doc["fontes_sha256"] and all(doc["fontes_sha256"].values())
    assert doc["fps"] == [3.0]
    assert doc["seeds"] == []
    assert doc["ruido"] is None
    assert doc["execucoes"] == {"cenarios": 2, "por_sistema": 2, "total": 4}
    assert [c["name"] for c in doc["cenarios"]] == [
        "stranger_takes",
        "owner_pulls_neighbour_closer",
    ]
    furto = doc["cenarios"][0]
    assert furto["premissas_validas"] is True
    assert furto["execucoes"][0]["sistema"]["veredito"] == "correto"
    assert set(doc["agregados"]["limpo"]) == {"P3", "base", "total"}
    assert "MODO: sintetico" in execucao.stdout
    assert "não demonstra detector" in execucao.stdout


def test_recusa_sobrescrever_resultado_existente(tmp_path):
    saida = tmp_path / "cenarios.json"
    saida.write_text("anterior", encoding="utf-8")

    execucao = roda("--output", str(saida), *SUBCONJUNTO)

    assert execucao.returncode != 0
    assert "já existe" in execucao.stderr
    assert saida.read_text(encoding="utf-8") == "anterior"


def test_execucao_sem_ruido_e_deterministica(tmp_path):
    primeira, segunda = tmp_path / "a.json", tmp_path / "b.json"

    for saida in (primeira, segunda):
        assert roda("--output", str(saida), *SUBCONJUNTO).returncode == 0

    a = json.loads(primeira.read_text(encoding="utf-8"))
    b = json.loads(segunda.read_text(encoding="utf-8"))
    assert a["cenarios"] == b["cenarios"]
    assert a["agregados"] == b["agregados"]


def test_ruido_usa_as_seeds_pedidas(tmp_path):
    saida = tmp_path / "ruido.json"

    execucao = roda(
        "--output",
        str(saida),
        "--fps",
        "3",
        "--scenarios",
        "stranger_takes",
        "--noise",
        "--seeds",
        "7",
    )

    assert execucao.returncode == 0, execucao.stderr
    doc = json.loads(saida.read_text(encoding="utf-8"))
    assert doc["seeds"] == [7]
    assert doc["ruido"]["pessoa"]["id_switch_rate"] == 0.0
    assert [r["seed"] for r in doc["cenarios"][0]["execucoes"]] == [None, 7]
    assert doc["execucoes"]["por_sistema"] == 2
    assert doc["agregados"]["ruido"]["total"]["sistema"]["execucoes"] == 1


def test_cenario_desconhecido_e_recusado(tmp_path):
    saida = tmp_path / "x.json"

    execucao = roda("--output", str(saida), "--scenarios", "nao_existe")

    assert execucao.returncode != 0
    assert not saida.exists()

"""O desfecho de custódia não pode depender da taxa de quadros da fonte.

Auditoria de 2026-11-24: `min_samples` conta QUADROS. A 25 fps dez amostras
levam 0,36 s; a 3 fps levam 3 s. Enquanto junta amostras, o track novo fica
fora de `people`, e se a confirmação de carregamento (`carry_confirm_s`, 1 s)
vence antes disso, a bagagem parece andar sozinha: sai `BAG_AMBIGUOUS` em vez
de `BAG_REMOVED_BY_STRANGER`, e o furto some da fila.

Limiar temporal vai em segundos, como limiar espacial vai em metros.
"""

import json

import pytest

from custody_watch.config import SAFE_BOUNDS, Config, ReidConfig, UnsafeValueError, load_config
from custody_watch.events import EventKind
from custody_watch.ground_plane import GroundPlane
from custody_watch.orchestrator import LiveSession
from custody_watch.reid import SIGNATURE_SIZE, Appearance, TrackLinker
from custody_watch.tracking import TrackedDetection
from custody_watch.types import Point

APARENCIA = Appearance(tuple([1.0 / SIGNATURE_SIZE] * SIGNATURE_SIZE))
LADRAO = 7
MALA = 100

TOLERANCIA_S = 0.5
"""Um intervalo entre quadros da fonte mais lenta testada (2 fps). O instante
de um evento só pode ser decidido num quadro, então fontes diferentes
discordam por até um intervalo -- mais que isso já é o sistema mudando de
comportamento, não de amostragem."""

TERMINAIS = {
    EventKind.BAG_AMBIGUOUS,
    EventKind.BAG_REMOVED_BY_STRANGER,
}


def _pessoa(tid: int, fx: int, fy: int) -> TrackedDetection:
    return TrackedDetection(tid, "person", (fx - 20, fy - 170, fx + 20, fy), APARENCIA)


def _mala(tid: int, fx: int, fy: int) -> TrackedDetection:
    return TrackedDetection(tid, "suitcase", (fx - 20, fy - 45, fx + 20, fy))


def _furto_simples(fps: float) -> LiveSession:
    """Mala órfã parada por 20 s; o ladrão entra colado nela e sai andando
    com ela a 1 m/s. Mesma cena do `repro_fps.py` da auditoria."""
    sessao = LiveSession(GroundPlane.uniform(0.01))
    for i in range(int(40 * fps)):
        t = i / fps
        if t < 20:
            quadro = [_mala(MALA, 550, 500)]
        else:
            dx = int((t - 20) * 100)
            quadro = [_pessoa(LADRAO, 600 + dx, 500), _mala(MALA, 570 + dx, 500)]
        sessao.feed(t, quadro)
    return sessao


def _terminais(sessao: LiveSession) -> list[tuple[EventKind, float, int | None]]:
    return [(e.kind, e.t_start, e.subject) for e in sessao.events if e.kind in TERMINAIS]


# --- cena completa ------------------------------------------------------------


def test_a_25_fps_o_desfecho_e_o_de_antes_da_correcao():
    """Regressão: valores observados no main antes da correção, a 25 fps com
    a config default. Os números publicados dependem deste caminho."""
    sessao = _furto_simples(25)

    assert _terminais(sessao) == [(EventKind.BAG_REMOVED_BY_STRANGER, 21.32, LADRAO)]
    assert [(q.person, q.top_level.name) for q in sessao.queue()] == [(LADRAO, "N3")]


@pytest.mark.parametrize("fps", [2, 3, 5, 10])
def test_mesma_cena_mesmo_desfecho_em_qualquer_fps(fps):
    """A 3 fps o ladrão ficava 3 s fora de `people` e a fila saía vazia."""
    referencia = _terminais(_furto_simples(25))
    sessao = _furto_simples(fps)
    obtido = _terminais(sessao)

    assert [(tipo, sujeito) for tipo, _, sujeito in obtido] == [
        (tipo, sujeito) for tipo, _, sujeito in referencia
    ]
    assert obtido[0][1] == pytest.approx(referencia[0][1], abs=TOLERANCIA_S)
    assert [(q.person, q.top_level.name) for q in sessao.queue()] == [(LADRAO, "N3")]


# --- TrackLinker ----------------------------------------------------------------


def _instante_de_liberacao(linker: TrackLinker, fps: float, t0: float = 20.0) -> float:
    for i in range(int(10 * fps)):
        t = t0 + i / fps
        if linker.observe(1, t, Point(0.0, 0.0), APARENCIA).settled:
            return t
    raise AssertionError(f"track nunca liberado a {fps} fps")


@pytest.mark.parametrize("fps", [2, 3, 5, 10, 25])
def test_track_novo_entra_em_people_antes_de_confirmar_o_carregamento(fps):
    """A latência até o track valer em `people` é limitada em segundos.

    O contrato que importa: quem carrega precisa existir antes de
    `carry_confirm_s` vencer, senão a bagagem anda 'sozinha' e o furto vira
    ambiguidade. Isso não pode depender de quantos quadros cabem nesse tempo.
    """
    config = Config()
    linker = TrackLinker(config.reid)

    latencia = _instante_de_liberacao(linker, fps) - 20.0

    assert latencia < config.custody.carry_confirm_s
    assert latencia <= config.reid.max_settle_s + 1.0 / fps


def test_prazo_libera_com_as_amostras_que_houver_e_ainda_religa():
    """Liberar no prazo não pula as guardas: o perfil curto passa pelas mesmas
    três, e a religação acontece se a evidência for clara."""
    linker = TrackLinker(ReidConfig(min_samples=10, max_settle_s=0.5))
    for i in range(10):
        linker.observe(1, i * 0.04, Point(0.0, 0.0), APARENCIA)
    assert linker.canonical(1) == 1

    primeiro = linker.observe(9, 5.0, Point(1.0, 0.0), APARENCIA)
    segundo = linker.observe(9, 5.0 + 1 / 3, Point(1.0, 0.0), APARENCIA)
    terceiro = linker.observe(9, 5.0 + 2 / 3, Point(1.0, 0.0), APARENCIA)

    assert primeiro.settled is False
    assert segundo.settled is False
    assert terceiro.settled is True
    assert terceiro.linked_from == 1
    assert linker.links() == {9: 1}


def test_a_25_fps_o_minimo_de_amostras_continua_decidindo():
    """Regressão: com a config default a 25 fps, a liberação acontece na
    décima amostra, exatamente como antes -- o prazo não antecipa nada."""
    linker = TrackLinker(Config().reid)
    t0 = 20.0
    instantes = [t0 + i / 25 for i in range(12)]

    estados = [linker.observe(1, t, Point(0.0, 0.0), APARENCIA).settled for t in instantes]

    assert estados == [False] * 9 + [True] * 3


def test_prazo_nao_libera_antes_de_existir_amostra():
    """O prazo conta a partir da primeira amostra: a primeira observação de
    um track com aparência nunca sai liberada sozinha por prazo."""
    linker = TrackLinker(ReidConfig(min_samples=10, max_settle_s=0.5))

    assert linker.observe(1, 100.0, Point(0.0, 0.0), APARENCIA).settled is False


# --- config -------------------------------------------------------------------


def test_prazo_de_assentamento_curto_demais_e_recusado(tmp_path):
    caminho = tmp_path / "c.json"
    caminho.write_text(json.dumps({"reid": {"max_settle_s": 0.04}}), encoding="utf-8")

    with pytest.raises(UnsafeValueError, match="reid.max_settle_s"):
        load_config(caminho)


def test_prazo_de_assentamento_longo_demais_e_recusado(tmp_path):
    caminho = tmp_path / "c.json"
    caminho.write_text(json.dumps({"reid": {"max_settle_s": 5.0}}), encoding="utf-8")

    with pytest.raises(UnsafeValueError, match="reid.max_settle_s"):
        load_config(caminho)


def test_erro_de_prazo_de_assentamento_nomeia_o_ataque(tmp_path):
    caminho = tmp_path / "c.json"
    caminho.write_text(json.dumps({"reid": {"max_settle_s": 5.0}}), encoding="utf-8")

    with pytest.raises(UnsafeValueError) as erro:
        load_config(caminho)

    assert SAFE_BOUNDS["reid.max_settle_s"].reason in str(erro.value)


def test_prazo_de_assentamento_dentro_da_faixa_carrega(tmp_path):
    caminho = tmp_path / "c.json"
    caminho.write_text(json.dumps({"reid": {"max_settle_s": 0.8}}), encoding="utf-8")

    assert load_config(caminho).reid.max_settle_s == 0.8


def test_prazo_default_cabe_folgado_acima_das_dez_amostras_a_25_fps():
    """Se o prazo default fosse menor que o tempo de `min_samples` a 25 fps,
    a correção mudaria os resultados publicados do CAVIAR."""
    reid = ReidConfig()

    assert reid.max_settle_s > (reid.min_samples - 1) / 25

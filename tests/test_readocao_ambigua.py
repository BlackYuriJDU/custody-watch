"""Readoção ambígua: TODAS as candidatas ficam incertas, não só a primeira.

`adopt_occluded` recusa readotar quando há mais de uma bagagem ocluída no
raio de readoção da observação. A ambiguidade é entre as candidatas, então a
regra P3 vale para cada uma delas. Marcar só a vizinhança da primeira deixa as
outras resolvíveis: cada candidata está a até um raio da observação, logo duas
candidatas podem estar a até dois raios uma da outra -- fora da vizinhança
uma da outra. A que sobrou ocluída estoura o timeout e acusa quem estiver
perto dela, um passante.

Trajetórias sintéticas escritas à mão, como em `test_occlusion.py`: provam
que a lógica decide certo, não que a percepção enxerga. A geometria é derivada
dos raios da config, não fixada, e cada cenário confere a própria premissa.
"""

import math

import numpy as np

from custody_watch.bag_registry import BagRegistry
from custody_watch.config import Config
from custody_watch.events import EventKind, EventLog
from custody_watch.ground_plane import GroundPlane
from custody_watch.orchestrator import LiveSession
from custody_watch.tracking import TrackedDetection
from custody_watch.types import BagState, FlagLevel, Observation, Point

FPS = 25.0
CONFIG = Config()
RAIO_READOCAO = CONFIG.registry.reidentification_radius_m
RAIO_AMBIGUIDADE = CONFIG.registry.ambiguity_radius_m

# Câmera de cima, 1px = 1cm, igual a test_occlusion.py.
PLANO = GroundPlane(np.diag([0.01, 0.01, 1.0]))

CENTRO = Point(10.0, 10.0)
NOVO, PASSANTE = 950, 7

# Bagagens paradas por 3s viram âncora; somem aos 3s; o track novo surge aos
# 4s. A decisão por timeout só sai depois de `max_occlusion_s`, então a cena
# vai além dela com folga.
SOMEM_EM = 3.0
NOVO_EM = 4.0
DURACAO_S = SOMEM_EM + CONFIG.custody.max_occlusion_s + 5.0
# Cenas que leem o estado logo depois da readoção, antes de qualquer timeout.
ANTES_DO_TIMEOUT = NOVO_EM + 1.0


def caixa(x_m: float, y_m: float, altura_px: float):
    """Caixa cujo pé cai em (x_m, y_m)."""
    cx, base = x_m * 100.0, y_m * 100.0
    return (cx - altura_px / 4, base - altura_px, cx + altura_px / 4, base)


def mala(track_id: int, p: Point) -> TrackedDetection:
    return TrackedDetection(track_id, "suitcase", caixa(p.x, p.y, 55))


def pessoa(track_id: int, p: Point) -> TrackedDetection:
    return TrackedDetection(track_id, "person", caixa(p.x, p.y, 170))


def em_volta(n: int, distancia: float) -> list[Point]:
    """`n` pontos a `distancia` de CENTRO, igualmente espaçados em ângulo."""
    return [
        Point(
            CENTRO.x + distancia * math.cos(2 * math.pi * k / n + math.pi / 2),
            CENTRO.y + distancia * math.sin(2 * math.pi * k / n + math.pi / 2),
        )
        for k in range(n)
    ]


def somem_e_volta_uma_no_meio(
    ocluidas: dict[int, Point],
    *,
    visiveis: dict[int, Point] | None = None,
    passante: Point | None = None,
    duracao_s: float = DURACAO_S,
):
    """As bagagens de `ocluidas` ficam paradas e somem juntas aos 3s; aos 4s
    um track novo de bagagem aparece em CENTRO e fica. `visiveis` ficam em
    cena o tempo todo. `passante`, se houver, chega aos 4s e fica parado."""
    visiveis = visiveis or {}
    for i in range(int(duracao_s * FPS)):
        t = i / FPS
        det = [mala(tid, p) for tid, p in visiveis.items()]
        if t < SOMEM_EM:
            det += [mala(tid, p) for tid, p in ocluidas.items()]
        if t >= NOVO_EM:
            det.append(mala(NOVO, CENTRO))
            if passante is not None:
                det.append(pessoa(PASSANTE, passante))
        yield t, det


def roda(frames) -> LiveSession:
    sessao = LiveSession(PLANO, CONFIG)
    for t, det in frames:
        sessao.feed(t, det)
    return sessao


def estado(sessao: LiveSession, track_id: int) -> BagState:
    bag = sessao.bag_for_track(track_id)
    assert bag is not None, f"track {track_id} nunca virou bagagem"
    return bag.state


def ambiguas_por_readocao(sessao: LiveSession) -> list:
    """`BAG_AMBIGUOUS` vindos da vizinhança, não do timeout de oclusão."""
    return [
        e for e in sessao.events.of_kind(EventKind.BAG_AMBIGUOUS) if e.evidence["neighbour_bags"]
    ]


# --- duas candidatas a quase dois raios uma da outra ---------------------------

A, B = 910, 911
# 0,9 raio de cada lado: as duas no raio de readoção do ponto médio, e a 1,8
# raio uma da outra -- fora do raio de ambiguidade com a config padrão.
LADO = 0.9 * RAIO_READOCAO
PONTO_A = Point(CENTRO.x - LADO, CENTRO.y)
PONTO_B = Point(CENTRO.x + LADO, CENTRO.y)
# Passante parado além de B, longe do ponto médio, dentro do raio de busca.
PONTO_PASSANTE = Point(PONTO_B.x + 0.7, CENTRO.y)


def test_premissa_da_cena_de_duas_candidatas():
    """Sem esta premissa o cenário não exercita o defeito: se B cair na
    vizinhança de A, o método antigo já marcava as duas."""
    assert PONTO_A.distance_to(CENTRO) <= RAIO_READOCAO
    assert PONTO_B.distance_to(CENTRO) <= RAIO_READOCAO
    assert PONTO_A.distance_to(PONTO_B) > RAIO_AMBIGUIDADE
    assert PONTO_PASSANTE.distance_to(PONTO_B) <= CONFIG.pipeline.owner_search_radius_m


def test_duas_candidatas_distantes_ficam_ambas_ambiguas():
    """A observação no meio era candidata às duas, então as duas são incertas.
    Antes, só a primeira virava AMBIGUA.

    O estado é lido logo depois da readoção recusada, bem antes do timeout:
    sem passante, a candidata esquecida chegaria a AMBIGUA de qualquer jeito
    pelo timeout sem candidatos, e o teste não distinguiria nada."""
    sessao = roda(somem_e_volta_uma_no_meio({A: PONTO_A, B: PONTO_B}, duracao_s=ANTES_DO_TIMEOUT))

    assert estado(sessao, A) is BagState.AMBIGUA
    assert estado(sessao, B) is BagState.AMBIGUA
    assert sessao.anchors() == []


def test_candidata_esquecida_nao_vira_furto_contra_passante():
    """O efeito visível do defeito: B seguia ocluída, estourava o timeout com
    o passante ao alcance e saía `bag_removed_by_stranger` contra ele, com N3
    na fila. A bagagem nem tinha saído do lugar -- o track novo era ela ou A."""
    sessao = roda(somem_e_volta_uma_no_meio({A: PONTO_A, B: PONTO_B}, passante=PONTO_PASSANTE))
    resultado = sessao.result()

    assert resultado.events.of_kind(EventKind.BAG_REMOVED_BY_STRANGER) == []
    assert [
        f for f in resultado.flags.for_person(PASSANTE) if f.kind == "retirada_por_estranho"
    ] == []
    assert [item for item in resultado.queue if item.top_level is FlagLevel.N3] == []


def test_cada_candidata_sai_no_log_uma_vez():
    """Uma ambiguidade por candidata, sem repetir: A e B não são vizinhas,
    então cada uma tem o próprio `BAG_AMBIGUOUS`, e nenhuma aparece em dois."""
    sessao = roda(somem_e_volta_uma_no_meio({A: PONTO_A, B: PONTO_B}))

    eventos = ambiguas_por_readocao(sessao)
    assert sorted(e.bag for e in eventos) == [A, B]
    citadas = [b for e in eventos for b in e.evidence["neighbour_bags"]]
    assert sorted(citadas) == [A, B]


# --- três candidatas ----------------------------------------------------------


def test_tres_candidatas_ficam_todas_ambiguas():
    """Três bagagens em volta do ponto, a 0,9 raio dele e a ~1,56 raio umas
    das outras: nenhuma está na vizinhança de outra."""
    pontos = em_volta(3, 0.9 * RAIO_READOCAO)
    for p in pontos:
        assert p.distance_to(CENTRO) <= RAIO_READOCAO
    for i, p in enumerate(pontos):
        for q in pontos[i + 1 :]:
            assert p.distance_to(q) > RAIO_AMBIGUIDADE

    ids = [920, 921, 922]
    sessao = roda(
        somem_e_volta_uma_no_meio(dict(zip(ids, pontos, strict=True)), duracao_s=ANTES_DO_TIMEOUT)
    )

    assert [estado(sessao, tid) for tid in ids] == [BagState.AMBIGUA] * 3
    assert sorted(e.bag for e in ambiguas_por_readocao(sessao)) == ids


# --- a vizinhança antiga continua marcada --------------------------------------

C, D = 930, 931


def test_vizinhanca_de_cada_candidata_continua_marcada():
    """A correção só amplia. Uma bagagem visível perto de uma candidata era
    marcada junto com ela -- a vizinhança da primeira -- e continua sendo; a
    perto da outra candidata passa a ser também."""
    perto_de_a = Point(PONTO_A.x - 0.5 * RAIO_AMBIGUIDADE, CENTRO.y)
    perto_de_b = Point(PONTO_B.x + 0.5 * RAIO_AMBIGUIDADE, CENTRO.y)
    # Nenhuma das duas é candidata de readoção: fora do raio do ponto médio.
    assert perto_de_a.distance_to(CENTRO) > RAIO_READOCAO
    assert perto_de_b.distance_to(CENTRO) > RAIO_READOCAO

    sessao = roda(
        somem_e_volta_uma_no_meio(
            {A: PONTO_A, B: PONTO_B},
            visiveis={C: perto_de_a, D: perto_de_b},
            duracao_s=ANTES_DO_TIMEOUT,
        )
    )

    assert {tid: estado(sessao, tid) for tid in (A, B, C, D)} == {
        A: BagState.AMBIGUA,
        B: BagState.AMBIGUA,
        C: BagState.AMBIGUA,
        D: BagState.AMBIGUA,
    }


# --- regressão: uma candidata só continua sendo readotada ---------------------


def test_uma_candidata_so_continua_sendo_readotada():
    """Volta a menos de `moved_threshold_m` da âncora: readoção pura. Mais
    longe que isso, com ninguém por perto, `carry_away` trata como bagagem
    que anda sozinha -- outro caminho, fora deste teste."""
    perto = Point(CENTRO.x - 0.5 * CONFIG.registry.moved_threshold_m, CENTRO.y)
    assert perto.distance_to(CENTRO) <= RAIO_READOCAO

    sessao = roda(somem_e_volta_uma_no_meio({A: perto}))

    assert sessao.bag_for_track(NOVO) is sessao.bag_for_track(A)
    assert estado(sessao, A) is not BagState.AMBIGUA
    religacoes = [e for e in sessao.events.of_kind(EventKind.TRACK_RELINKED) if e.bag == A]
    assert len(religacoes) == 1
    assert sessao.events.of_kind(EventKind.BAG_AMBIGUOUS) == []


# --- BagRegistry.mark_ambiguous_candidates ------------------------------------


def obs(track_id: int, x: float, y: float) -> Observation:
    return Observation(track_id=track_id, cls="suitcase", position=Point(x, y), t=0.0)


def test_registro_marca_todas_as_candidatas_e_suas_vizinhas():
    """O conjunto marcado contém o que `mark_ambiguous_neighbours` marcava a
    partir da primeira candidata, e a vizinhança de todas as outras."""
    referencia = BagRegistry()
    registry = BagRegistry()
    for r in (referencia, registry):
        r.observe(obs(1, 5.0, 5.0))  # candidata
        r.observe(obs(2, 4.6, 5.0))  # vizinha da 1
        r.observe(obs(3, 6.8, 5.0))  # candidata, a 1,8m da 1
        r.observe(obs(4, 7.2, 5.0))  # vizinha da 3
        r.observe(obs(5, 20.0, 5.0))  # longe de tudo
    antigo = set(referencia.mark_ambiguous_neighbours(1))

    afetadas = registry.mark_ambiguous_candidates([1, 3])

    assert antigo <= set(afetadas)
    assert sorted(afetadas) == [1, 2, 3, 4]
    assert len(afetadas) == len(set(afetadas))
    assert registry.get(5).state is BagState.NOVA


def test_registro_nao_repete_evento_de_candidata_ja_marcada():
    """Duas candidatas vizinhas: a primeira já marca a segunda, e a segunda
    não ganha um segundo `BAG_AMBIGUOUS` sobre a mesma ambiguidade."""
    registry = BagRegistry()
    registry.observe(obs(1, 5.0, 5.0))
    registry.observe(obs(2, 5.3, 5.0))
    log = EventLog()

    registry.mark_ambiguous_candidates([1, 2], t=4.0, events=log)

    (evento,) = log.of_kind(EventKind.BAG_AMBIGUOUS)
    assert evento.bag == 1
    assert evento.evidence["neighbour_bags"] == [1, 2]
    assert registry.get(2).state is BagState.AMBIGUA

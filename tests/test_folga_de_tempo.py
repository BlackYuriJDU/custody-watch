"""Toda régua em segundos decide no quadro devido, em qualquer taxa e fase.

O PR #126 achou o defeito no repouso da bagagem: instantes vêm de
`quadro / fps`, e `26/3 - 20/3` dá 1,9999999999999991, menos que os 2 s do
limiar. Com `limiar * fps` inteiro, o quadro que cobre o limiar exatamente
podia ser lido como "ainda não" -- a decisão saía um quadro depois -- ou, nas
réguas inclusivas do outro lado ("no máximo X s"), como "já passou", e a
decisão que cabia naquele quadro não saía nunca.

Cada teste daqui calcula à mão, em aritmética exata (`Fraction`), o quadro em
que a régua deve decidir, e roda a cena justamente nas fases em que a
subtração em ponto flutuante cai do lado errado do limiar. A premissa de cada
teste confere que essas fases existem: sem ela, o teste passaria sem exercitar
o arredondamento.
"""

import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from custody_watch.config import Config, ReidConfig
from custody_watch.custody import update_attendance
from custody_watch.events import EventKind
from custody_watch.ground_plane import GroundPlane
from custody_watch.orchestrator import MIN_MISSING_FRAMES, LiveSession
from custody_watch.party import PartyManager
from custody_watch.reid import SIGNATURE_SIZE, Appearance, TrackLinker
from custody_watch.tracking import TrackedDetection
from custody_watch.types import Bag, BagState, Observation, Point

CONFIG = Config()
TAXAS = (3.0, 10.0, 25.0, 29.97, 3.25)
"""As do pedido: três inteiras, a NTSC e uma em que `limiar * fps` raramente
é inteiro."""

CANDIDATAS = 2000
"""Quadros candidatos a início de cronômetro; as fases críticas saem daqui."""
PREPARO = 400
"""Quadros reservados antes do cronômetro para a cena se montar (a mala
assentar, o prazo de posse passar), sem instante negativo."""

# Câmera de cima, 1px = 1cm, igual a test_repouso_fps.py.
PLANO = GroundPlane(np.diag([0.01, 0.01, 1.0]))
MALA = 101
DONO, ESTRANHO = 1, 4
ANCORA = Point(10.0, 10.0)
LUGAR_DO_DONO = Point(ANCORA.x - 0.7, ANCORA.y + 1.0)
"""De lado e um metro abaixo da mala, fora da silhueta, como em
`test_fusao_fps.py`."""


# --- aritmética exata -------------------------------------------------------------


def _exato(x: float) -> Fraction:
    """O número decimal que a config e as taxas escrevem, sem o binário."""
    return Fraction(str(x))


def quadros_ate_cobrir(limiar: float, fps: float) -> int:
    """Menor número de períodos cuja duração exata alcança `limiar`."""
    return math.ceil(_exato(limiar) * _exato(fps))


def quadros_dentro(limiar: float, fps: float) -> int:
    """Maior número de períodos cuja duração exata não passa de `limiar`."""
    return math.floor(_exato(limiar) * _exato(fps))


def fases_que_encurtam(fps: float, quadros: int, limiar: float, fases: range) -> list[int]:
    """Inícios em que `quadros` períodos, medidos como a sessão mede
    (`(s + n) / fps - s / fps`), dão menos que `limiar` em ponto flutuante."""
    return [s for s in fases if (s + quadros) / fps - s / fps < limiar]


def fases_que_alongam(fps: float, quadros: int, limiar: float, fases: range) -> list[int]:
    """Inícios em que `quadros` períodos dão mais que `limiar`."""
    return [s for s in fases if (s + quadros) / fps - s / fps > limiar]


def fase_critica(fps: float, quadros: int, limiar: float, *, encurta: bool, minimo: int = 0) -> int:
    """Um quadro `s >= minimo` em que o arredondamento cai do lado errado,
    quando `quadros` períodos medem exatamente `limiar`; senão, qualquer um
    -- não há arredondamento a exercitar, e a margem é de uma fração de
    período."""
    fases = range(minimo, minimo + CANDIDATAS)
    if quadros != _exato(limiar) * _exato(fps):
        return fases[-1]
    busca = fases_que_encurtam if encurta else fases_que_alongam
    criticas = busca(fps, quadros, limiar, fases)
    assert criticas, f"premissa: nenhuma fase arredonda a {fps} fps; o teste não exercita nada"
    return criticas[0]


def quadro(t: float, fps: float) -> int:
    return round(t * fps)


# --- cena -------------------------------------------------------------------------


def caixa(p: Point, altura_px: float):
    """Caixa cujo pé cai em `p`."""
    cx, base = p.x * 100.0, p.y * 100.0
    return (cx - altura_px / 4, base - altura_px, cx + altura_px / 4, base)


def mala(p: Point = ANCORA) -> TrackedDetection:
    return TrackedDetection(MALA, "suitcase", caixa(p, 55))


def pessoa(track_id: int, p: Point) -> TrackedDetection:
    return TrackedDetection(track_id, "person", caixa(p, 170))


def roda(cena, inicio: int, fim: int, fps: float, config: Config = CONFIG) -> LiveSession:
    """Alimenta `cena(i)` nos quadros `[inicio, fim)`, em `t = i / fps`."""
    sessao = LiveSession(PLANO, config)
    for i in range(inicio, fim):
        sessao.feed(i / fps, cena(i))
    return sessao


def ancora_em(inicio: int, fps: float) -> int:
    """Quadro em que a mala parada desde `inicio` vira âncora (PR #126)."""
    return inicio + quadros_ate_cobrir(CONFIG.custody.rest_confirm_s, fps)


def ocluida_em(sumiu: int, fps: float, config: Config = CONFIG) -> int:
    """Quadro em que a mala que falta desde `sumiu` passa a contar como
    ocluída: as duas réguas de `_ausencia_basta`, a física em segundos e o
    piso em quadros."""
    faltas = max(
        MIN_MISSING_FRAMES,
        quadros_ate_cobrir(config.pipeline.missing_before_occluded_s, fps),
    )
    return sumiu + faltas - 1


def antes_de_ocluir(fps: float, config: Config = CONFIG) -> int:
    """Quadros entre o primeiro da cena e o da oclusão: a mala assenta, fica
    `VISTA_DEPOIS` quadros e some."""
    return ocluida_em(ancora_em(0, fps) + VISTA_DEPOIS, fps, config)


VISTA_DEPOIS = 3


# --- custody.unattended_time_s ----------------------------------------------------


@pytest.mark.parametrize("fps", TAXAS)
def test_desacompanhada_no_quadro_que_cobre_o_prazo(fps):
    """O dono se afasta no quadro `s`: a mala vira DESACOMPANHADA exatamente
    `ceil(25 * fps)` quadros depois, nem antes nem depois."""
    limiar = CONFIG.custody.unattended_time_s
    n = quadros_ate_cobrir(limiar, fps)
    s = fase_critica(fps, n, limiar, encurta=True)

    manager = PartyManager()
    grupo = manager.form_on_arrival([DONO])
    bag = Bag(bag_id=MALA, anchor=Point(0.0, 0.0), owner_party=grupo.party_id)
    longe = Point(10.0, 0.0)

    desacompanhada = None
    for i in range(s, s + n + 3):
        t = i / fps
        dono = Observation(track_id=DONO, cls="person", position=longe, t=t)
        update_attendance(bag, [dono], manager, t, CONFIG.custody)
        if bag.state is BagState.DESACOMPANHADA:
            desacompanhada = i
            break

    assert desacompanhada == s + n


# --- reid.max_settle_s -----------------------------------------------------------

APARENCIA = Appearance(tuple([1.0 / SIGNATURE_SIZE] * SIGNATURE_SIZE))


@pytest.mark.parametrize("fps", TAXAS)
def test_track_novo_assenta_no_quadro_do_prazo_ou_das_amostras(fps):
    """Assenta na `min_samples`-ésima amostra ou no primeiro quadro a
    `max_settle_s` da primeira, o que vier antes."""
    reid = CONFIG.reid
    por_prazo = quadros_ate_cobrir(reid.max_settle_s, fps)
    n = min(reid.min_samples - 1, por_prazo)
    s = fase_critica(fps, n, reid.max_settle_s, encurta=True) if n == por_prazo else 0

    linker = TrackLinker(reid)
    assentou = next(
        i
        for i in range(s, s + n + 3)
        if linker.observe(1, i / fps, Point(0.0, 0.0), APARENCIA).settled
    )

    assert assentou == s + n


# --- reid.max_gap_s --------------------------------------------------------------


def _religa(fps: float, saiu: int, volta: int) -> bool:
    """O track 1 assenta e some no quadro `saiu`; o 9, idêntico e no mesmo
    lugar, aparece no quadro `volta`. Ele é religado ao 1?"""
    reid = replace(ReidConfig(), min_samples=1)
    linker = TrackLinker(reid)
    linker.observe(1, saiu / fps, Point(0.0, 0.0), APARENCIA)
    return linker.observe(9, volta / fps, Point(0.0, 0.0), APARENCIA).linked_from == 1


@pytest.mark.parametrize("fps", TAXAS)
def test_religa_com_lacuna_de_no_maximo_max_gap_s(fps):
    """A guarda aceita lacuna de até `max_gap_s`, inclusive: o quadro mais
    distante dentro dela religa, o seguinte não."""
    limiar = CONFIG.reid.max_gap_s
    n = quadros_dentro(limiar, fps)
    s = fase_critica(fps, n, limiar, encurta=False)

    assert _religa(fps, s, s + n)
    assert not _religa(fps, s, s + n + 1)


# --- pipeline.merge_every_s ------------------------------------------------------


def _instantes_de_fusao(fps: float, inicio: int, config: Config) -> list[int]:
    sessao = LiveSession(PLANO, config)
    fusoes: list[int] = []
    funde = sessao._s.merge_parties

    def registra(people) -> None:
        fusoes.append(quadro(sessao.t, fps))
        funde(people)

    sessao._s.merge_parties = registra
    for i in range(inicio, inicio + round(6 * fps)):
        sessao.feed(i / fps, [])
    return fusoes


@pytest.mark.parametrize("cadencia", [CONFIG.pipeline.merge_every_s, 1.0])
@pytest.mark.parametrize("fps", TAXAS)
def test_fusao_a_cada_quadros_que_cobrem_a_cadencia(fps, cadencia):
    """Com o default (0,98 s, nunca inteiro nas taxas testadas) e com 1 s,
    inteiro a 3, 10 e 25 fps: toda fusão cai `ceil(cadencia * fps)` quadros
    depois da anterior, a primeira contada do primeiro quadro."""
    config = replace(CONFIG, pipeline=replace(CONFIG.pipeline, merge_every_s=cadencia))
    n = quadros_ate_cobrir(cadencia, fps)
    inicio = fase_critica(fps, n, cadencia, encurta=True)

    fusoes = _instantes_de_fusao(fps, inicio, config)

    assert len(fusoes) >= 4
    assert fusoes == [inicio + n * (k + 1) for k in range(len(fusoes))]


# --- pipeline.missing_before_occluded_s e custody.max_occlusion_s ----------------


def _mala_some(fps: float, inicio: int):
    sumiu = ancora_em(inicio, fps) + VISTA_DEPOIS

    def cena(i: int) -> list[TrackedDetection]:
        return [mala()] if i < sumiu else []

    return cena


@pytest.mark.parametrize("fps", TAXAS)
def test_oclusao_sem_volta_resolve_no_quadro_que_cobre_o_prazo(fps):
    """A mala some e ninguém esteve nela: BAG_AMBIGUOUS sai `ceil(30 * fps)`
    quadros depois do quadro em que a oclusão abriu."""
    limiar = CONFIG.custody.max_occlusion_s
    n = quadros_ate_cobrir(limiar, fps)
    ocluiu = fase_critica(fps, n, limiar, encurta=True, minimo=PREPARO)
    inicio = ocluiu - antes_de_ocluir(fps)

    sessao = roda(_mala_some(fps, inicio), inicio, ocluiu + n + 3, fps)

    (ambigua,) = sessao.events.of_kind(EventKind.BAG_AMBIGUOUS)
    assert quadro(ambigua.t_start, fps) == ocluiu
    assert quadro(ambigua.t_end, fps) == ocluiu + n


def _oclusao_relatada(
    fps: float, ocluiu: int, volta: int, config: Config = CONFIG
) -> list[tuple[int, int]]:
    """A mala some, a oclusão abre em `ocluiu` e ela volta no quadro `volta`.
    Os intervalos de BAG_OCCLUDED, em quadros."""
    inicio = ocluiu - antes_de_ocluir(fps, config)
    sumiu = ancora_em(inicio, fps) + VISTA_DEPOIS

    def cena(i: int) -> list[TrackedDetection]:
        return [] if sumiu <= i < volta else [mala()]

    sessao = roda(cena, inicio, volta + 3, fps, config)
    return [
        (quadro(e.t_start, fps), quadro(e.t_end, fps))
        for e in sessao.events.of_kind(EventKind.BAG_OCCLUDED)
    ]


AUSENCIA_INTEIRA_S = 0.2
"""`missing_before_occluded_s` inteiro em quadros a 10 e a 25 fps (2 e 5),
dentro da faixa segura e no limite de cima do intervalo que reproduz os 5
quadros antigos a 25 fps."""


@pytest.mark.parametrize("fps", [10.0, 25.0])
def test_oclusao_abre_no_quadro_que_cobre_a_ausencia_somada(fps):
    """A ausência é somada quadro a quadro, `t - previous_t`, e não medida
    numa subtração só: com 0,2 s, a oclusão abre na `ceil(0,2 * fps)`-ésima
    falta em toda fase -- a 25 fps, a quinta, como o default."""
    config = replace(
        CONFIG, pipeline=replace(CONFIG.pipeline, missing_before_occluded_s=AUSENCIA_INTEIRA_S)
    )
    fora_do_devido = {}
    for ocluiu in range(PREPARO, PREPARO + 100):
        relatada = _oclusao_relatada(fps, ocluiu, ocluiu + round(2 * fps), config)
        if [inicio for inicio, _ in relatada] != [ocluiu]:
            fora_do_devido[ocluiu] = relatada

    assert fora_do_devido == {}


@pytest.mark.parametrize("fps", TAXAS)
def test_oclusao_relatada_a_partir_do_piso(fps):
    """Ocluída por `ceil(1 * fps)` quadros, a mala sai no log; por um a
    menos, não."""
    limiar = CONFIG.custody.min_occlusion_report_s
    n = quadros_ate_cobrir(limiar, fps)
    ocluiu = fase_critica(fps, n, limiar, encurta=True, minimo=PREPARO)

    assert _oclusao_relatada(fps, ocluiu, ocluiu + n) == [(ocluiu, ocluiu + n)]
    assert _oclusao_relatada(fps, ocluiu, ocluiu + n - 1) == []


# --- custody.carry_confirm_s -----------------------------------------------------

LADRAO = 7
SALTO_M = 0.6
"""Primeiro deslocamento: passa `moved_threshold_m` num quadro só."""
VELOCIDADE_MS = 1.0


@pytest.mark.parametrize("fps", TAXAS)
def test_furto_confirmado_no_quadro_que_cobre_o_carregamento(fps):
    """Mala órfã; passado o prazo de posse, um estranho a leva. O primeiro
    quadro fora da âncora é `m`, e o furto sai em `m + ceil(1 * fps)`."""
    limiar = CONFIG.custody.carry_confirm_s
    n = quadros_ate_cobrir(limiar, fps)
    espera = quadros_ate_cobrir(CONFIG.custody.ownership_window_s, fps) + 5
    andou = fase_critica(fps, n, limiar, encurta=True, minimo=PREPARO)
    inicio = andou - ancora_em(0, fps) - espera - 1

    def cena(i: int) -> list[TrackedDetection]:
        if i < andou - 1:
            return [mala()]
        dx = 0.0 if i < andou else SALTO_M + VELOCIDADE_MS * (i - andou) / fps
        p = Point(ANCORA.x + dx, ANCORA.y)
        return [pessoa(LADRAO, Point(p.x + 0.3, p.y)), mala(p)]

    sessao = roda(cena, inicio, andou + n + 3, fps)

    (furto,) = sessao.events.of_kind(EventKind.BAG_REMOVED_BY_STRANGER)
    assert furto.subject == LADRAO
    assert quadro(furto.t_start, fps) == andou + n


# --- custody.ownership_window_s --------------------------------------------------


def _dono_chega(fps: float, ancorou: int, chega: int) -> list[int]:
    """Mala órfã que assenta em `ancorou`; alguém chega ao lado dela em
    `chega`. Quadros de BAG_OWNED."""
    inicio = ancorou - ancora_em(0, fps)

    def cena(i: int) -> list[TrackedDetection]:
        return [mala(), pessoa(DONO, LUGAR_DO_DONO)] if i >= chega else [mala()]

    sessao = roda(cena, inicio, chega + 3, fps)
    (apareceu,) = sessao.events.of_kind(EventKind.BAG_APPEARED)
    assert quadro(apareceu.t_start, fps) == ancorou
    return [quadro(e.t_start, fps) for e in sessao.events.of_kind(EventKind.BAG_OWNED)]


@pytest.mark.parametrize("fps", TAXAS)
def test_posse_da_orfa_cabe_ate_o_fim_do_prazo_inclusive(fps):
    """Quem chega no último quadro a até `ownership_window_s` da âncora herda
    a órfã; quem chega no seguinte, não (P1)."""
    limiar = CONFIG.custody.ownership_window_s
    n = quadros_dentro(limiar, fps)
    ancorou = fase_critica(fps, n, limiar, encurta=False, minimo=PREPARO)

    assert _dono_chega(fps, ancorou, ancorou + n) == [ancorou + n]
    assert _dono_chega(fps, ancorou, ancorou + n + 1) == []


# --- pipeline.proximity_flag_s ---------------------------------------------------

PERTO_DA_MALA = Point(ANCORA.x + 1.5, ANCORA.y)
"""No raio de permanência e fora do de contato; a mais de `party.proximity_m`
do dono, para nenhum vínculo entrar na cena."""


@pytest.mark.parametrize("fps", TAXAS)
def test_permanencia_sinalizada_no_quadro_que_cobre_o_prazo(fps):
    """O estranho para perto da mala do dono no quadro `e`: o N1 sai em
    `e + ceil(90 * fps)`, nem antes nem depois."""
    limiar = CONFIG.pipeline.proximity_flag_s
    n = quadros_ate_cobrir(limiar, fps)
    chega = fase_critica(fps, n, limiar, encurta=True, minimo=PREPARO)
    inicio = chega - ancora_em(0, fps) - 5

    def cena(i: int) -> list[TrackedDetection]:
        quadro_ = [mala(), pessoa(DONO, LUGAR_DO_DONO)]
        if i >= chega:
            quadro_.append(pessoa(ESTRANHO, PERTO_DA_MALA))
        return quadro_

    sessao = roda(cena, inicio, chega + n + 3, fps)

    flags = sessao.result().flags.for_person(ESTRANHO)
    assert PERTO_DA_MALA.distance_to(LUGAR_DO_DONO) > CONFIG.party.proximity_m
    assert [(f.level.name, quadro(f.t, fps)) for f in flags] == [("N1", chega + n)]


# --- party.weak_bond_s -----------------------------------------------------------

ASSENTO = Point(LUGAR_DO_DONO.x - 0.8 * CONFIG.party.proximity_m, LUGAR_DO_DONO.y)
"""Perto do dono, longe da mala: o de `test_fusao_fps.py`."""


@pytest.mark.parametrize("fps", TAXAS)
def test_vinculo_fraco_na_primeira_fusao_que_cobre_o_prazo(fps):
    """O estranho senta ao lado do dono num quadro de fusão `e`. O vínculo
    fraco sai na primeira fusão a `weak_bond_s` de `e`: fusões caem a cada
    `c = ceil(0,98 * fps)` quadros, então em `e + c * ceil(60 * fps / c)`."""
    limiar = CONFIG.party.weak_bond_s
    c = quadros_ate_cobrir(CONFIG.pipeline.merge_every_s, fps)
    n = c * math.ceil(_exato(limiar) * _exato(fps) / c)
    senta = fase_critica(fps, n, limiar, encurta=True, minimo=PREPARO)
    k = math.ceil((ancora_em(0, fps) + 5) / c)
    inicio = senta - k * c

    def cena(i: int) -> list[TrackedDetection]:
        quadro_ = [mala(), pessoa(DONO, LUGAR_DO_DONO)]
        if i >= senta:
            quadro_.append(pessoa(ESTRANHO, ASSENTO))
        return quadro_

    sessao = roda(cena, inicio, senta + n + 3, fps)

    (fraco,) = sessao.events.of_kind(EventKind.PARTY_JOINED_WEAK)
    assert fraco.subject == ESTRANHO
    assert quadro(fraco.t_start, fps) == senta + n

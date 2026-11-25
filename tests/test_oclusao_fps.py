"""A oclusão de bagagem não pode depender da taxa de quadros da fonte.

`pipeline.missing_frames_before_occluded` contava QUADROS: cinco quadros sem a
bagagem são 0,2 s a 25 fps e 1,67 s a 3 fps. A mesma cena, vista pela mesma
câmera com outro fps, esperava oito vezes mais para abrir a oclusão. A mesma
régua decide se um track de bagagem está voltando de uma ausência
(`_confere_ancora`): a 3 fps, malas gêmeas que somem por um segundo e voltam
com os ids trocados não contavam como ausentes, e a posição deixava de
decidir quem é quem (regra P2).

Limiar temporal vai em segundos, como limiar espacial vai em metros -- a mesma
correção de `reid.max_settle_s` em `test_reid_fps.py`. Mas falha de detector é
evento por quadro: a 3 fps uma única falha já dura 0,33 s, mais que o limiar.
Então as duas réguas valem juntas -- a duração em segundos e um piso de
quadros seguidos sem a bagagem.
"""

import json

import numpy as np
import pytest

from custody_watch.config import Config, PipelineConfig, load_config
from custody_watch.events import EventKind
from custody_watch.ground_plane import GroundPlane
from custody_watch.orchestrator import MIN_MISSING_FRAMES, LiveSession
from custody_watch.tracking import TrackedDetection
from custody_watch.types import BagState, Point

TAXAS = (3.0, 10.0, 25.0)
REFERENCIA_FPS = 25.0
"""Taxa do CAVIAR e de `test_occlusion.py`: onde a régua em quadros foi
ajustada, e onde o comportamento tem de continuar o mesmo."""
QUADROS_DA_REGUA_ANTIGA = 5
"""O default de `missing_frames_before_occluded` antes desta correção."""
PISO_QUADROS = MIN_MISSING_FRAMES
"""Quadros seguidos sem a bagagem abaixo dos quais nenhuma ausência é oclusão."""

CONFIG = Config()
REGISTRO = CONFIG.registry
LIMIAR_S = CONFIG.pipeline.missing_before_occluded_s

# Câmera de cima, 1px = 1cm, igual a test_occlusion.py.
PLANO = GroundPlane(np.diag([0.01, 0.01, 1.0]))

MALA, MALA_A, MALA_B = 900, 101, 102
DONO_A = 1
ANCORA = Point(10.0, 10.0)
# Vão de malas gêmeas de `test_malas_gemeas.py`: cada âncora fora do raio de
# readoção e do de ambiguidade da outra, então a posição decide sem dúvida.
PONTO_B = Point(
    ANCORA.x + 1.5 * max(REGISTRO.reidentification_radius_m, REGISTRO.ambiguity_radius_m),
    ANCORA.y,
)
# O dono fica de lado e um metro abaixo da mala, fora da silhueta (o veto
# "carregada" barraria a âncora), como em `test_malas_gemeas.py`.
AFASTAMENTO = Point(0.7, 1.0)

# A mala assenta (rest_confirm_s) bem antes de sumir.
SOME_EM = 6.0
VOLTA_EM = 10.0
FIM = 12.0
AUSENCIA_GEMEAS_S = 1.2
"""Acima das duas réguas na taxa mais lenta testada, abaixo dos cinco quadros
que a régua antiga exigia a 3 fps."""

LENTA_FPS = 3.0
LEVA_EM = 8.0
VELOCIDADE_MS = 1.5
CHEGA_EM_B = LEVA_EM + ANCORA.distance_to(PONTO_B) / VELOCIDADE_MS
"""Instante em que a mala levada pelo dono passa exatamente sobre a âncora de
B. Cai num quadro de 3 fps, e a falha única é o quadro anterior."""


def caixa(p: Point, altura_px: float):
    """Caixa cujo pé cai em `p`."""
    cx, base = p.x * 100.0, p.y * 100.0
    return (cx - altura_px / 4, base - altura_px, cx + altura_px / 4, base)


def mala(track_id: int, p: Point) -> TrackedDetection:
    return TrackedDetection(track_id, "suitcase", caixa(p, 55))


def pessoa(track_id: int, p: Point) -> TrackedDetection:
    return TrackedDetection(track_id, "person", caixa(p, 170))


def instantes(fps: float, fim: float = FIM) -> list[float]:
    return [i / fps for i in range(round(fim * fps))]


def atraso_da_oclusao(fps: float) -> float:
    """Mala órfã parada some de `SOME_EM` a `VOLTA_EM`. Devolve quanto a
    oclusão demorou a abrir, contado da última vez que a mala foi vista."""
    sessao = LiveSession(PLANO, CONFIG)
    for t in instantes(fps):
        sessao.feed(t, [] if SOME_EM <= t < VOLTA_EM else [mala(MALA, ANCORA)])

    (oclusao,) = sessao.events.of_kind(EventKind.BAG_OCCLUDED)
    ultimo_visto = max(t for t in instantes(fps) if t < SOME_EM)
    return oclusao.t_start - ultimo_visto


def oclusao_abre(fps: float, quadros_sem_mala: int) -> bool:
    """A oclusão chega a abrir quando a mala falta `quadros_sem_mala` quadros
    seguidos? A falha é curta demais para sair no log (`min_occlusion_report_s`),
    então a pergunta vai ao estado da bagagem a cada quadro."""
    sessao = LiveSession(PLANO, CONFIG)
    inicio = round(SOME_EM * fps)
    abriu = False
    for i, t in enumerate(instantes(fps)):
        falta = inicio <= i < inicio + quadros_sem_mala
        sessao.feed(t, [] if falta else [mala(MALA, ANCORA)])
        bag = sessao.bag_for_track(MALA)
        if bag is not None and bag.occluded_since is not None:
            abriu = True
    return abriu


def gemeas_trocadas(fps: float):
    """Duas malas iguais somem juntas por `AUSENCIA_GEMEAS_S` e cada uma volta
    na própria âncora com o id da outra."""
    for t in instantes(fps):
        if t < SOME_EM:
            det = [mala(MALA_A, ANCORA), mala(MALA_B, PONTO_B)]
        elif t < SOME_EM + AUSENCIA_GEMEAS_S:
            det = []
        else:
            det = [mala(MALA_A, PONTO_B), mala(MALA_B, ANCORA)]
        yield t, det


def roda(frames) -> LiveSession:
    sessao = LiveSession(PLANO, CONFIG)
    for t, det in frames:
        sessao.feed(t, det)
    return sessao


def religacoes_de_bagagem(sessao: LiveSession) -> list[tuple[int, int]]:
    """(track bruto, bagagem) de cada `TRACK_RELINKED` de bagagem."""
    return sorted(
        (e.evidence["from_track"], e.bag)
        for e in sessao.events.of_kind(EventKind.TRACK_RELINKED)
        if e.bag is not None
    )


def dono_leva_a_propria_com_uma_falha_sobre_b():
    """B some em `SOME_EM` e fica ocluída. A partir de `LEVA_EM` o dono de A
    leva a própria mala para a direita, a 3 fps, passando exatamente sobre a
    âncora de B -- e o detector perde a mala de A no único quadro anterior."""
    falha = round(CHEGA_EM_B * LENTA_FPS) - 1
    dono_parado = Point(ANCORA.x - AFASTAMENTO.x, ANCORA.y + AFASTAMENTO.y)
    for i, t in enumerate(instantes(LENTA_FPS)):
        andou = max(0.0, t - LEVA_EM) * VELOCIDADE_MS
        dono = Point(dono_parado.x + andou, dono_parado.y)
        det = [pessoa(DONO_A, dono)]
        if i != falha:
            det.append(mala(MALA_A, Point(dono.x + AFASTAMENTO.x, dono.y - AFASTAMENTO.y)))
        if t < SOME_EM:
            det.append(mala(MALA_B, PONTO_B))
        yield t, det


# --- abrir a oclusão ------------------------------------------------------------


def test_a_3_fps_falha_de_um_quadro_nao_abre_oclusao():
    """Uma falha isolada a 3 fps dura 0,33 s, mais que o limiar em segundos,
    mas é um tropeço do detector, não uma oclusão: abrir aqui faria a lista
    de candidatos acumular quem só passou perto a cada falha."""
    assert oclusao_abre(LENTA_FPS, 1) is False


def test_a_3_fps_duas_falhas_seguidas_abrem_oclusao():
    assert oclusao_abre(LENTA_FPS, 2) is True


@pytest.mark.parametrize(
    ("fps", "quadros", "abre"),
    [(10.0, 1, False), (10.0, 2, True), (25.0, 4, False), (25.0, 5, True)],
)
def test_fronteira_entre_tropeco_e_oclusao(fps, quadros, abre):
    """A 25 fps a fronteira segue entre o quarto e o quinto quadro, como
    antes; a 10 fps, os 0,18 s caem no segundo quadro."""
    assert oclusao_abre(fps, quadros) is abre


@pytest.mark.parametrize(("fps", "quadros"), [(3.0, 2), (10.0, 2), (25.0, 5)])
def test_atraso_da_oclusao_por_taxa(fps, quadros):
    """Com a config default: 0,2 s a 25 fps (o quinto quadro de antes) e a
    10 fps; a 3 fps manda o piso de dois quadros, 0,67 s."""
    assert atraso_da_oclusao(fps) == pytest.approx(quadros / fps, abs=1e-9)


@pytest.mark.parametrize("fps", [10.0, 15.0])
def test_mesma_ausencia_abre_a_oclusao_no_mesmo_tempo_real(fps):
    """A 10 fps os cinco quadros eram 0,5 s, contra 0,2 s da mesma cena a
    25 fps. Onde o piso de quadros cabe no limiar mais um período, a duração
    real é a mesma da referência; um período de diferença é o que decidir em
    quadro impõe, mais que isso é a régua mudando com o fps."""
    piso_cabe = LIMIAR_S + 1.0 / fps
    assert piso_cabe >= PISO_QUADROS / fps

    referencia = atraso_da_oclusao(REFERENCIA_FPS)
    assert atraso_da_oclusao(fps) == pytest.approx(referencia, abs=1.0 / fps)


@pytest.mark.parametrize("fps", TAXAS)
def test_oclusao_nunca_abre_antes_das_duas_reguas(fps):
    """Nunca antes do limiar em segundos, nunca com menos de dois quadros sem
    a bagagem, e no máximo um período depois de a mais exigente vencer."""
    atraso = atraso_da_oclusao(fps)
    mais_exigente = max(LIMIAR_S, PISO_QUADROS / fps)

    assert atraso >= LIMIAR_S
    assert round(atraso * fps) >= PISO_QUADROS
    assert atraso < mais_exigente + 1.0 / fps


# --- voltar de uma ausência -------------------------------------------------------


def test_premissa_das_gemeas():
    """A ausência passa das duas réguas com folga de um período na taxa mais
    lenta, mas fica abaixo dos cinco quadros que a régua antiga pedia a 3 fps
    -- é o que torna a cena discriminante. E as âncoras ficam fora do raio de
    readoção e de ambiguidade uma da outra."""
    periodo_lento = 1.0 / min(TAXAS)
    regua_nova = max(LIMIAR_S, PISO_QUADROS * periodo_lento) + periodo_lento
    regua_antiga = QUADROS_DA_REGUA_ANTIGA * periodo_lento
    vao = ANCORA.distance_to(PONTO_B)

    assert regua_nova < AUSENCIA_GEMEAS_S
    assert regua_antiga > AUSENCIA_GEMEAS_S
    assert vao > REGISTRO.reidentification_radius_m
    assert vao > REGISTRO.ambiguity_radius_m
    assert vao >= REGISTRO.moved_threshold_m


@pytest.mark.parametrize("fps", TAXAS)
def test_gemeas_trocadas_respondem_pela_ancora_em_qualquer_fps(fps):
    """1,2 s de sumiço são 30 quadros a 25 fps e quatro a 3 fps. Com a régua
    em quadros, a 3 fps nenhum dos dois tracks contava como voltando de
    ausência, e cada um seguia respondendo pela mala do id -- lida como
    deslocada para a âncora da vizinha."""
    sessao = roda(gemeas_trocadas(fps))

    assert religacoes_de_bagagem(sessao) == [(MALA_A, MALA_B), (MALA_B, MALA_A)]


@pytest.mark.parametrize("fps", TAXAS)
def test_gemeas_trocadas_ficam_paradas_em_qualquer_fps(fps):
    """Nenhuma das duas saiu do lugar: a custódia segue aberta, sem
    ambiguidade e com cada âncora onde a mala foi deixada."""
    sessao = roda(gemeas_trocadas(fps))

    for track_id, ancora in ((MALA_B, ANCORA), (MALA_A, PONTO_B)):
        bag = sessao.bag_for_track(track_id)
        assert bag is not None
        assert bag.state is not BagState.AMBIGUA
        assert bag.anchor.distance_to(ancora) < REGISTRO.moved_threshold_m


def test_premissa_da_falha_unica_sobre_b():
    """A falha cai no quadro anterior ao que põe a mala sobre a âncora de B,
    B já está declarada ocluída, e a falha dura mais que o limiar em segundos
    -- só o piso de quadros separa a cena de uma volta de ausência."""
    quadro_sobre_b = CHEGA_EM_B * LENTA_FPS

    assert quadro_sobre_b == pytest.approx(round(quadro_sobre_b), abs=1e-9)
    assert LEVA_EM - SOME_EM > PISO_QUADROS / LENTA_FPS + 1.0 / LENTA_FPS
    assert 1.0 / LENTA_FPS > LIMIAR_S


def test_mala_levada_com_uma_falha_sobre_ancora_ocluida_segue_sendo_ela():
    """O dono leva a própria mala sobre a âncora de B, ocluída, e o detector
    perde a mala dele num único quadro de 3 fps. Um track que faltou um
    quadro não está voltando de ausência, e a mala dele não está ocluída:
    segue sendo a mala do dono, aberta. Só com a régua em segundos, a falha
    lia a volta sobre B como volta de ausência, com a própria mala do dono
    também ocluída no raio -- as duas viravam AMBIGUA e a custódia de ambas
    se perdia."""
    sessao = roda(dono_leva_a_propria_com_uma_falha_sobre_b())

    bag = sessao.bag_for_track(MALA_A)
    assert bag is not None
    assert bag.bag_id == MALA_A
    assert bag.state is not BagState.AMBIGUA
    assert religacoes_de_bagagem(sessao) == []


# --- config ---------------------------------------------------------------------


def test_default_reproduz_o_quinto_quadro_a_25_fps():
    """Qualquer limiar em (4/25, 5/25] s abre a oclusão no quinto quadro a
    25 fps. O default fica no meio do intervalo, longe das duas bordas: a
    soma binária de 0,04 s e o jitter de timestamp de uma câmera real não
    podem decidir se a oclusão abre no quarto, no quinto ou no sexto."""
    limiar = PipelineConfig().missing_before_occluded_s

    assert limiar > (QUADROS_DA_REGUA_ANTIGA - 1) / REFERENCIA_FPS
    assert limiar < QUADROS_DA_REGUA_ANTIGA / REFERENCIA_FPS


def test_campo_antigo_em_quadros_e_recusado_apontando_o_novo(tmp_path):
    """Converter quadros em segundos exige o fps da fonte, que a config não
    conhece; adivinhar 25 fps reabriria o defeito em silêncio. Então a config
    antiga falha alto e diz o que pôr no lugar."""
    caminho = tmp_path / "c.json"
    caminho.write_text(
        json.dumps({"pipeline": {"missing_frames_before_occluded": 5}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="missing_before_occluded_s") as erro:
        load_config(caminho)

    assert "fps" in str(erro.value)


def test_limiar_em_segundos_dentro_da_faixa_carrega(tmp_path):
    caminho = tmp_path / "c.json"
    caminho.write_text(
        json.dumps({"pipeline": {"missing_before_occluded_s": 0.5}}), encoding="utf-8"
    )

    assert load_config(caminho).pipeline.missing_before_occluded_s == pytest.approx(0.5)

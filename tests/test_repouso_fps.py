"""A bagagem parada vira âncora depois de `rest_confirm_s`, em qualquer taxa.

`anchoring` guardava só as amostras com `t >= agora - rest_confirm_s`, e
`_Pendente.parada` exigia que elas cobrissem `rest_confirm_s` inteiros. As
duas condições só se encontram quando uma amostra cai exatamente no corte:
com `rest_confirm_s * fps` inteiro, e mesmo assim só se a subtração em ponto
flutuante der exata. A 3,25 fps a janela cobria no máximo 6 / 3,25 = 1,846 s
dos 2 s, e nenhuma bagagem virava âncora -- sem âncora não há posse,
desacompanhada nem acusação, e o benchmark caía de 11/11 para 7/11 em toda
taxa com `2 * fps` não inteiro (3,25; 3,75; ...; 29,75). A 29,97 fps, a taxa
NTSC, o mesmo; e ao vivo, com o instante vindo do relógio, nenhuma amostra
caía no corte. Mesmo em taxa inteira, conforme o quadro em que a mala
aparecia, o arredondamento a assentava um ou mais quadros depois.

A régua do repouso decide como as outras do orquestrador em segundos
(`merge_every_s`, `missing_before_occluded_s`): no primeiro quadro em que o
tempo parada vence o limiar, nunca antes. O que sobra de diferença entre
taxas é um período de amostragem.
"""

import math
import random

import numpy as np
import pytest

from custody_watch.config import Config
from custody_watch.events import EventKind
from custody_watch.ground_plane import GroundPlane
from custody_watch.orchestrator import LiveSession
from custody_watch.tracking import TrackedDetection
from custody_watch.types import Point

CONFIG = Config()
REPOUSO_S = CONFIG.custody.rest_confirm_s
LIMIAR_M = CONFIG.registry.moved_threshold_m

TAXAS_INTEIRAS = (3.0, 10.0, 25.0)
TAXAS_NAO_INTEIRAS = (3.25, 4.75, 7.3, 12.5, 29.97)
"""As duas do relato (3,25 e 4,75), uma sem relação com meio segundo (7,3),
uma em que `2 * fps` é inteiro sem a taxa ser (12,5) e a NTSC (29,97)."""
QUADROS_INICIAIS = 40
"""Quadros em que a mala pode aparecer pela primeira vez. A fase importa: a
3 fps a mala que aparece no quadro 20 tinha a amostra do corte descartada
por arredondamento, e assentava um quadro depois do devido."""
FOLGA_QUADROS = 3
"""Quadros além do devido que a sessão roda: atraso aparece como atraso, não
como ausência de evento."""

# Câmera de cima, 1px = 1cm, igual a test_oclusao_fps.py.
PLANO = GroundPlane(np.diag([0.01, 0.01, 1.0]))
MALA = 101
ANCORA = Point(10.0, 10.0)

RELOGIO_FPS = 10.0
JITTER_S = 0.01
"""Variação do intervalo entre quadros de uma câmera ao vivo, cujo instante
vem do relógio (`aovivo`) e não do índice do quadro."""

VELOCIDADE_EM_TRANSITO_MS = 2.0 * LIMIAR_M / REPOUSO_S
"""O dobro do corte de velocidade de `parada` (limiar / janela): mala em
trânsito, que nenhuma janela pode tomar por parada."""
TRANSITO_S = 10.0


def caixa(p: Point, altura_px: float):
    """Caixa cujo pé cai em `p`."""
    cx, base = p.x * 100.0, p.y * 100.0
    return (cx - altura_px / 4, base - altura_px, cx + altura_px / 4, base)


def mala(p: Point) -> TrackedDetection:
    return TrackedDetection(MALA, "suitcase", caixa(p, 55))


def quadros_devidos(fps: float) -> int:
    """Períodos até a mala parada cobrir `rest_confirm_s`: o primeiro quadro
    em que o tempo parada vence o limiar."""
    return math.ceil(REPOUSO_S * fps)


def quadros_ate_assentar(fps: float, primeiro_quadro: int) -> int | None:
    """A mala aparece parada no quadro `primeiro_quadro` (`t = i / fps`, como
    a fonte entrega) e fica. Quantos quadros depois ela vira âncora, ou
    `None` se não vira."""
    sessao = LiveSession(PLANO, CONFIG)
    fim = primeiro_quadro + quadros_devidos(fps) + FOLGA_QUADROS
    for i in range(fim):
        sessao.feed(i / fps, [mala(ANCORA)] if i >= primeiro_quadro else [])

    aparicoes = sessao.events.of_kind(EventKind.BAG_APPEARED)
    if not aparicoes:
        return None
    return round(aparicoes[0].t_start * fps) - primeiro_quadro


@pytest.mark.parametrize("fps", TAXAS_NAO_INTEIRAS)
def test_mala_parada_vira_ancora_em_taxa_nao_inteira(fps):
    """O sintoma: a 3,25 e a 4,75 fps a mala nunca virava âncora."""
    assert quadros_ate_assentar(fps, 0) == quadros_devidos(fps)


@pytest.mark.parametrize("fps", (*TAXAS_INTEIRAS, *TAXAS_NAO_INTEIRAS))
def test_assenta_no_primeiro_quadro_que_cobre_o_repouso_em_qualquer_fase(fps):
    """Nunca antes de `rest_confirm_s` parada, e não um quadro depois por
    arredondamento: o mesmo quadro devido, apareça a mala em que quadro for."""
    fora_do_devido = {
        inicio: quadros
        for inicio in range(QUADROS_INICIAIS)
        if (quadros := quadros_ate_assentar(fps, inicio)) != quadros_devidos(fps)
    }

    assert fora_do_devido == {}


def instantes_de_relogio(seed: int, fim: float) -> list[float]:
    rng = random.Random(seed)
    instantes, t = [], 0.0
    while t < fim:
        instantes.append(t)
        t += 1.0 / RELOGIO_FPS + rng.uniform(-JITTER_S, JITTER_S)
    return instantes


@pytest.mark.parametrize("seed", range(5))
def test_mala_parada_vira_ancora_com_instantes_de_relogio(seed):
    """Ao vivo nenhuma amostra cai exatamente no corte, então nenhuma mala
    parada virava âncora. Vira no primeiro quadro a `rest_confirm_s` do
    primeiro em que foi vista."""
    instantes = instantes_de_relogio(seed, 2.0 * REPOUSO_S)
    sessao = LiveSession(PLANO, CONFIG)
    for t in instantes:
        sessao.feed(t, [mala(ANCORA)])

    aparicoes = sessao.events.of_kind(EventKind.BAG_APPEARED)
    devido = min(t for t in instantes if t >= REPOUSO_S)
    assert len(aparicoes) == 1
    assert aparicoes[0].t_start == pytest.approx(devido, abs=1e-9)


@pytest.mark.parametrize("fps", (*TAXAS_INTEIRAS, *TAXAS_NAO_INTEIRAS))
def test_mala_em_transito_nunca_vira_ancora(fps):
    """A janela que cobre o repouso não pode afrouxar `parada`: mala andando
    ao dobro do corte de velocidade segue sem âncora a cena inteira."""
    sessao = LiveSession(PLANO, CONFIG)
    for i in range(round(TRANSITO_S * fps)):
        t = i / fps
        sessao.feed(t, [mala(Point(ANCORA.x + VELOCIDADE_EM_TRANSITO_MS * t, ANCORA.y))])

    assert sessao.events.of_kind(EventKind.BAG_APPEARED) == []

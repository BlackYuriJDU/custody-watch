"""Cronômetros de proximidade só contam tempo OBSERVADO.

`near_since` (flag N1) e `together_since` (vínculo fraco) mediam "desde
quando" sem saber se a pessoa continuou à vista. Quem ficava 10s perto da
mala, sumia de quadro por 150s e voltava recebia N1 no ato, com a evidência
"Permaneceu 160s" -- uma relação que ninguém observou (P3/P4).

O oposto também é defeito: o detector perde pessoas por instantes (rajadas
medidas no CAVIAR, ver `noise.CAVIAR_PERSON_NOISE`), e zerar a cada falha
tornaria N1 e vínculo fraco inalcançáveis. Lacuna curta não zera; ausência
longa zera, e a chave sai do estado.
"""

from collections.abc import Callable

import numpy as np

from custody_watch.config import Config, PartyConfig, PipelineConfig
from custody_watch.events import EventKind
from custody_watch.ground_plane import GroundPlane
from custody_watch.orchestrator import LiveSession
from custody_watch.tracking import TrackedDetection
from custody_watch.types import FlagLevel

PLANO = GroundPlane(np.eye(3))
SEM_PISO = 0.0
PASSO = 0.1
"""10 Hz: lacunas de fração de segundo precisam de quadros para existir."""

DONO = 1
ESTRANHO = 7
PROXIMIDADE_S = 10.0
VINCULO_FRACO_S = 10.0

CHEGADA = 3.0
SAIDA = 8.0
"""Cinco segundos perto: menos que os dois cronômetros, nada sai antes de sumir."""
RETORNO = 158.0
"""150s fora de quadro."""
FIM = 175.0


def pessoa(track_id: int, x: float) -> TrackedDetection:
    return TrackedDetection(track_id=track_id, cls="person", bbox=(x - 0.5, 0.0, x + 0.5, 1.0))


def mala(track_id: int, x: float) -> TrackedDetection:
    """Base na linha dos pés de quem está ao lado, como em test_orchestrator."""
    return TrackedDetection(track_id=track_id, cls="suitcase", bbox=(x - 0.2, 0.6, x + 0.2, 1.0))


def config_proximidade() -> Config:
    """N1 em 10s. Vínculo fraco no teto da faixa, para não pôr o estranho no
    grupo do dono -- quem é do grupo não recebe N1, e o teste mediria outra
    coisa."""
    return Config(
        party=PartyConfig(weak_bond_s=600.0),
        pipeline=PipelineConfig(
            merge_every_s=0.0, proximity_flag_s=PROXIMIDADE_S, min_bag_height_px=SEM_PISO
        ),
    )


def config_vinculo() -> Config:
    """Vínculo fraco em 10s, o piso da faixa segura."""
    return Config(
        party=PartyConfig(weak_bond_s=VINCULO_FRACO_S),
        pipeline=PipelineConfig(merge_every_s=0.0, min_bag_height_px=SEM_PISO),
    )


def roda(config: Config, estranho_visivel: Callable[[float], bool], fim: float = FIM):
    """Dono parado ao lado da mala o tempo todo; o estranho a 1,5m do dono e
    a 1,2m da mala -- perto dos dois, longe do contato -- quando visível."""
    sessao = LiveSession(PLANO, config)
    for i in range(round(fim / PASSO) + 1):
        t = round(i * PASSO, 9)
        quadro = [pessoa(DONO, 0.0), mala(9, 0.3)]
        if estranho_visivel(t):
            quadro.append(pessoa(ESTRANHO, 1.5))
        sessao.feed(t, quadro)
    return sessao


def n1_do_estranho(sessao: LiveSession):
    return [f for f in sessao._s.flags.for_person(ESTRANHO) if f.level is FlagLevel.N1]


def vinculos_fracos(sessao: LiveSession):
    return [e for e in sessao.events.of_kind(EventKind.PARTY_JOINED_WEAK) if e.subject == ESTRANHO]


def ausencia_longa(t: float) -> bool:
    return CHEGADA <= t < SAIDA or t >= RETORNO


def lacunas_curtas(t: float) -> bool:
    """Três falhas de detector de até 1s, como as rajadas do CAVIAR."""
    falhas = ((5.0, 5.5), (8.0, 9.0), (11.0, 11.5))
    return t >= CHEGADA and not any(inicio <= t < fim for inicio, fim in falhas)


def continuo(t: float) -> bool:
    return t >= CHEGADA


# --- N1 / near_since ----------------------------------------------------------


def test_ausencia_longa_nao_conta_como_permanencia_para_n1():
    """O defeito: N1 saía no retorno, com 160s de "permanência" inventados."""
    sessao = roda(config_proximidade(), ausencia_longa)

    flags = n1_do_estranho(sessao)

    assert len(flags) == 1
    assert flags[0].t >= RETORNO + PROXIMIDADE_S - 1e-6, (
        f"N1 em t={flags[0].t}, antes de {PROXIMIDADE_S}s observados desde o retorno"
    )
    assert flags[0].t <= RETORNO + PROXIMIDADE_S + PASSO + 1e-6
    assert f"Permaneceu {int(PROXIMIDADE_S)}s" in flags[0].explanation


def test_lacuna_curta_de_deteccao_nao_zera_cronometro_de_n1():
    """Zerar a cada falha do detector tornaria o N1 inalcançável."""
    sessao = roda(config_proximidade(), lacunas_curtas, fim=20.0)

    flags = n1_do_estranho(sessao)

    assert len(flags) == 1
    assert abs(flags[0].t - (CHEGADA + PROXIMIDADE_S)) < 1e-6


def test_permanencia_continua_gera_n1_no_mesmo_instante_de_antes():
    """Regressão: sem lacuna nenhuma, nada muda."""
    sessao = roda(config_proximidade(), continuo, fim=20.0)

    flags = n1_do_estranho(sessao)

    assert len(flags) == 1
    assert abs(flags[0].t - (CHEGADA + PROXIMIDADE_S)) < 1e-6
    assert f"Permaneceu {int(PROXIMIDADE_S)}s" in flags[0].explanation


# --- vínculo fraco / together_since -------------------------------------------


def test_ausencia_longa_nao_vira_vinculo_fraco_instantaneo():
    """Mesmo mecanismo do N1: o tempo fora de quadro contava como "juntos"."""
    sessao = roda(config_vinculo(), ausencia_longa)

    vinculos = vinculos_fracos(sessao)

    assert len(vinculos) == 1
    assert vinculos[0].t_start >= RETORNO + VINCULO_FRACO_S - 1e-6, (
        f"vínculo fraco em t={vinculos[0].t_start}, antes de {VINCULO_FRACO_S}s "
        "observados desde o retorno"
    )
    assert vinculos[0].t_start <= RETORNO + VINCULO_FRACO_S + PASSO + 1e-6


def test_lacuna_curta_de_deteccao_nao_zera_cronometro_de_vinculo_fraco():
    sessao = roda(config_vinculo(), lacunas_curtas, fim=20.0)

    vinculos = vinculos_fracos(sessao)

    assert len(vinculos) == 1
    assert abs(vinculos[0].t_start - (CHEGADA + VINCULO_FRACO_S)) < 1e-6


# --- estado -------------------------------------------------------------------


def test_quem_sumiu_de_vez_sai_dos_cronometros():
    """Parte da issue #30: chaves de tracks que somem nunca eram removidas.

    Vinte passantes, cada um 3s perto da mala e do dono, depois 150s só com o
    dono à vista. Nenhum cronômetro pode sobrar para quem não está mais lá.
    """
    sessao = LiveSession(PLANO, config_proximidade())
    passantes = range(100, 120)
    fim = 3.0 + 3.0 * len(passantes) + 150.0
    for i in range(round(fim / PASSO) + 1):
        t = round(i * PASSO, 9)
        quadro = [pessoa(DONO, 0.0), mala(9, 0.3)]
        indice = int((t - 3.0) // 3.0)
        if t >= 3.0 and indice < len(passantes):
            quadro.append(pessoa(passantes[indice], 1.5))
        sessao.feed(t, quadro)

    assert sessao._s.near_since == {}
    assert sessao._s.together_since == {}

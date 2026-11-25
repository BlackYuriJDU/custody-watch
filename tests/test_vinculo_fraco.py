"""Regra P1 nos pontos em que "é do grupo dono" vira decisão.

Vínculo fraco é quem só ficou parado perto de um membro por `weak_bond_s`.
Ele não transfere posse (`Party.owns`) e, pela mesma regra, não pode apagar
nem impedir o flag: "Weak bonds attenuate a flag; they never clear it"
(CONTRIBUTING). Comparar `party_of(pessoa) == bag.owner_party` trata o membro
fraco como dono -- e o ladrão que senta ao lado da vítima por um minuto passa
a encostar na mala dela sem flag nenhum, e a mantém "acompanhada" depois que
a dona sai.
"""

import numpy as np
import pytest

from custody_watch.config import Config, CustodyConfig, PartyConfig, PipelineConfig
from custody_watch.custody import update_attendance
from custody_watch.events import EventKind
from custody_watch.flags import flag_contact
from custody_watch.ground_plane import GroundPlane
from custody_watch.orchestrator import LiveSession, run_session
from custody_watch.party import PartyManager
from custody_watch.tracking import TrackedDetection
from custody_watch.types import Bag, BagState, FlagLevel, Observation, Point

PLANO = GroundPlane(np.eye(3))

# Merge a cada quadro e vínculo fraco no piso da faixa segura (10 s): o
# cenário precisa que o estranho vire membro fraco bem antes do N1 de
# permanência (90 s no default), senão o flag sairia antes do vínculo e o
# teste não exercitaria nada.
WEAK_BOND_S = 10.0
CONFIG = Config(
    party=PartyConfig(weak_bond_s=WEAK_BOND_S),
    pipeline=PipelineConfig(merge_every_s=0.0, min_bag_height_px=0.0),
    custody=CustodyConfig(max_occlusion_s=5.0),
)

DONO = 1
ESTRANHO = 7
MALA = 9


def pessoa(track_id: int, x: float) -> TrackedDetection:
    return TrackedDetection(track_id=track_id, cls="person", bbox=(x - 0.5, 0.0, x + 0.5, 1.0))


def mala(track_id: int, x: float) -> TrackedDetection:
    return TrackedDetection(track_id=track_id, cls="suitcase", bbox=(x - 0.2, 0.6, x + 0.2, 1.0))


def cena(*frames: list[TrackedDetection]):
    return ((float(i), frame) for i, frame in enumerate(frames))


def estranho_vira_membro_fraco(segundos: int = 15) -> list[list[TrackedDetection]]:
    """Dono ao lado da mala; estranho parado a 1,5 m dele e 1,2 m dela.

    Perto o bastante do dono para virar vínculo fraco, longe o bastante da
    mala para não contar como contato.
    """
    return [[pessoa(DONO, 0.0), mala(MALA, 0.3), pessoa(ESTRANHO, 1.5)]] * segundos


def vinculos_fracos(resultado) -> list[int]:
    return [e.subject for e in resultado.events.of_kind(EventKind.PARTY_JOINED_WEAK)]


# --- flags relacionais ----------------------------------------------------------


def test_membro_fraco_que_encosta_na_mala_do_grupo_recebe_n2():
    """Sentar perto do dono não autoriza tocar na mala dele sem registro."""
    quadros = estranho_vira_membro_fraco()
    quadros += [[pessoa(DONO, 0.0), mala(MALA, 0.3), pessoa(ESTRANHO, 0.3)]] * 3

    resultado = run_session(cena(*quadros), PLANO, CONFIG)

    assert vinculos_fracos(resultado) == [ESTRANHO]
    flags = resultado.flags.for_person(ESTRANHO)
    assert [f.level for f in flags] == [FlagLevel.N2]
    assert flags[0].bag == MALA


def test_membro_fraco_que_permanece_junto_a_mala_do_grupo_recebe_n1():
    """A permanência prolongada continua contando depois do vínculo fraco."""
    config = Config(
        party=PartyConfig(weak_bond_s=WEAK_BOND_S),
        pipeline=PipelineConfig(merge_every_s=0.0, proximity_flag_s=30.0, min_bag_height_px=0.0),
    )
    quadros = estranho_vira_membro_fraco(segundos=40)

    resultado = run_session(cena(*quadros), PLANO, config)

    assert vinculos_fracos(resultado) == [ESTRANHO]
    flags = resultado.flags.for_person(ESTRANHO)
    assert [f.level for f in flags] == [FlagLevel.N1]
    assert flags[0].bag == MALA


def test_posse_tardia_nao_apaga_flag_de_membro_fraco():
    """`esquece_contatos_do_dono` perdoa quem se revela dono, não quem só sentou perto."""
    sessao = LiveSession(PLANO, CONFIG)
    s = sessao._s
    grupo = s.parties.form_on_arrival([DONO])
    assert s.parties.join_weak(grupo.party_id, ESTRANHO)
    bagagem = Bag(bag_id=MALA, anchor=Point(0.0, 0.0), owner_party=grupo.party_id)
    for quem in (DONO, ESTRANHO):
        s.flags.add(flag_contact(quem, bagagem, t=5.0))
        s.contact_flagged.add((quem, MALA))

    s.esquece_contatos_do_dono(bagagem)

    assert [f.level for f in s.flags.for_person(ESTRANHO)] == [FlagLevel.N2]
    assert (ESTRANHO, MALA) in s.contact_flagged
    # Regressão: o dono de vínculo forte continua perdoado.
    assert s.flags.for_person(DONO) == []
    assert (DONO, MALA) not in s.contact_flagged


# --- desacompanhamento ----------------------------------------------------------


def test_membro_fraco_nao_conta_como_dono_presente():
    """Unidade: a distância que importa é a do vínculo forte mais próximo."""
    manager = PartyManager()
    grupo = manager.form_on_arrival([DONO])
    manager.join_weak(grupo.party_id, ESTRANHO)
    bagagem = Bag(bag_id=MALA, anchor=Point(0.0, 0.0), owner_party=grupo.party_id)

    def quadro(t: float) -> list[Observation]:
        return [
            Observation(track_id=DONO, cls="person", position=Point(10.0, 0.0), t=t),
            Observation(track_id=ESTRANHO, cls="person", position=Point(1.0, 0.0), t=t),
        ]

    update_attendance(bagagem, quadro(0.0), manager, t=0.0)
    update_attendance(bagagem, quadro(26.0), manager, t=26.0)

    assert bagagem.state is BagState.DESACOMPANHADA


def test_dono_sai_e_membro_fraco_fica_mala_desacompanhada_no_prazo():
    """O ladrão sentado ao lado não mantém a mala "acompanhada" depois que a dona sai."""
    saida = 15
    quadros = estranho_vira_membro_fraco(segundos=saida)
    quadros += [[pessoa(DONO, 10.0), mala(MALA, 0.3), pessoa(ESTRANHO, 1.5)]] * 35

    resultado = run_session(cena(*quadros), PLANO, CONFIG)

    assert vinculos_fracos(resultado) == [ESTRANHO]
    alertas = resultado.events.of_kind(EventKind.BAG_UNATTENDED)
    assert len(alertas) == 1
    prazo = CONFIG.custody.unattended_time_s
    assert alertas[0].t_start == pytest.approx(saida + prazo, abs=1.0)


# --- regressão: vínculo forte continua sendo dono ---------------------------------


def casal_deposita_e_um_sai() -> list[list[TrackedDetection]]:
    """O casal anda junto, deposita a mala, e só o companheiro fica, encostado nela."""
    quadros = [[pessoa(DONO, 2.0 * i), pessoa(2, 2.0 * i + 0.5)] for i in range(6)]
    quadros += [[pessoa(DONO, 10.0), pessoa(2, 10.5), mala(MALA, 10.2)]] * 5
    quadros += [[pessoa(DONO, 30.0), pessoa(2, 10.3), mala(MALA, 10.2)]] * 35
    return quadros


def test_membro_forte_continua_sem_flag_na_mala_do_grupo():
    resultado = run_session(cena(*casal_deposita_e_um_sai()), PLANO, CONFIG)

    assert len(resultado.events.of_kind(EventKind.PARTY_JOINED_STRONG)) == 1
    assert resultado.flags.for_person(2) == []
    assert resultado.flags.for_person(DONO) == []


def test_membro_forte_continua_contando_como_presenca():
    resultado = run_session(cena(*casal_deposita_e_um_sai()), PLANO, CONFIG)

    assert len(resultado.events.of_kind(EventKind.PARTY_JOINED_STRONG)) == 1
    assert len(resultado.events.of_kind(EventKind.BAG_OWNED)) == 1
    assert resultado.events.of_kind(EventKind.BAG_UNATTENDED) == []

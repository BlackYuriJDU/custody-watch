"""Custódia encerrada não aceita dono apontado depois.

`aponta_dono` só recusava bagagem **arquivada** (commit 141cc72), e o arquivo
terminal é opt-in. Sem ele, apontar como dono quem levou a bagagem era aceito:
saía `BAG_OWNED`, `esquece_contatos_do_dono` apagava o N3 e a fila passava de
`[(7, N3)]` para `[]`. Um clique -- ou um POST forjado -- apagava a evidência
mais grave de uma custódia já decidida. A recusa não pode depender de o
arquivo estar ligado.
"""

import pytest

from custody_watch.config import Config
from custody_watch.events import EventKind
from custody_watch.ground_plane import GroundPlane
from custody_watch.orchestrator import LiveSession
from custody_watch.tracking import TrackedDetection
from custody_watch.types import TERMINAL_BAG_STATES, BagState, FlagLevel
from tests.test_orchestrator import PLANO, RAPIDO, casal_andando, mala, pessoa

LADRAO, COMPANHEIRO, MALA = 7, 2, 9


def _centro(tracked: list[TrackedDetection], track_id: int) -> tuple[float, float]:
    x0, y0, x1, y1 = next(d.bbox for d in tracked if d.track_id == track_id)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _alimenta(sessao: LiveSession, quadros, passo: float = 1.0) -> list[TrackedDetection]:
    tracked: list[TrackedDetection] = []
    for i, tracked in enumerate(quadros):
        sessao.feed(i * passo, tracked)
    return tracked


def _orfa_levada_andando():
    """O cenário da auditoria: mala órfã, e quem chega aos 15s sai com ela.

    Resolve por `carry_away`, com a bagagem à vista o tempo todo.
    """
    plano = GroundPlane.uniform(0.01)

    def pessoa_px(tid, fx, fy):
        return TrackedDetection(tid, "person", (fx - 20, fy - 170, fx + 20, fy))

    def mala_px(tid, fx, fy):
        return TrackedDetection(tid, "suitcase", (fx - 20, fy - 45, fx + 20, fy))

    quadros = []
    for i in range(400):
        t = i / 10
        dets = []
        if t < 20:
            dets.append(mala_px(100, 550, 500))
        if t >= 15:
            x = 600 if t < 20 else 600 + int((t - 20) * 100)
            dets.append(pessoa_px(LADRAO, x, 500))
            if t >= 20:
                dets.append(mala_px(100, x + 30, 500))
        quadros.append(dets)

    sessao = LiveSession(plano)
    ultimo = _alimenta(sessao, quadros, passo=0.1)
    return sessao, sessao.bag_for_track(100).bag_id, _centro(ultimo, LADRAO)


def _estranho_recolhe_a_mala_do_casal():
    """Mala com dono, levada por quem não é do grupo: resolve no timeout de
    oclusão, com o ladrão como único candidato."""
    quadros = casal_andando()
    quadros += [[pessoa(1, 10.0), pessoa(2, 10.5), mala(MALA, 10.2)]] * 3
    quadros += [[pessoa(LADRAO, 10.2)]] * 16

    sessao = LiveSession(PLANO, RAPIDO)
    ultimo = _alimenta(sessao, quadros)
    return sessao, MALA, _centro(ultimo, LADRAO)


def _companheiro_recolhe_a_mala():
    """Retirada legítima: o membro forte do grupo leva a mala."""
    quadros = casal_andando()
    quadros += [[pessoa(1, 10.0), pessoa(2, 10.5), mala(MALA, 10.2)]] * 3
    quadros += [[pessoa(2, 10.3), mala(MALA, 10.2)]] * 3
    quadros += [[pessoa(2, 10.3)]] * 16

    sessao = LiveSession(PLANO, RAPIDO)
    ultimo = _alimenta(sessao, quadros)
    return sessao, MALA, _centro(ultimo, COMPANHEIRO)


CENARIOS = {
    "orfa_levada_andando": (_orfa_levada_andando, BagState.RETIRADA_ESTRANHO),
    "estranho_recolhe_mala_do_casal": (
        _estranho_recolhe_a_mala_do_casal,
        BagState.RETIRADA_ESTRANHO,
    ),
    "companheiro_recolhe_a_mala": (_companheiro_recolhe_a_mala, BagState.RETIRADA_DONO),
}


@pytest.fixture(params=list(CENARIOS), ids=list(CENARIOS))
def encerrada(request):
    montar, estado = CENARIOS[request.param]
    sessao, bag_id, ponto = montar()
    bagagem = sessao._s.registry.get(bag_id)
    assert bagagem.state is estado, "o cenário precisa terminar no estado que diz cobrir"
    return sessao, bagagem, ponto


def _retrato(sessao: LiveSession, bagagem) -> dict:
    s = sessao._s
    return {
        "fila": sessao.queue(),
        "flags": {p: s.flags.for_person(p) for p in s.flags.people()},
        "contatos": set(s.contact_flagged),
        "eventos": len(list(sessao.events)),
        "grupos": len(s.parties._parties),
        "dono": bagagem.owner_party,
        "estado": bagagem.state,
    }


def test_os_cenarios_cobrem_todos_os_estados_terminais():
    """Se um estado terminal novo surgir, este teste obriga a ganhar cenário."""
    assert {estado for _, estado in CENARIOS.values()} == TERMINAL_BAG_STATES


def test_apontar_dono_de_custodia_encerrada_recusa(encerrada):
    sessao, bagagem, ponto = encerrada

    with pytest.raises(ValueError, match="custódia .* encerrada"):
        sessao.aponta_dono(bagagem.bag_id, *ponto)


def test_recusa_preserva_fila_flags_e_posse(encerrada):
    """Inclusive o N3: é a evidência que a tentativa apagava."""
    sessao, bagagem, ponto = encerrada
    antes = _retrato(sessao, bagagem)

    with pytest.raises(ValueError):
        sessao.aponta_dono(bagagem.bag_id, *ponto)

    assert _retrato(sessao, bagagem) == antes


def test_recusa_nao_emite_evento(encerrada):
    sessao, bagagem, ponto = encerrada
    eventos_antes = list(sessao.events)

    with pytest.raises(ValueError):
        sessao.aponta_dono(bagagem.bag_id, *ponto)

    assert list(sessao.events) == eventos_antes
    assert not [
        e
        for e in sessao.events.of_kind(EventKind.BAG_OWNED)
        if e.bag == bagagem.bag_id and e.evidence.get("origem") == "operador"
    ]


def test_recusa_vem_antes_de_procurar_quem_esta_no_ponto(encerrada):
    """O motivo é a bagagem, não o clique: clicar no vazio também diz que a
    custódia está encerrada."""
    sessao, bagagem, _ = encerrada

    with pytest.raises(ValueError, match="encerrada"):
        sessao.aponta_dono(bagagem.bag_id, -1.0, -1.0)


def test_n3_do_ladrao_segue_no_topo_da_fila_depois_da_tentativa():
    sessao, bag_id, ponto = _estranho_recolhe_a_mala_do_casal()

    with pytest.raises(ValueError):
        sessao.aponta_dono(bag_id, *ponto)

    fila = sessao.queue()
    assert [(i.person, i.top_level) for i in fila] == [(LADRAO, FlagLevel.N3)]


def test_dono_em_bagagem_nao_terminal_continua_aceito():
    """Regressão: com a custódia em aberto, apontar o dono segue igual."""
    quadros = casal_andando()
    quadros += [[pessoa(1, 10.0), pessoa(2, 10.5), mala(MALA, 10.2)]] * 3
    sessao = LiveSession(PLANO, RAPIDO)
    ultimo = _alimenta(sessao, quadros)
    bagagem = sessao._s.registry.get(MALA)
    assert bagagem.state not in TERMINAL_BAG_STATES

    devolvida = sessao.aponta_dono(MALA, *_centro(ultimo, 1))

    assert devolvida is bagagem
    assert bagagem.owner_party == sessao.party_of(1)
    posse = [e for e in sessao.events.of_kind(EventKind.BAG_OWNED) if e.bag == MALA]
    assert posse[-1].evidence.get("origem") == "operador"


def test_apontar_bagagem_sobre_custodia_encerrada_nao_a_reabre():
    """`aponta_bagagem` sempre cria âncora nova, com id negativo: clicar onde
    estava a mala levada não toca na custódia encerrada nem na fila."""
    sessao, bag_id, _ = _estranho_recolhe_a_mala_do_casal()
    encerrada = sessao._s.registry.get(bag_id)
    antes = _retrato(sessao, encerrada)

    nova = sessao.aponta_bagagem(encerrada.anchor.x, encerrada.anchor.y)

    assert nova.bag_id != bag_id
    assert nova.owner_party is None
    depois = _retrato(sessao, encerrada)
    assert depois["eventos"] == antes["eventos"] + 1  # só o BAG_APPEARED da nova
    assert {k: v for k, v in depois.items() if k != "eventos"} == {
        k: v for k, v in antes.items() if k != "eventos"
    }


def test_posse_tardia_nao_apaga_flags_de_custodia_encerrada():
    """Defesa em profundidade, pelo caminho automático.

    A mala órfã é levada (N3), devolvida ao lugar dentro da janela de posse, e
    `claim_ownership` dá a posse a quem a levou. O N3 descreve a relação da
    pessoa com a bagagem no instante da retirada (P4); uma posse que chega
    depois não muda o que aconteceu ali, e apagá-lo reescreveria a história
    de uma custódia já decidida.
    """
    plano = GroundPlane.uniform(0.01)

    def pessoa_px(tid, fx):
        return TrackedDetection(tid, "person", (fx - 20, 330, fx + 20, 500))

    def mala_px(fx):
        return TrackedDetection(100, "suitcase", (fx - 20, 455, fx + 20, 500))

    sessao = LiveSession(plano, Config())
    for i in range(120):
        t = i / 10
        if t < 3:
            dets = [mala_px(550)]
        elif t < 4:
            dets = [pessoa_px(LADRAO, 600)]
        elif t < 6:
            dets = [pessoa_px(LADRAO, 700), mala_px(730)]
        else:
            dets = [pessoa_px(LADRAO, 570), mala_px(550)]
        sessao.feed(t, dets)

    bagagem = sessao.bag_for_track(100)
    assert bagagem.state is BagState.RETIRADA_ESTRANHO
    assert FlagLevel.N3 in [f.level for f in sessao._s.flags.for_person(LADRAO)]
    assert [(i.person, i.top_level) for i in sessao.queue()] == [(LADRAO, FlagLevel.N3)]


def test_orfa_levada_nao_ganha_dono_ao_voltar_para_a_ancora():
    """`claim_ownership` é a rede para a órfã cujo dono o detector perdeu no
    depósito; não é para quem já a levou. Devolvida ao lugar dentro da janela
    de posse, a bagagem seguia órfã e a rede a dava a quem estivesse perto --
    o próprio ladrão --, com `BAG_OWNED` sobre uma custódia encerrada.
    """
    plano = GroundPlane.uniform(0.01)

    def pessoa_px(tid, fx):
        return TrackedDetection(tid, "person", (fx - 20, 330, fx + 20, 500))

    def mala_px(fx):
        return TrackedDetection(100, "suitcase", (fx - 20, 455, fx + 20, 500))

    sessao = LiveSession(plano, Config())
    retrato = None
    for i in range(120):
        t = i / 10
        if t < 3:
            dets = [mala_px(550)]
        elif t < 4:
            dets = [pessoa_px(LADRAO, 600)]
        elif t < 6:
            dets = [pessoa_px(LADRAO, 700), mala_px(730)]
        else:
            dets = [pessoa_px(LADRAO, 570), mala_px(550)]
        sessao.feed(t, dets)
        bagagem = sessao.bag_for_track(100)
        if retrato is None and bagagem is not None and bagagem.state in TERMINAL_BAG_STATES:
            retrato = {
                "estado": bagagem.state,
                "dono": bagagem.owner_party,
                "fila": [(a.person, a.top_level) for a in sessao.queue()],
                "flags": {p: sessao._s.flags.for_person(p) for p in sessao._s.flags.people()},
                "t": t,
            }

    assert retrato is not None and retrato["estado"] is BagState.RETIRADA_ESTRANHO
    assert retrato["dono"] is None
    assert retrato["t"] < Config().custody.ownership_window_s, "precisa voltar dentro da janela"
    bagagem = sessao.bag_for_track(100)
    assert bagagem.state is retrato["estado"]
    assert bagagem.owner_party is None
    assert not [e for e in sessao.events.of_kind(EventKind.BAG_OWNED) if e.bag == bagagem.bag_id]
    assert [(a.person, a.top_level) for a in sessao.queue()] == retrato["fila"]
    assert {p: sessao._s.flags.for_person(p) for p in sessao._s.flags.people()} == retrato["flags"]

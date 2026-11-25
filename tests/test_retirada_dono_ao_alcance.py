"""Retirada em plena vista com o dono ao alcance.

`carry_away` escolhia o carregador pela pessoa MAIS PRÓXIMA da bagagem. Numa
fila, o dono puxa a própria mala e quem vem atrás fica alguns centímetros mais
perto dela: o sistema emitia `bag_removed_by_stranger` e N3 contra o vizinho
(auditoria de 2026-11-24, `repro_ambig.py`, caso 2).

Enquanto um membro de vínculo forte do grupo dono está ao alcance da posição
ATUAL da mala, ele pode ser quem a está levando, e acusar outro é chutar -- P3.
A custódia continua, como já continuava quando o dono era o mais próximo.
Quando nenhum membro forte acompanha mais a mala, decide-se como antes -- e é
isso que mantém visível o furto canônico, em que o dono fica sentado e a mala
se afasta dele. O empate exato na menor distância, sem dono ao alcance, vira
AMBIGUA: escolher pela ordem da lista é desempate arbitrário.

As cenas são funções do tempo, com a geometria derivada da `Config`, e cada
uma confere a própria premissa em todo quadro da janela de decisão antes de
afirmar o desfecho. Trajetórias sintéticas: provam a lógica, não a percepção.
"""

from collections.abc import Callable, Iterator

import pytest

from custody_watch.config import Config
from custody_watch.events import Event, EventKind
from custody_watch.ground_plane import GroundPlane
from custody_watch.orchestrator import SessionResult, run_session
from custody_watch.tracking import TrackedDetection
from custody_watch.types import FlagLevel, Point

CONFIG = Config()
RAIO = CONFIG.pipeline.owner_search_radius_m
LIMIAR_MOVIMENTO = CONFIG.registry.moved_threshold_m
CONFIRMA = CONFIG.custody.carry_confirm_s
PROXIMIDADE_GRUPO = CONFIG.party.proximity_m
VINCULO_FRACO = CONFIG.party.weak_bond_s
EXTENSAO_ENTRADA_TARDIA = CONFIG.party.late_join_extent_m

PASSO = 0.1
"""10 Hz: a espera de `carry_confirm_s` atravessa vários quadros."""

PLANO = GroundPlane.uniform(0.01)
"""1 px = 1 cm, sem perspectiva: o que está sob teste é a custódia."""

DONO, ESTRANHO, OUTRO, FRACO = 1, 2, 3, 4
MALA = 100

X0, Y0 = 5.0, 5.0
"""Onde a mala é depositada, em metros."""

VEL = 1.0
"""Passo de quem puxa a mala, em m/s."""

JANELA = LIMIAR_MOVIMENTO / VEL + CONFIRMA + 1.0
"""Do início do puxão até depois da decisão: a âncora precisa sair do limiar,
o deslocamento precisa se sustentar por `carry_confirm_s`, e sobra um segundo
de folga para quadros e arredondamento."""

Quadro = list[TrackedDetection]
Cena = Callable[[float], Quadro]


def pessoa(tid: int, x: float, y: float = Y0) -> TrackedDetection:
    """Pessoa em pé com os pés em (x, y) metros."""
    cx, base = x * 100.0, y * 100.0
    return TrackedDetection(tid, "person", (cx - 20.0, base - 170.0, cx + 20.0, base))


def mala(x: float, y: float = Y0) -> TrackedDetection:
    """Mala no chão, com a base em (x, y) metros e 45 px de altura."""
    cx, base = x * 100.0, y * 100.0
    return TrackedDetection(MALA, "suitcase", (cx - 20.0, base - 45.0, cx + 20.0, base))


def rodar(cena: Cena, duracao_s: float) -> SessionResult:
    def quadros() -> Iterator[tuple[float, Quadro]]:
        for i in range(round(duracao_s / PASSO)):
            t = i * PASSO
            yield t, cena(t)

    return run_session(quadros(), PLANO, CONFIG)


def posicoes(quadro: Quadro) -> dict[int, Point]:
    return {d.track_id: PLANO.foot_point(d.bbox) for d in quadro}


def distancias_a_mala(cena: Cena, t: float) -> dict[int, float]:
    """Distância de cada pessoa à mala no quadro `t`, como o pipeline a mede."""
    lugar = posicoes(cena(t))
    alvo = lugar[MALA]
    return {tid: p.distance_to(alvo) for tid, p in lugar.items() if tid != MALA}


def janela_de_decisao(inicio_puxao: float) -> list[float]:
    passos = round(JANELA / PASSO)
    return [inicio_puxao + i * PASSO for i in range(passos + 1)]


def de_tipo(resultado: SessionResult, kind: EventKind) -> list[Event]:
    return [e for e in resultado.events if e.kind is kind]


def entradas_fortes(resultado: SessionResult, tid: int) -> list[Event]:
    return [e for e in de_tipo(resultado, EventKind.PARTY_JOINED_STRONG) if e.subject == tid]


def retiradas_contra(resultado: SessionResult, tid: int) -> list:
    return [f for f in resultado.flags.for_person(tid) if f.kind == "retirada_por_estranho"]


def n3_na_fila(resultado: SessionResult) -> set[int]:
    return {item.person for item in resultado.queue if item.top_level is FlagLevel.N3}


def assert_dono_forte_da_mala(resultado: SessionResult, tid: int) -> int:
    """Premissa: `tid` depositou a mala e é membro forte do grupo dono."""
    (posse,) = de_tipo(resultado, EventKind.BAG_OWNED)
    formados = [e for e in de_tipo(resultado, EventKind.PARTY_FORMED) if e.party == posse.party]
    assert formados, "o grupo dono nunca foi formado"
    assert tid in formados[0].evidence["members"]
    return posse.party


# --- (a) o defeito: dono forte puxa, estranho mais perto -----------------------

T_PUXA_FILA = 20.0
D_DONO_FILA = RAIO / 5
D_ESTRANHO_FILA = RAIO / 10


def fila(t: float) -> Quadro:
    """O dono deposita a mala e fica ao lado; aos 15s alguém entra na fila
    atrás dela, MAIS PERTO da mala que o dono. Aos 20s o dono puxa a mala e a
    fila anda junto, na mesma geometria."""
    dx = max(0.0, t - T_PUXA_FILA) * VEL
    quadro = [pessoa(DONO, X0 - D_DONO_FILA - dx), mala(X0 - dx)]
    if t >= 15.0:
        quadro.append(pessoa(ESTRANHO, X0 + D_ESTRANHO_FILA - dx))
    return quadro


def test_dono_forte_puxando_a_mala_com_estranho_mais_perto_nao_acusa_o_estranho():
    """Medido antes da correção: `bag_removed_by_stranger` contra o vizinho
    de fila aos ~21.5s, com N3 na fila do operador.

    O dono acompanha a mala, ao alcance dela em todo quadro: é custódia
    continuando. Nem acusação, nem AMBIGUA -- a bagagem segue viva para que
    a retirada pelo dono possa ser resolvida depois (ver `fila_longa`)."""
    duracao = T_PUXA_FILA + JANELA + 1.0
    # A fila anda junto por menos que a entrada tardia exige: o estranho não
    # pode virar membro forte do grupo dono durante a cena.
    assert (duracao - T_PUXA_FILA) * VEL < EXTENSAO_ENTRADA_TARDIA
    for t in janela_de_decisao(T_PUXA_FILA):
        d = distancias_a_mala(fila, t)
        assert d[ESTRANHO] < d[DONO] <= RAIO, (t, d)

    resultado = rodar(fila, duracao)

    assert_dono_forte_da_mala(resultado, DONO)
    assert entradas_fortes(resultado, ESTRANHO) == []
    assert de_tipo(resultado, EventKind.BAG_REMOVED_BY_STRANGER) == []
    assert de_tipo(resultado, EventKind.BAG_REMOVED_BY_OWNER) == []
    assert retiradas_contra(resultado, ESTRANHO) == []
    assert ESTRANHO not in n3_na_fila(resultado)
    assert de_tipo(resultado, EventKind.BAG_AMBIGUOUS) == []


# --- dono acompanhando a mala por muito tempo, depois saindo com ela -----------

VEL_FILA = 0.2
"""Fila que anda devagar: a âncora acompanha em degraus de `moved_threshold_m`,
então `carry_away` roda de novo a cada degrau, por toda a cena."""

T_FIM_FILA = T_PUXA_FILA + 30.0
OCLUSAO = CONFIG.custody.max_occlusion_s
ATE_OCLUIR_S = CONFIG.pipeline.missing_before_occluded_s + PASSO
"""O limiar de oclusão decide até um quadro depois de vencer."""


def fila_longa(t: float) -> Quadro:
    """Como `fila`, mas a fila anda por 30s com o estranho sempre mais perto
    da mala. Aos `T_FIM_FILA` o estranho sai de cena e a mala some (o dono a
    põe na esteira); o dono fica onde estava."""
    dx = (min(t, T_FIM_FILA) - T_PUXA_FILA) * VEL_FILA if t > T_PUXA_FILA else 0.0
    quadro = [pessoa(DONO, X0 - D_DONO_FILA - dx)]
    if t < T_FIM_FILA:
        quadro.append(mala(X0 - dx))
        if t >= 15.0:
            quadro.append(pessoa(ESTRANHO, X0 + D_ESTRANHO_FILA - dx))
    return quadro


def test_dono_acompanhando_a_mala_por_muito_tempo_ainda_retira_a_propria_mala():
    """A custódia que continua não pode prender a bagagem num estado sem
    saída, nem gerar evento a cada degrau da âncora: quando a mala some com só
    o dono ao alcance, o timeout de oclusão resolve como retirada pelo dono."""
    # A fila anda devagar demais para a entrada tardia: na janela de
    # histórico o estranho não cobre a extensão que o faria membro forte.
    assert VEL_FILA * CONFIG.pipeline.history_window_s < EXTENSAO_ENTRADA_TARDIA
    for i in range(round((T_FIM_FILA - T_PUXA_FILA) / PASSO)):
        t = T_PUXA_FILA + i * PASSO
        d = distancias_a_mala(fila_longa, t)
        assert d[ESTRANHO] < d[DONO] <= RAIO, (t, d)

    duracao = T_FIM_FILA + ATE_OCLUIR_S + OCLUSAO + 2.0
    resultado = rodar(fila_longa, duracao)

    assert_dono_forte_da_mala(resultado, DONO)
    assert entradas_fortes(resultado, ESTRANHO) == []
    custodia = [
        e.kind
        for e in resultado.events
        if e.kind
        in {
            EventKind.BAG_REMOVED_BY_OWNER,
            EventKind.BAG_REMOVED_BY_STRANGER,
            EventKind.BAG_AMBIGUOUS,
            EventKind.BAG_UNATTENDED,
            EventKind.BAG_REATTENDED,
        }
    ]
    assert custodia == [EventKind.BAG_REMOVED_BY_OWNER]
    (retirada,) = de_tipo(resultado, EventKind.BAG_REMOVED_BY_OWNER)
    assert retirada.subject == DONO
    assert retirada.t_start >= T_FIM_FILA + OCLUSAO
    assert n3_na_fila(resultado) == set()


# --- o furto canônico: o dono fica sentado, a mala se afasta dele --------------

D_DONO_LADO = RAIO / 5
T_PUXA_LADO = 20.0
SAIDA_DO_ALCANCE = (RAIO - D_DONO_LADO) / VEL
"""Quanto tempo a mala leva, levada pelo estranho, para sair do alcance do dono
sentado. Até ali o dono PODE ser quem a leva, e a decisão espera."""

ATRASO_MAX_LADO = SAIDA_DO_ALCANCE + LIMIAR_MOVIMENTO / VEL + CONFIRMA + 2 * PASSO
"""Atraso máximo da acusação, contado do início do puxão: sair do alcance do
dono, a âncora (que acompanhou a mala em degraus de `moved_threshold_m`
enquanto o dono a alcançava) ficar para trás de novo, e a espera de
`carry_confirm_s`, mais dois quadros de quantização. Com a `Config` padrão:
2.4 + 0.5 + 1.0 + 0.2 = 4.1s, contra 1.5s quando o dono está fora do raio
desde o início."""


def furto_ao_lado(t: float) -> Quadro:
    """O dono deposita a mala e fica PARADO ao lado dela a cena inteira. O
    estranho encosta aos 15s e aos 20s leva a mala embora, para longe do dono."""
    dx = max(0.0, t - T_PUXA_LADO) * VEL
    quadro = [pessoa(DONO, X0 - D_DONO_LADO), mala(X0 + dx)]
    if t >= T_CHEGA_ESTRANHO:
        quadro.append(pessoa(ESTRANHO, X0 + D_PERTO + dx))
    return quadro


def test_dono_sentado_e_mala_levada_por_estranho_continua_acusacao():
    """ "O ladrão senta ao seu lado": o dono está dentro do raio quando a mala
    começa a andar, mas não a acompanha. Medido com a regra anterior desta
    branch (membro forte no raio no instante da confirmação => AMBIGUA): o
    furto ficava mudo."""
    assert distancias_a_mala(furto_ao_lado, T_PUXA_LADO)[DONO] <= RAIO
    fora = T_PUXA_LADO + SAIDA_DO_ALCANCE + PASSO
    assert distancias_a_mala(furto_ao_lado, fora)[DONO] > RAIO
    quadros = round(ATRASO_MAX_LADO / PASSO)
    for t in [T_PUXA_LADO + i * PASSO for i in range(quadros + 1)]:
        d = distancias_a_mala(furto_ao_lado, t)
        assert d[ESTRANHO] < d[DONO], (t, d)
        assert posicoes(furto_ao_lado(t))[DONO] == posicoes(furto_ao_lado(0.0))[DONO]

    resultado = rodar(furto_ao_lado, T_PUXA_LADO + ATRASO_MAX_LADO + 1.0)

    assert_dono_forte_da_mala(resultado, DONO)
    (furto,) = de_tipo(resultado, EventKind.BAG_REMOVED_BY_STRANGER)
    assert furto.subject == ESTRANHO
    assert T_PUXA_LADO + CONFIRMA <= furto.t_start <= T_PUXA_LADO + ATRASO_MAX_LADO
    assert ESTRANHO in n3_na_fila(resultado)
    assert de_tipo(resultado, EventKind.BAG_AMBIGUOUS) == []


D_DONO_BORDA = RAIO - LIMIAR_MOVIMENTO / 2
ATRASO_SEM_DONO = LIMIAR_MOVIMENTO / VEL + CONFIRMA + 2 * PASSO
"""Atraso de quando nenhum membro forte alcança a mala: sair do limiar de
movimento e sustentar por `carry_confirm_s`, mais dois quadros."""


def furto_com_dono_na_borda(t: float) -> Quadro:
    """Como `furto_ao_lado`, com o dono parado quase no limite do raio, do
    lado oposto ao da fuga: ele alcança a âncora, mas não a mala que já saiu
    do lugar."""
    dx = max(0.0, t - T_PUXA_LADO) * VEL
    quadro = [pessoa(DONO, X0 - D_DONO_BORDA), mala(X0 + dx)]
    if t >= T_CHEGA_ESTRANHO:
        quadro.append(pessoa(ESTRANHO, X0 + D_PERTO + dx))
    return quadro


def test_alcance_do_dono_e_medido_da_mala_e_nao_da_ancora():
    """Medido da âncora, o dono que ficou para trás ainda "acompanharia" a
    mala por um degrau a mais, e a acusação atrasaria sem motivo."""
    assert D_DONO_BORDA <= RAIO < D_DONO_BORDA + LIMIAR_MOVIMENTO
    resultado = rodar(furto_com_dono_na_borda, T_PUXA_LADO + ATRASO_MAX_LADO + 1.0)

    assert_dono_forte_da_mala(resultado, DONO)
    (furto,) = de_tipo(resultado, EventKind.BAG_REMOVED_BY_STRANGER)
    assert furto.subject == ESTRANHO
    assert T_PUXA_LADO + CONFIRMA <= furto.t_start <= T_PUXA_LADO + ATRASO_SEM_DONO


# --- (b) o furto base continua visto ----------------------------------------

T_SAIDA_DONO = 5.0
T_CHEGA_ESTRANHO = 15.0
T_PUXA_FURTO = 20.0
D_PERTO = RAIO / 10
LONGE = 2 * RAIO


def dono_que_sai(t: float) -> float:
    """Deposita à direita da mala e vai embora para a direita, até 2 raios."""
    return X0 + D_PERTO + min(max(0.0, t - T_SAIDA_DONO) * VEL, LONGE)


def furto_sem_dono(t: float) -> Quadro:
    """O dono deposita e sai para a direita. O estranho chega pela esquerda e
    leva a mala para a esquerda, para longe do dono."""
    dx = max(0.0, t - T_PUXA_FURTO) * VEL
    quadro = [pessoa(DONO, dono_que_sai(t)), mala(X0 - dx)]
    if t >= T_CHEGA_ESTRANHO:
        quadro.append(pessoa(ESTRANHO, X0 - D_PERTO - dx))
    return quadro


def test_estranho_leva_a_mala_com_o_dono_fora_do_raio_continua_acusado():
    """O espelho: a correção não pode cegar o sistema para o furto base."""
    for t in janela_de_decisao(T_PUXA_FURTO):
        d = distancias_a_mala(furto_sem_dono, t)
        assert d[ESTRANHO] <= RAIO < d[DONO], (t, d)

    resultado = rodar(furto_sem_dono, T_PUXA_FURTO + JANELA + 1.0)

    assert_dono_forte_da_mala(resultado, DONO)
    (furto,) = de_tipo(resultado, EventKind.BAG_REMOVED_BY_STRANGER)
    assert furto.subject == ESTRANHO
    assert T_PUXA_FURTO + CONFIRMA <= furto.t_start <= T_PUXA_FURTO + JANELA
    assert len(retiradas_contra(resultado, ESTRANHO)) == 1
    assert ESTRANHO in n3_na_fila(resultado)
    assert de_tipo(resultado, EventKind.BAG_AMBIGUOUS) == []


# --- (c) vínculo fraco não é dono (P1) ---------------------------------------

T_SAIDA_DONO_FRACO = VINCULO_FRACO + 5.0
T_CHEGA_ESTRANHO_FRACO = T_SAIDA_DONO_FRACO + 10.0
T_PUXA_FRACO = T_CHEGA_ESTRANHO_FRACO + 5.0
D_FRACO = RAIO / 6


def furto_com_fraco_ao_lado(t: float) -> Quadro:
    """Alguém senta à esquerda da mala, perto do dono, por mais que
    `weak_bond_s`: vira membro FRACO do grupo dono. O dono sai para a direita;
    um estranho chega pela direita e leva a mala para a direita, com o membro
    fraco ainda ao alcance -- mas mais longe da mala que o estranho."""
    dx = max(0.0, t - T_PUXA_FRACO) * VEL
    dono_x = X0 + D_PERTO + min(max(0.0, t - T_SAIDA_DONO_FRACO) * VEL, LONGE)
    quadro = [pessoa(DONO, dono_x), pessoa(FRACO, X0 - D_FRACO), mala(X0 + dx)]
    if t >= T_CHEGA_ESTRANHO_FRACO:
        quadro.append(pessoa(ESTRANHO, X0 + D_PERTO + dx))
    return quadro


def test_so_membro_fraco_ao_alcance_nao_impede_a_acusacao():
    """P1: `Party.owns` exige vínculo forte. Quem só sentou perto pertence ao
    grupo para `party_of`, mas não legitima a retirada -- e por isso também
    não torna incerto quem a levou."""
    assert abs((X0 + D_PERTO) - (X0 - D_FRACO)) <= PROXIMIDADE_GRUPO
    for t in janela_de_decisao(T_PUXA_FRACO):
        d = distancias_a_mala(furto_com_fraco_ao_lado, t)
        assert d[ESTRANHO] < d[FRACO] <= RAIO < d[DONO], (t, d)

    resultado = rodar(furto_com_fraco_ao_lado, T_PUXA_FRACO + JANELA + 1.0)

    party = assert_dono_forte_da_mala(resultado, DONO)
    (fraco,) = [e for e in de_tipo(resultado, EventKind.PARTY_JOINED_WEAK) if e.subject == FRACO]
    assert fraco.party == party
    assert fraco.t_start < T_PUXA_FRACO
    assert entradas_fortes(resultado, FRACO) == []

    (furto,) = de_tipo(resultado, EventKind.BAG_REMOVED_BY_STRANGER)
    assert furto.subject == ESTRANHO
    assert ESTRANHO in n3_na_fila(resultado)
    assert de_tipo(resultado, EventKind.BAG_AMBIGUOUS) == []


# --- (d) empate entre dois estranhos ----------------------------------------

D_EMPATE = RAIO / 6


def dois_estranhos_empatados(t: float) -> Quadro:
    """O dono deposita e sai. Dois estranhos ladeiam a mala, um de cada lado,
    à MESMA distância, e ela anda entre os dois -- perpendicular ao eixo que os
    separa, então o empate se mantém quadro a quadro."""
    dy = max(0.0, t - T_PUXA_FURTO) * VEL
    quadro = [pessoa(DONO, dono_que_sai(t)), mala(X0, Y0 + dy)]
    if t >= T_CHEGA_ESTRANHO:
        quadro.append(pessoa(ESTRANHO, X0 - D_EMPATE, Y0 + dy))
        quadro.append(pessoa(OUTRO, X0 + D_EMPATE, Y0 + dy))
    return quadro


def test_dois_estranhos_equidistantes_nao_sao_desempatados_pela_ordem_da_lista():
    """`min` devolve o primeiro empatado, e o primeiro depende só da ordem em
    que o detector listou as caixas. Acusar um deles é desempate arbitrário."""
    for t in janela_de_decisao(T_PUXA_FURTO):
        d = distancias_a_mala(dois_estranhos_empatados, t)
        assert d[ESTRANHO] == d[OUTRO] <= RAIO < d[DONO], (t, d)

    resultado = rodar(dois_estranhos_empatados, T_PUXA_FURTO + JANELA + 1.0)

    assert_dono_forte_da_mala(resultado, DONO)
    assert de_tipo(resultado, EventKind.BAG_REMOVED_BY_STRANGER) == []
    assert retiradas_contra(resultado, ESTRANHO) == []
    assert retiradas_contra(resultado, OUTRO) == []
    (ambigua,) = de_tipo(resultado, EventKind.BAG_AMBIGUOUS)
    assert ambigua.evidence["candidate_people"] == sorted([ESTRANHO, OUTRO])


# --- (e) regressões: o dono levando a própria mala ---------------------------


def dono_sozinho(t: float) -> Quadro:
    dx = max(0.0, t - T_PUXA_FILA) * VEL
    return [pessoa(DONO, X0 - D_DONO_FILA - dx), mala(X0 - dx)]


def dono_na_frente_da_fila(t: float) -> Quadro:
    """Como `fila`, com os papéis de distância trocados: o dono é o mais
    próximo, e o estranho segue atrás, ao alcance."""
    dx = max(0.0, t - T_PUXA_FILA) * VEL
    quadro = [pessoa(DONO, X0 - D_ESTRANHO_FILA - dx), mala(X0 - dx)]
    if t >= 15.0:
        quadro.append(pessoa(ESTRANHO, X0 + D_DONO_FILA - dx))
    return quadro


@pytest.mark.parametrize("cena", [dono_sozinho, dono_na_frente_da_fila])
def test_dono_mais_proximo_levando_a_propria_mala_e_custodia_continuando(cena: Cena):
    """Membro forte movendo a bagagem é custódia continuando, não terminando:
    nem retirada, nem AMBIGUA, nem fila."""
    for t in janela_de_decisao(T_PUXA_FILA):
        d = distancias_a_mala(cena, t)
        assert d[DONO] == min(d.values()), (t, d)
        assert all(v <= RAIO for v in d.values()), (t, d)

    resultado = rodar(cena, T_PUXA_FILA + JANELA + 1.0)

    assert_dono_forte_da_mala(resultado, DONO)
    assert de_tipo(resultado, EventKind.BAG_REMOVED_BY_STRANGER) == []
    assert de_tipo(resultado, EventKind.BAG_REMOVED_BY_OWNER) == []
    assert de_tipo(resultado, EventKind.BAG_AMBIGUOUS) == []
    assert n3_na_fila(resultado) == set()

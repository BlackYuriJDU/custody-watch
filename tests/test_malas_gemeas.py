"""Malas gêmeas: dois tracks de bagagem voltam trocados de uma oclusão.

Duas malas idênticas paradas somem juntas e cada uma reaparece exatamente na
própria âncora, mas o rastreador devolve os track ids trocados -- o que
aparência nenhuma impede, porque as malas são iguais. A regra P2 diz que
bagagem parada é identificada pela âncora espacial, não pelo id: cada track
passa a responder pela bagagem cuja âncora ele ocupa. Com o id mandando, o
sistema lia as duas malas como deslocadas e, quando cada dono levava a sua,
acusava os dois de levar a do outro (N3 indevido).

Quando o lugar onde o track reaparece cabe em mais de uma âncora ocluída, a
regra P3 manda marcar AMBIGUA e calar, como na readoção de track novo.

Trajetórias sintéticas escritas à mão, como em `test_readocao_ambigua.py`:
provam que a lógica decide certo, não que a percepção enxerga. A geometria é
derivada dos raios da config, e cada cena confere a própria premissa.
"""

import numpy as np

from custody_watch.config import Config
from custody_watch.events import EventKind
from custody_watch.ground_plane import GroundPlane
from custody_watch.orchestrator import LiveSession
from custody_watch.tracking import TrackedDetection
from custody_watch.types import TERMINAL_BAG_STATES, Bag, BagState, FlagLevel, Point

FPS = 25.0
CONFIG = Config()
REGISTRO = CONFIG.registry
PIPELINE = CONFIG.pipeline

# Câmera de cima, 1px = 1cm, igual a test_occlusion.py.
PLANO = GroundPlane(np.diag([0.01, 0.01, 1.0]))

DONO_A, DONO_B = 1, 2
MALA_A, MALA_B = 101, 102

# As malas assentam (rest_confirm_s) e somem aos 3s; voltam aos 5s, com 2s de
# ausência -- bem acima de `missing_before_occluded_s`. Os donos levam cada
# um a sua aos 7s e andam para fora até o fim.
SOMEM_EM = 3.0
VOLTAM_EM = 5.0
LEVAM_EM = 7.0
FIM = 13.0
VELOCIDADE_MS = 1.5

PONTO_A = Point(10.0, 10.0)
# Vão de malas gêmeas do benchmark: cada âncora fica fora do raio de readoção
# e do de ambiguidade da outra, então a posição decide sem dúvida.
VAO_GEMEAS = 1.5 * max(REGISTRO.reidentification_radius_m, REGISTRO.ambiguity_radius_m)
PONTO_B = Point(PONTO_A.x + VAO_GEMEAS, PONTO_A.y)
# Vão ambíguo: a outra âncora cabe no raio de readoção do lugar onde o track
# reaparece, e o track está deslocado da própria âncora além do limiar.
VAO_AMBIGUO = 0.8 * min(REGISTRO.reidentification_radius_m, REGISTRO.ambiguity_radius_m)
PONTO_B_AMBIGUO = Point(PONTO_A.x + VAO_AMBIGUO, PONTO_A.y)

# Cada dono fica de lado e um metro abaixo da própria mala: fora da silhueta
# (o veto "carregada" barraria a âncora) e a mais de `party.proximity_m` do
# outro dono, para ninguém virar grupo de ninguém.
AFASTAMENTO_X, AFASTAMENTO_Y = 0.7, 1.0


def caixa(x_m: float, y_m: float, altura_px: float):
    """Caixa cujo pé cai em (x_m, y_m)."""
    cx, base = x_m * 100.0, y_m * 100.0
    return (cx - altura_px / 4, base - altura_px, cx + altura_px / 4, base)


def mala(track_id: int, p: Point) -> TrackedDetection:
    return TrackedDetection(track_id, "suitcase", caixa(p.x, p.y, 55))


def pessoa(track_id: int, p: Point) -> TrackedDetection:
    return TrackedDetection(track_id, "person", caixa(p.x, p.y, 170))


def lugar_do_dono(ancora: Point, sentido: float) -> Point:
    """Onde o dono espera: do lado de fora do par, `sentido` -1 ou +1."""
    return Point(ancora.x + sentido * AFASTAMENTO_X, ancora.y + AFASTAMENTO_Y)


def andando(inicio: Point, sentido: float, t: float, desde: float) -> Point:
    """Posição de quem parte de `inicio` em `desde`, andando no eixo x."""
    passo = max(0.0, t - desde) * VELOCIDADE_MS
    return Point(inicio.x + sentido * passo, inicio.y)


def gemeas_trocadas(ponto_b: Point, *, b_volta_em: float = VOLTAM_EM, fim: float = FIM):
    """Malas em PONTO_A e `ponto_b`, cada dono ao lado da sua. Somem juntas;
    `MALA_A` volta sobre `ponto_b` e `MALA_B` sobre PONTO_A -- cada mala na
    própria âncora, com o id da outra. Em `LEVAM_EM` cada dono sai para o
    seu lado levando a mala que está na âncora dele."""
    dono_a, dono_b = lugar_do_dono(PONTO_A, -1.0), lugar_do_dono(ponto_b, +1.0)
    for i in range(int(fim * FPS)):
        t = i / FPS
        pa, pb = andando(dono_a, -1.0, t, LEVAM_EM), andando(dono_b, +1.0, t, LEVAM_EM)
        det = [pessoa(DONO_A, pa), pessoa(DONO_B, pb)]
        # A mala acompanha o dono com o mesmo afastamento em que foi deixada.
        na_ancora_a = Point(pa.x + AFASTAMENTO_X, pa.y - AFASTAMENTO_Y)
        na_ancora_b = Point(pb.x - AFASTAMENTO_X, pb.y - AFASTAMENTO_Y)
        if t < SOMEM_EM:
            det += [mala(MALA_A, na_ancora_a), mala(MALA_B, na_ancora_b)]
        if t >= VOLTAM_EM:
            det.append(mala(MALA_A, na_ancora_b))
        if t >= b_volta_em:
            det.append(mala(MALA_B, na_ancora_a))
        yield t, det


def roda(frames) -> LiveSession:
    sessao = LiveSession(PLANO, CONFIG)
    for t, det in frames:
        sessao.feed(t, det)
    return sessao


def bagagem_do_track(sessao: LiveSession, track_id: int) -> Bag:
    bag = sessao.bag_for_track(track_id)
    assert bag is not None, f"track {track_id} nunca virou bagagem"
    return bag


def religacoes_de_bagagem(sessao: LiveSession) -> list[tuple[int, int]]:
    """(track bruto, bagagem) de cada `TRACK_RELINKED` de bagagem."""
    return sorted(
        (e.evidence["from_track"], e.bag)
        for e in sessao.events.of_kind(EventKind.TRACK_RELINKED)
        if e.bag is not None
    )


def acusacoes_n3(sessao: LiveSession) -> list:
    return [item for item in sessao.queue() if item.top_level is FlagLevel.N3]


# --- premissas ----------------------------------------------------------------


def test_premissa_das_gemeas_sem_ambiguidade():
    """Cada âncora fora do raio de readoção e do de ambiguidade da outra, e o
    deslocamento aparente do track além do limiar de movimento: sem isso a
    cena não exercita a troca."""
    vao = PONTO_A.distance_to(PONTO_B)
    assert vao > REGISTRO.reidentification_radius_m
    assert vao > REGISTRO.ambiguity_radius_m
    assert vao >= REGISTRO.moved_threshold_m


def test_premissa_das_gemeas_ambiguas():
    """A outra âncora cabe no raio de readoção do ponto onde o track volta, e
    o track está deslocado da própria âncora além do limiar de movimento."""
    vao = PONTO_A.distance_to(PONTO_B_AMBIGUO)
    assert vao <= REGISTRO.reidentification_radius_m
    assert vao <= REGISTRO.ambiguity_radius_m
    assert vao >= REGISTRO.moved_threshold_m


def test_premissa_da_ausencia_e_dos_donos():
    """A ausência passa de `missing_before_occluded_s` com um quadro de folga,
    e cada dono fica longe da mala do outro e do outro dono."""
    ausencia_s = VOLTAM_EM - SOMEM_EM
    assert ausencia_s > PIPELINE.missing_before_occluded_s + 1.0 / FPS
    for ponto_b in (PONTO_B, PONTO_B_AMBIGUO):
        dono_a, dono_b = lugar_do_dono(PONTO_A, -1.0), lugar_do_dono(ponto_b, +1.0)
        assert dono_a.distance_to(dono_b) > CONFIG.party.proximity_m
        assert dono_a.distance_to(ponto_b) > PIPELINE.contact_radius_m
        assert dono_b.distance_to(PONTO_A) > PIPELINE.contact_radius_m


# --- a posição decide quem é quem ---------------------------------------------


def test_tracks_trocados_respondem_pela_ancora_onde_reaparecem():
    """Lido antes de os donos levarem as malas: cada track responde pela
    bagagem cuja âncora ocupa, não pela que o id indicava."""
    sessao = roda(gemeas_trocadas(PONTO_B, fim=LEVAM_EM))

    assert bagagem_do_track(sessao, MALA_A).bag_id == MALA_B
    assert bagagem_do_track(sessao, MALA_B).bag_id == MALA_A
    assert religacoes_de_bagagem(sessao) == [(MALA_A, MALA_B), (MALA_B, MALA_A)]


def test_gemeas_trocadas_ficam_paradas_e_resolvidas():
    """Nenhuma das duas saiu do lugar: a âncora de cada bagagem fica onde ela
    foi deixada -- não viaja para a da vizinha atrás do id -- e a custódia
    segue aberta, sem ambiguidade nem retirada."""
    sessao = roda(gemeas_trocadas(PONTO_B, fim=LEVAM_EM))

    malas = {bag.bag_id: bag for bag in (bagagem_do_track(sessao, t) for t in (MALA_A, MALA_B))}
    for bag_id, ancora in ((MALA_A, PONTO_A), (MALA_B, PONTO_B)):
        bag = malas[bag_id]
        assert bag.anchor.distance_to(ancora) < REGISTRO.moved_threshold_m
        assert bag.state is not BagState.AMBIGUA
        assert bag.state not in TERMINAL_BAG_STATES
        assert bag.occluded_since is None


def test_cada_dono_leva_a_sua_e_ninguem_e_acusado():
    """O defeito visível: com o id mandando, cada dono levava a mala que o
    sistema achava ser do outro, e os dois saíam com N3."""
    sessao = roda(gemeas_trocadas(PONTO_B))

    assert sessao.events.of_kind(EventKind.BAG_REMOVED_BY_STRANGER) == []
    assert acusacoes_n3(sessao) == []
    for dono in (DONO_A, DONO_B):
        assert [f for f in sessao.result().flags.for_person(dono) if f.level is FlagLevel.N3] == []


def test_troca_vale_quando_os_tracks_voltam_em_quadros_diferentes():
    """`MALA_B` volta um segundo depois de `MALA_A`. A bagagem B já foi vista
    de novo pelo track de A quando o track de B reaparece sobre a âncora de A
    -- é o track que voltou de uma ausência, não a bagagem dele estar
    ocluída, que libera a decisão pela posição."""
    sessao = roda(gemeas_trocadas(PONTO_B, b_volta_em=VOLTAM_EM + 1.0))

    assert bagagem_do_track(sessao, MALA_A).bag_id == MALA_B
    assert bagagem_do_track(sessao, MALA_B).bag_id == MALA_A
    assert sessao.events.of_kind(EventKind.BAG_REMOVED_BY_STRANGER) == []
    assert acusacoes_n3(sessao) == []


# --- posição ambígua entre âncoras: P3 ----------------------------------------


def test_track_que_volta_entre_ancoras_proximas_deixa_as_duas_ambiguas():
    """O track volta deslocado da própria âncora e sobre um ponto que cabe na
    outra também: não há como saber qual é qual, as duas viram AMBIGUA."""
    sessao = roda(gemeas_trocadas(PONTO_B_AMBIGUO, fim=LEVAM_EM))

    malas = [bagagem_do_track(sessao, track_id) for track_id in (MALA_A, MALA_B)]
    estados = {bag.bag_id: bag.state for bag in malas}
    assert estados == {MALA_A: BagState.AMBIGUA, MALA_B: BagState.AMBIGUA}


def test_ambiguidade_entre_ancoras_nao_acusa_ninguem():
    sessao = roda(gemeas_trocadas(PONTO_B_AMBIGUO))

    assert sessao.events.of_kind(EventKind.BAG_REMOVED_BY_STRANGER) == []
    assert acusacoes_n3(sessao) == []


# --- regressão: bagagem visível não troca de identidade -----------------------


def dono_leva_a_propria_sobre_ancora_ocluida():
    """B some aos 3s e fica ocluída até o fim. A partir de VOLTAM_EM o dono A
    leva a própria mala, SEMPRE visível, para a direita, passando exatamente
    sobre a âncora de B e seguindo para longe do dono B."""
    dono_a = lugar_do_dono(PONTO_A, -1.0)
    dono_b = lugar_do_dono(PONTO_B, +1.0)
    for i in range(int(FIM * FPS)):
        t = i / FPS
        pa = andando(dono_a, +1.0, t, VOLTAM_EM)
        det = [pessoa(DONO_A, pa), pessoa(DONO_B, dono_b)]
        det.append(mala(MALA_A, Point(pa.x + AFASTAMENTO_X, pa.y - AFASTAMENTO_Y)))
        if t < SOMEM_EM:
            det.append(mala(MALA_B, PONTO_B))
        yield t, det


def test_premissa_da_mala_levada_sobre_a_vizinha():
    """A mala de A passa sobre a âncora de B e termina fora do alcance do
    dono B: uma troca indevida de identidade viraria acusação contra A."""
    dono_a = lugar_do_dono(PONTO_A, -1.0)
    final = andando(dono_a, +1.0, FIM, VOLTAM_EM)
    mala_final = Point(final.x + AFASTAMENTO_X, final.y - AFASTAMENTO_Y)
    assert mala_final.x > PONTO_B.x
    assert mala_final.distance_to(lugar_do_dono(PONTO_B, +1.0)) > PIPELINE.owner_search_radius_m
    ocluida_s = FIM - SOMEM_EM
    assert ocluida_s < CONFIG.custody.max_occlusion_s


def test_mala_visivel_levada_sobre_ancora_ocluida_segue_sendo_ela():
    """Só track que volta de uma ausência é decidido pela posição. A mala que
    o dono leva em plena vista continua sendo a dele ao cruzar a âncora da
    vizinha ocluída -- senão o dono saía acusado de levar a vizinha."""
    sessao = roda(dono_leva_a_propria_sobre_ancora_ocluida())

    assert bagagem_do_track(sessao, MALA_A).bag_id == MALA_A
    assert religacoes_de_bagagem(sessao) == []
    assert sessao.events.of_kind(EventKind.BAG_REMOVED_BY_STRANGER) == []
    assert acusacoes_n3(sessao) == []

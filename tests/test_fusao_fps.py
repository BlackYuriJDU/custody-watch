"""A fusão de grupos não pode depender da taxa de quadros da fonte.

`pipeline.merge_every_frames` contava QUADROS: a fusão rodava a cada 25
quadros, 1 s a 25 fps e 8,3 s a 3 fps. E a fusão não é manutenção: é nela que
co-movimento vira vínculo forte, que o cronômetro do vínculo fraco começa e
conclui, e que um afastamento zera esse cronômetro. A mesma cena formava o
grupo em tempos reais diferentes conforme a fonte, e quem conta como dono
(regra P1) quando alguém pega a mala dependia do fps.

Cadência vai em segundos, como o limiar de oclusão em `test_oclusao_fps.py`:
a fusão roda no primeiro quadro a `pipeline.merge_every_s` da anterior. O que
sobra de diferença entre taxas é o que decidir em quadro impõe, um período de
amostragem.
"""

import json
from itertools import pairwise

import numpy as np
import pytest

from custody_watch.config import Config, PipelineConfig, UnsafeValueError, load_config
from custody_watch.events import EventKind
from custody_watch.ground_plane import GroundPlane
from custody_watch.orchestrator import LiveSession
from custody_watch.tracking import TrackedDetection
from custody_watch.types import Point

TAXAS = (3.0, 10.0, 25.0)
OUTRAS_TAXAS = (3.0, 10.0)
REFERENCIA_FPS = 25.0
"""Taxa do CAVIAR: onde os 25 quadros foram ajustados, e onde o comportamento
tem de continuar o mesmo."""
LENTA_FPS = 3.0
QUADROS_DA_CADENCIA_ANTIGA = 25
"""O default de `merge_every_frames` antes desta correção."""

CONFIG = Config()
GRUPO = CONFIG.party
PIPELINE = CONFIG.pipeline

# Câmera de cima, 1px = 1cm, igual a test_oclusao_fps.py.
PLANO = GroundPlane(np.diag([0.01, 0.01, 1.0]))
VELOCIDADE_MS = 1.2
"""Passo de pedestre, o `WALK_SPEED_MS` dos cenários."""

# --- dois que andam juntos: vínculo forte ---------------------------------------

A, B = 1, 2
PARTIDA = Point(10.0, 10.0)
LADO_M = GRUPO.proximity_m / 2.0
PARTEM_EM = 1.0
"""Entram parados lado a lado; a caminhada começa aqui."""
FIM_JUNTOS = 9.0

# --- estranho que senta ao lado do dono: vínculo fraco --------------------------

DONO, ESTRANHO, MALA = 1, 4, 101
ANCORA = Point(10.0, 10.0)
# O dono fica de lado e um metro abaixo da mala, fora da silhueta (o veto
# "carregada" barraria a âncora), como em `test_oclusao_fps.py`.
LUGAR_DO_DONO = Point(ANCORA.x - 0.7, ANCORA.y + 1.0)
ASSENTO = Point(LUGAR_DO_DONO.x - 0.8 * GRUPO.proximity_m, LUGAR_DO_DONO.y)
"""Perto do dono para virar vínculo fraco; longe da mala para não haver
contato nem permanência (N1) que misture outra régua na cena."""
VEM_DE_M = 8.0
ENTRA_EM = 14.8
"""O estranho entra em quadro bem depois de a mala virar âncora do dono."""
FIM_SENTADO = 100.0


def caixa(p: Point, altura_px: float):
    """Caixa cujo pé cai em `p`."""
    cx, base = p.x * 100.0, p.y * 100.0
    return (cx - altura_px / 4, base - altura_px, cx + altura_px / 4, base)


def pessoa(track_id: int, p: Point) -> TrackedDetection:
    return TrackedDetection(track_id, "person", caixa(p, 170))


def mala(track_id: int, p: Point) -> TrackedDetection:
    return TrackedDetection(track_id, "suitcase", caixa(p, 55))


def instantes(fps: float, fim: float) -> list[float]:
    return [i / fps for i in range(round(fim * fps))]


def roda(frames) -> LiveSession:
    sessao = LiveSession(PLANO, CONFIG)
    for t, det in frames:
        sessao.feed(t, det)
    return sessao


def andam_juntos(fps: float):
    """A e B lado a lado, a `LADO_M`, parados até `PARTEM_EM` e depois andando
    juntos no mesmo passo."""
    for t in instantes(fps, FIM_JUNTOS):
        x = PARTIDA.x + max(0.0, t - PARTEM_EM) * VELOCIDADE_MS
        yield t, [pessoa(A, Point(x, PARTIDA.y)), pessoa(B, Point(x, PARTIDA.y + LADO_M))]


def evidencia_completa(fps: float) -> float:
    """Primeiro quadro em que os dois cobriram `late_join_extent_m` juntos: a
    partir dele o co-movimento basta para o vínculo forte, e só falta fundir."""
    return min(
        t
        for t in instantes(fps, FIM_JUNTOS)
        if (t - PARTEM_EM) * VELOCIDADE_MS >= GRUPO.late_join_extent_m
    )


def vinculo_forte_em(fps: float) -> float:
    (forte,) = roda(andam_juntos(fps)).events.of_kind(EventKind.PARTY_JOINED_STRONG)
    return forte.t_end


def estranho_em(t: float) -> Point:
    andou = min(max(0.0, t - ENTRA_EM) * VELOCIDADE_MS, VEM_DE_M)
    return Point(ASSENTO.x, ASSENTO.y + VEM_DE_M - andou)


def estranho_senta_ao_lado(fps: float):
    """O dono fica ao lado da própria mala; o estranho entra em `ENTRA_EM`,
    anda até o assento ao lado do dono e fica."""
    for t in instantes(fps, FIM_SENTADO):
        det = [pessoa(DONO, LUGAR_DO_DONO), mala(MALA, ANCORA)]
        if t >= ENTRA_EM:
            det.append(pessoa(ESTRANHO, estranho_em(t)))
        yield t, det


def entra_no_raio_do_dono(fps: float) -> float:
    """Primeiro quadro com o estranho a ≤ party.proximity_m do dono."""
    return min(
        t
        for t in instantes(fps, FIM_SENTADO)
        if t >= ENTRA_EM and estranho_em(t).distance_to(LUGAR_DO_DONO) <= GRUPO.proximity_m
    )


def vinculo_fraco_em(fps: float) -> float:
    (fraco,) = roda(estranho_senta_ao_lado(fps)).events.of_kind(EventKind.PARTY_JOINED_WEAK)
    assert fraco.subject == ESTRANHO
    return fraco.t_start


def instantes_de_fusao(fps: float) -> list[float]:
    """Instante de cada fusão numa cena qualquer, espiando a sessão: a cadência
    em si não sai no log, só o que ela decide."""
    sessao = LiveSession(PLANO, CONFIG)
    fusoes: list[float] = []
    funde = sessao._s.merge_parties

    def registra(people) -> None:
        fusoes.append(sessao.t)
        funde(people)

    sessao._s.merge_parties = registra
    for t, det in andam_juntos(fps):
        sessao.feed(t, det)
    return fusoes


# --- a cadência -----------------------------------------------------------------


def test_a_25_fps_a_fusao_segue_a_cada_25_quadros():
    """O período de antes, em quadros, na taxa em que ele foi ajustado."""
    fusoes = instantes_de_fusao(REFERENCIA_FPS)
    quadros = [round((depois - antes) * REFERENCIA_FPS) for antes, depois in pairwise(fusoes)]

    assert len(quadros) >= 5
    assert set(quadros) == {QUADROS_DA_CADENCIA_ANTIGA}


@pytest.mark.parametrize("fps", OUTRAS_TAXAS)
def test_intervalo_real_entre_fusoes_e_o_mesmo_em_qualquer_fps(fps):
    """A 3 fps os 25 quadros eram 8,3 s, a 10 fps 2,5 s, contra 1 s a 25 fps.
    Em segundos, o intervalo real é o da referência a menos de um período."""
    referencia = instantes_de_fusao(REFERENCIA_FPS)
    intervalo = referencia[1] - referencia[0]
    fusoes = instantes_de_fusao(fps)

    assert len(fusoes) >= len(referencia) - 1
    for antes, depois in pairwise(fusoes):
        assert depois - antes == pytest.approx(intervalo, abs=1.0 / fps)


@pytest.mark.parametrize("fps", TAXAS)
def test_fusao_nunca_antes_da_cadencia_e_no_maximo_um_quadro_depois(fps):
    """A fusão roda no primeiro quadro a `merge_every_s` da anterior -- ou do
    primeiro quadro, para a primeira."""
    cadencia = PIPELINE.merge_every_s
    fusoes = instantes_de_fusao(fps)

    for antes, depois in pairwise([0.0, *fusoes]):
        assert depois - antes >= cadencia
        assert depois - antes < cadencia + 1.0 / fps


# --- vínculo forte ----------------------------------------------------------------


def test_premissa_dos_que_andam_juntos():
    """Lado a lado a menos de party.proximity_m, a cena inteira cabe na janela
    de histórico, e a primeira fusão antiga a 3 fps -- no 25º quadro, 8 s --
    cai mais de um período depois de a referência ter fundido: é o que torna
    a cena discriminante."""
    fusao_antiga_lenta = (QUADROS_DA_CADENCIA_ANTIGA - 1) / LENTA_FPS
    fusao_antiga_referencia = (
        evidencia_completa(REFERENCIA_FPS) + QUADROS_DA_CADENCIA_ANTIGA / REFERENCIA_FPS
    )

    assert GRUPO.proximity_m > LADO_M
    assert PIPELINE.history_window_s >= FIM_JUNTOS
    assert fusao_antiga_lenta < FIM_JUNTOS
    assert fusao_antiga_lenta - fusao_antiga_referencia > 1.0 / LENTA_FPS


@pytest.mark.parametrize("fps", OUTRAS_TAXAS)
def test_vinculo_forte_se_forma_no_mesmo_tempo_real(fps):
    """Com 25 quadros, o par que anda junto desde 1 s virava grupo em 5,96 s a
    25 fps, 7,4 s a 10 fps e 8 s a 3 fps. Quem pega a mala nesse intervalo é
    acompanhante numa fonte e estranho na outra. Com a cadência em segundos,
    a fusão cai no mesmo instante real a menos de um período: a evidência
    fica completa entre duas fusões nas três taxas (5,2 s a 25 e a 10 fps,
    5,33 s a 3 fps), e a fusão seguinte é a mesma."""
    referencia = vinculo_forte_em(REFERENCIA_FPS)

    assert vinculo_forte_em(fps) == pytest.approx(referencia, abs=1.0 / fps)


@pytest.mark.parametrize("fps", TAXAS)
def test_vinculo_forte_nunca_antes_da_evidencia_e_no_maximo_uma_cadencia_depois(fps):
    """Garantia que vale em qualquer fase: o grupo sai na primeira fusão
    depois de a evidência bastar, e fusões distam no máximo a cadência mais
    um quadro."""
    cadencia = PIPELINE.merge_every_s
    evidencia = evidencia_completa(fps)
    forte = vinculo_forte_em(fps)

    assert forte >= evidencia
    assert forte < evidencia + cadencia + 1.0 / fps


# --- vínculo fraco ----------------------------------------------------------------


def test_premissa_do_estranho_que_senta_ao_lado():
    """Sentado, o estranho fica no raio de grupo do dono e fora dos raios da
    mala. A 3 fps a cadência antiga começava a contar até 8,3 s depois de ele
    chegar e concluía até 8,3 s depois do prazo; a cena dura o bastante para
    a conclusão antiga sair, então a diferença aparece como tempo e não como
    evento faltando."""
    cadencia_antiga_lenta = QUADROS_DA_CADENCIA_ANTIGA / LENTA_FPS
    pior_conclusao_antiga = (
        entra_no_raio_do_dono(LENTA_FPS) + GRUPO.weak_bond_s + 2.0 * cadencia_antiga_lenta
    )

    assert ASSENTO.distance_to(LUGAR_DO_DONO) < GRUPO.proximity_m
    assert ASSENTO.distance_to(ANCORA) > PIPELINE.proximity_radius_m
    assert ASSENTO.distance_to(ANCORA) > PIPELINE.contact_radius_m
    assert pior_conclusao_antiga < FIM_SENTADO


@pytest.mark.parametrize("fps", OUTRAS_TAXAS)
def test_vinculo_fraco_se_forma_no_mesmo_tempo_real(fps):
    """Com 25 quadros, os mesmos 60 s ao lado do dono viravam vínculo fraco em
    81,96 s a 25 fps, 82,4 s a 10 fps e 91,33 s a 3 fps: quase dez segundos a
    mais, a 3 fps, em que quem senta ao lado não tinha o flag atenuado. Em
    segundos, a diferença é a de um período."""
    referencia = vinculo_fraco_em(REFERENCIA_FPS)

    assert vinculo_fraco_em(fps) == pytest.approx(referencia, abs=1.0 / fps)


@pytest.mark.parametrize("fps", TAXAS)
def test_vinculo_fraco_conclui_no_prazo_com_no_maximo_duas_cadencias(fps):
    """Uma fusão para começar a contar, outra para concluir -- a mesma folga
    que `Timing.weak_bond_wait_s` dá aos cenários."""
    cadencia = PIPELINE.merge_every_s
    chegou = entra_no_raio_do_dono(fps)
    fraco = vinculo_fraco_em(fps)

    assert fraco >= chegou + GRUPO.weak_bond_s
    assert fraco < chegou + GRUPO.weak_bond_s + 2.0 * (cadencia + 1.0 / fps)


# --- config ---------------------------------------------------------------------


def test_default_reproduz_os_25_quadros_a_25_fps():
    """Qualquer cadência em (24/25, 25/25] s funde a cada 25 quadros a 25 fps.
    O default fica no meio do intervalo, longe das duas bordas: a soma binária
    de k/25 e o jitter de timestamp de uma câmera real não podem decidir se a
    fusão cai no 24º, no 25º ou no 26º quadro."""
    cadencia = PipelineConfig().merge_every_s

    assert cadencia > (QUADROS_DA_CADENCIA_ANTIGA - 1) / REFERENCIA_FPS
    assert cadencia < QUADROS_DA_CADENCIA_ANTIGA / REFERENCIA_FPS


def test_campo_antigo_em_quadros_e_recusado_apontando_o_novo(tmp_path):
    """Converter quadros em segundos exige o fps da fonte, que a config não
    conhece; adivinhar 25 fps reabriria o defeito em silêncio. Então a config
    antiga falha alto e diz o que pôr no lugar."""
    caminho = tmp_path / "c.json"
    caminho.write_text(json.dumps({"pipeline": {"merge_every_frames": 25}}), encoding="utf-8")

    with pytest.raises(ValueError, match="merge_every_s") as erro:
        load_config(caminho)

    assert "fps" in str(erro.value)


@pytest.mark.parametrize("cadencia", [0.0, 2.0])
def test_cadencia_em_segundos_dentro_da_faixa_carrega(tmp_path, cadencia):
    """Zero é permitido: fundir a cada quadro custa CPU, não segurança."""
    caminho = tmp_path / "c.json"
    caminho.write_text(json.dumps({"pipeline": {"merge_every_s": cadencia}}), encoding="utf-8")

    assert load_config(caminho).pipeline.merge_every_s == pytest.approx(cadencia)


def test_cadencia_antiga_a_3_fps_fica_fora_da_faixa_segura(tmp_path):
    """Os 25 quadros a 3 fps, 8,3 s, são justamente o defeito: entre duas
    fusões, a caminhada mínima de quem chegou junto sai da janela de
    histórico, e o acompanhante vira estranho."""
    caminho = tmp_path / "c.json"
    caminho.write_text(
        json.dumps({"pipeline": {"merge_every_s": QUADROS_DA_CADENCIA_ANTIGA / LENTA_FPS}}),
        encoding="utf-8",
    )

    with pytest.raises(UnsafeValueError, match="merge_every_s"):
        load_config(caminho)

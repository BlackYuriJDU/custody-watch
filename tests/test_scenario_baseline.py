"""Baseline ingênua: a regra de distância/tempo, sem grupo, incerteza nem re-ID.

Alguns destes testes afirmam que a baseline ERRA. É de propósito: ela é a
referência do que P1–P3 compram, e o erro dela é o comportamento esperado.
"""

from dataclasses import replace

from custody_watch.config import Config
from custody_watch.scenario_baseline import NaiveBaseline, run_baseline
from custody_watch.scenario_scoring import Category
from custody_watch.scenarios import build_scenarios, identity_plane
from custody_watch.types import Observation, Point

CONFIG = Config()
CENARIOS = {s.name: s for s in build_scenarios(CONFIG)}
FPS = 10.0


def alarmes(nome: str, config: Config = CONFIG, cenarios=CENARIOS):
    return run_baseline(cenarios[nome].frames(FPS), identity_plane(), config)


def acusados(emissoes) -> list[int]:
    return [e.person for e in emissoes if e.category is Category.ACCUSATION]


def test_baseline_acusa_o_estranho_no_furto_base():
    cena = CENARIOS["stranger_takes"]
    emissoes = alarmes(cena.name)
    acusacao = next(e for e in emissoes if e.category is Category.ACCUSATION)

    assert acusados(emissoes) == [cena.role("estranho")]
    assert acusacao.bag == cena.role("bagagem")
    assert cena.mark("retirada") <= acusacao.t


def test_baseline_decide_no_primeiro_quadro_fora_da_ancora():
    """Sem estado de incerteza: não espera `carry_confirm_s`. A mala cruza
    `moved_threshold_m` andando a 1,2 m/s, e o alarme sai no quadro seguinte."""
    cena = CENARIOS["stranger_takes"]
    acusacao = next(e for e in alarmes(cena.name) if e.category is Category.ACCUSATION)
    alcance = CONFIG.registry.moved_threshold_m / 1.2

    assert acusacao.t - cena.mark("retirada") <= alcance + 1.0 / FPS + 1e-9
    assert acusacao.t - cena.mark("retirada") < CONFIG.custody.carry_confirm_s


def test_baseline_acusa_o_acompanhante_que_chegou_junto():
    """Comportamento ESPERADO: sem grupo, o acompanhante é um estranho."""
    cena = CENARIOS["companion_takes"]

    assert acusados(alarmes(cena.name)) == [cena.role("acompanhante")]


def test_baseline_acusa_o_vizinho_mais_proximo_na_fila():
    cena = CENARIOS["owner_pulls_neighbour_closer"]

    assert acusados(alarmes(cena.name)) == [cena.role("vizinho")]


def test_baseline_nao_acusa_o_dono_que_volta_e_marca_o_abandono():
    cena = CENARIOS["owner_returns"]
    emissoes = alarmes(cena.name)

    assert acusados(emissoes) == []
    assert [(e.category, e.bag) for e in emissoes] == [(Category.UNATTENDED, cena.role("bagagem"))]


def test_baseline_ignora_vinculo_fraco_e_marca_o_abandono():
    cena = CENARIOS["owner_leaves_weak_stays"]
    emissoes = alarmes(cena.name)

    assert [e.category for e in emissoes] == [Category.UNATTENDED]
    assert emissoes[0].t >= cena.mark("dono_cruza_raio") + CONFIG.custody.unattended_time_s


def test_baseline_sem_reid_toma_tracks_trocados_por_furto():
    cena = CENARIOS["twin_bags_swapped_ids"]

    assert sorted(acusados(alarmes(cena.name))) == [cena.role("dono_a"), cena.role("dono_b")]


def test_baseline_trata_track_novo_como_mala_nova_com_dono_novo():
    """Tracks novos ancoram de novo e ganham dono por proximidade; aqui o
    mais próximo é o dono certo, e ninguém é acusado."""
    assert alarmes("twin_bags_new_ids") == ()


def test_baseline_usa_os_prazos_da_config():
    curta = replace(CONFIG, custody=replace(CONFIG.custody, unattended_time_s=10.0))
    cenarios = {s.name: s for s in build_scenarios(curta)}
    cena = cenarios["owner_returns"]

    abandono = next(
        e for e in alarmes(cena.name, curta, cenarios) if e.category is Category.UNATTENDED
    )

    assert 10.0 <= abandono.t - cena.mark("dono_cruza_raio") <= 10.0 + 2.0 / FPS + 0.01


def _obs(track: int, cls: str, x: float, t: float) -> Observation:
    return Observation(track_id=track, cls=cls, position=Point(x, 0.0), t=t)


def test_dono_e_quem_estava_mais_perto_quando_a_mala_parou():
    """Regra crua, quadro a quadro: o dono leva sem alarme; outro, com alarme."""
    baseline = NaiveBaseline(CONFIG)
    passo = 0.5
    parada = CONFIG.custody.rest_confirm_s
    for k in range(int(parada / passo) + 2):
        t = k * passo
        baseline.feed(
            t, [_obs(1, "person", 0.5, t), _obs(2, "person", 2.5, t), _obs(9, "suitcase", 0.0, t)]
        )
    t = parada + 2 * passo
    baseline.feed(
        t, [_obs(1, "person", 5.0, t), _obs(2, "person", 1.0, t), _obs(9, "suitcase", 1.2, t)]
    )

    assert [(e.category, e.person, e.bag) for e in baseline.emissions] == [
        (Category.ACCUSATION, 2, 9)
    ]

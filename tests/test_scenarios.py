"""Biblioteca de cenários: determinismo, premissas, papéis, e o desfecho do
sistema nos cenários em que a `main` já acerta.

Os demais cenários expõem defeitos que estão sendo corrigidos em outras
branches. Para eles este arquivo só verifica premissas e que a sessão roda;
o veredito é resultado do benchmark, não asserção -- afrouxar o gabarito
para passar apagaria justamente o que ele existe para mostrar.
"""

from dataclasses import replace
from itertools import pairwise

import pytest

from custody_watch.config import Config
from custody_watch.events import EventKind
from custody_watch.orchestrator import run_session
from custody_watch.scenario_model import INVARIANTS, KINDS, MIN_FPS, WALK_SPEED_MS
from custody_watch.scenario_scoring import Category, Emission, Verdict, score, system_emissions
from custody_watch.scenarios import build_scenarios, identity_plane
from custody_watch.tracking import to_observations

CONFIG = Config()
CENARIOS = {s.name: s for s in build_scenarios(CONFIG)}
TAXAS = (3.0, 10.0, 25.0)

PASSAM_NA_MAIN = {
    "owner_returns": Verdict.CORRECT,
    "companion_takes": Verdict.SILENT_CORRECT,
    "stranger_takes": Verdict.CORRECT,
    "sit_beside_then_take": Verdict.CORRECT,
    "twin_bags_new_ids": Verdict.SILENT_CORRECT,
    "passerby_no_luggage": Verdict.SILENT_CORRECT,
}


def sessao(nome: str, fps: float):
    return run_session(CENARIOS[nome].frames(fps), identity_plane(), CONFIG)


def veredito(nome: str, fps: float) -> Verdict:
    resultado = sessao(nome, fps)
    emissoes = system_emissions(resultado.flags, resultado.events)
    return score(CENARIOS[nome].expected, emissoes).verdict


def test_catalogo_cobre_os_cenarios_minimos_e_todas_as_invariantes():
    assert {
        "owner_returns",
        "companion_takes",
        "stranger_takes",
        "sit_beside_then_take",
        "owner_leaves_weak_stays",
        "twin_bags_new_ids",
        "new_track_between_occluded",
        "owner_pulls_neighbour_closer",
        "passerby_no_luggage",
        "detection_gap_near_bag",
    } <= set(CENARIOS)
    assert {s.invariant for s in CENARIOS.values()} == set(INVARIANTS)
    assert {s.kind for s in CENARIOS.values()} == set(KINDS)
    assert len(CENARIOS) <= 12


@pytest.mark.parametrize("nome", sorted(CENARIOS))
def test_mesma_config_e_fps_geram_os_mesmos_quadros(nome):
    de_novo = {s.name: s for s in build_scenarios(Config())}[nome]

    assert list(CENARIOS[nome].frames(10.0)) == list(de_novo.frames(10.0))


@pytest.mark.parametrize("nome", sorted(CENARIOS))
@pytest.mark.parametrize("fps", TAXAS)
def test_tempo_estritamente_crescente(nome, fps):
    tempos = [t for t, _ in CENARIOS[nome].frames(fps)]

    assert tempos[0] == 0.0
    assert all(b > a for a, b in pairwise(tempos))
    assert tempos[-1] <= CENARIOS[nome].duration_s


@pytest.mark.parametrize("nome", sorted(CENARIOS))
def test_premissas_valem_na_config_default(nome):
    falhas = [
        f"{p.descricao}: {p.medida}" for p in CENARIOS[nome].premissas(CONFIG) if not p.valida
    ]

    assert falhas == []


def test_premissa_denuncia_config_que_invalida_a_cena():
    """A cena foi montada para weak_bond_s = 60 s; com 600 s o estranho não
    chega a virar vínculo fraco, e é a premissa que tem de dizer isso."""
    lenta = replace(CONFIG, party=replace(CONFIG.party, weak_bond_s=600.0))

    falhas = [
        p.descricao for p in CENARIOS["sit_beside_then_take"].premissas(lenta) if not p.valida
    ]

    assert any("vínculo fraco" in f for f in falhas)


def test_premissa_denuncia_veto_de_imagem():
    exigente = replace(CONFIG, pipeline=replace(CONFIG.pipeline, min_bag_height_px=200.0))

    falhas = [p.descricao for p in CENARIOS["stranger_takes"].premissas(exigente) if not p.valida]

    assert any("veto de imagem" in f for f in falhas)


@pytest.mark.parametrize("nome", sorted(CENARIOS))
def test_papeis_e_gabarito_consistentes(nome):
    cena = CENARIOS[nome]
    atores = {a.track_id: a for a in cena.actors}
    pessoas = {t for t, a in atores.items() if a.is_person}
    malas = set(atores) - pessoas

    assert set(cena.papeis.values()) <= set(atores)
    assert cena.expected.never_accuse <= pessoas
    for item in cena.expected.due:
        assert item.target in (malas if item.category is Category.UNATTENDED else pessoas)
        assert item.tolerance_s > 0.0
        assert 0.0 <= item.reference_t <= cena.duration_s
    acusados = {d.target for d in cena.expected.due if d.category is Category.ACCUSATION}
    assert acusados.isdisjoint(cena.expected.never_accuse)
    if cena.expected.silence:
        assert cena.expected.due == ()


def test_cenarios_de_furto_tem_sinal_devido():
    for cena in CENARIOS.values():
        if cena.kind == "furto":
            assert cena.expected.due, cena.name


def test_quadros_estao_em_metros_no_plano_identidade():
    """O pé de cada caixa, projetado, é a posição do ator na cena."""
    cena = CENARIOS["stranger_takes"]
    t, deteccoes = next(
        (t, d) for t, d in cena.frames(10.0) if t > cena.mark("retirada") + 1.0 and len(d) == 2
    )
    assert {d.cls for d in deteccoes} == {"person", "suitcase"}

    for obs in to_observations(deteccoes, identity_plane(), t):
        esperado = cena.actor(obs.track_id).position(t)
        assert obs.position.distance_to(esperado) < 1e-9


def test_fps_abaixo_do_minimo_e_recusado():
    with pytest.raises(ValueError, match="abaixo"):
        CENARIOS["owner_returns"].frames(MIN_FPS / 2.0)


@pytest.mark.parametrize("nome", sorted(CENARIOS))
def test_todo_cenario_roda_no_sistema_e_recebe_veredito(nome):
    assert isinstance(veredito(nome, MIN_FPS), Verdict)


@pytest.mark.parametrize("fps", TAXAS)
@pytest.mark.parametrize("nome", sorted(PASSAM_NA_MAIN))
def test_desfecho_do_sistema_nos_cenarios_que_a_main_acerta(nome, fps):
    assert veredito(nome, fps) is PASSAM_NA_MAIN[nome]


def test_furto_base_tem_o_mesmo_desfecho_em_3_10_e_25_fps():
    cena = CENARIOS["stranger_takes"]
    acusados = set()
    for fps in TAXAS:
        resultado = sessao(cena.name, fps)
        acusados |= {e.subject for e in resultado.events.of_kind(EventKind.BAG_REMOVED_BY_STRANGER)}

    assert {veredito(cena.name, fps) for fps in TAXAS} == {Verdict.CORRECT}
    assert acusados == {cena.role("estranho")}


# --- prazo da retirada com o dono ao alcance ----------------------------------


def _acusacao_devida(cena):
    return next(d for d in cena.expected.due if d.category is Category.ACCUSATION)


def _prazo_esperado(config: Config, cena) -> float:
    """A conta de `accusation_beside_owner_tolerance_s`, refeita a partir da
    config e das posições da cena -- não do que a cena declarou."""
    t_take = cena.mark("retirada")
    mala, dono = cena.actor(cena.role("bagagem")), cena.actor(cena.role("dono"))
    d0 = mala.position(t_take).distance_to(dono.position(t_take))
    saida = config.pipeline.owner_search_radius_m - d0 + config.registry.moved_threshold_m
    return saida / WALK_SPEED_MS + config.custody.carry_confirm_s + 2.0 / MIN_FPS


def test_prazo_do_furto_ao_lado_e_a_soma_dos_termos_da_regra():
    cena = CENARIOS["sit_beside_then_take"]
    devida = _acusacao_devida(cena)

    assert devida.reference_t == pytest.approx(cena.mark("retirada"))
    assert devida.tolerance_s == pytest.approx(_prazo_esperado(CONFIG, cena))


def test_prazo_do_furto_ao_lado_acompanha_owner_search_radius_m():
    raio = CONFIG.pipeline.owner_search_radius_m + 1.0
    maior = replace(CONFIG, pipeline=replace(CONFIG.pipeline, owner_search_radius_m=raio))
    cena = {s.name: s for s in build_scenarios(maior)}["sit_beside_then_take"]
    delta = _acusacao_devida(cena).tolerance_s - _acusacao_devida(CENARIOS[cena.name]).tolerance_s

    assert delta == pytest.approx(1.0 / WALK_SPEED_MS)
    assert _acusacao_devida(cena).tolerance_s == pytest.approx(_prazo_esperado(maior, cena))


@pytest.mark.parametrize(
    ("atraso_extra_s", "esperado"),
    [(0.0, Verdict.CORRECT), (1.0 / MIN_FPS + 0.01, Verdict.MISSED)],
)
def test_prazo_do_furto_ao_lado_nao_e_frouxo(atraso_extra_s, esperado):
    """Acusação no prazo derivado acerta; um período de amostragem além, perde."""
    cena = CENARIOS["sit_beside_then_take"]
    ladrao = cena.role("estranho")
    bagagem = cena.role("bagagem")
    devida = _acusacao_devida(cena)
    t_acusa = devida.reference_t + _prazo_esperado(CONFIG, cena) + atraso_extra_s
    emissoes = [
        Emission(Category.PERSON_SIGNAL, cena.mark("encosta"), ladrao, bagagem, "sintetica"),
        Emission(Category.ACCUSATION, t_acusa, ladrao, bagagem, "sintetica"),
    ]

    assert score(cena.expected, emissoes).verdict is esperado


def test_premissa_denuncia_mala_que_nao_sai_do_alcance_do_dono():
    """Com raio maior que todo o trajeto, a mala nunca sai do alcance: a conta
    do prazo deixa de descrever a cena, e é a premissa que tem de dizer isso."""
    enorme = replace(CONFIG, pipeline=replace(CONFIG.pipeline, owner_search_radius_m=50.0))
    cena = {s.name: s for s in build_scenarios(enorme)}["sit_beside_then_take"]

    falhas = [p.descricao for p in cena.premissas(enorme) if not p.valida]

    assert any("sai do alcance" in f for f in falhas)


# --- a cena exercita o mecanismo que diz exercitar ----------------------------


def test_acompanhante_vira_vinculo_forte():
    cena = CENARIOS["companion_takes"]
    fortes = sessao(cena.name, MIN_FPS).events.of_kind(EventKind.PARTY_JOINED_STRONG)

    assert [e.subject for e in fortes] == [cena.role("acompanhante")]


@pytest.mark.parametrize("nome", ["sit_beside_then_take", "owner_leaves_weak_stays"])
def test_estranho_sentado_vira_vinculo_fraco(nome):
    cena = CENARIOS[nome]
    fracos = sessao(nome, MIN_FPS).events.of_kind(EventKind.PARTY_JOINED_WEAK)

    assert [e.subject for e in fracos] == [cena.role("estranho")]


def test_malas_gemeas_sao_readotadas_pela_posicao():
    cena = CENARIOS["twin_bags_new_ids"]
    religadas = [
        e for e in sessao(cena.name, 10.0).events.of_kind(EventKind.TRACK_RELINKED) if e.bag
    ]

    assert {(e.bag, e.evidence["from_track"]) for e in religadas} == {
        (cena.role("bagagem_a"), cena.role("bagagem_a_novo_track")),
        (cena.role("bagagem_b"), cena.role("bagagem_b_novo_track")),
    }


def test_track_do_meio_e_declarado_ambiguo():
    eventos = sessao("new_track_between_occluded", 10.0).events

    assert eventos.of_kind(EventKind.BAG_AMBIGUOUS)


def test_lacuna_nao_cria_vinculo_que_mascare_a_permanencia():
    eventos = sessao("detection_gap_near_bag", 10.0).events

    assert eventos.of_kind(EventKind.PARTY_JOINED_WEAK) == []
    assert eventos.of_kind(EventKind.BAG_UNATTENDED) == []

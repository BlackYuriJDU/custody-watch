"""Pontuação de cenário: casos montados à mão, com resposta calculada."""

from custody_watch.events import Event, EventKind
from custody_watch.flags import FlagStore
from custody_watch.scenario_scoring import (
    Category,
    DueSignal,
    Emission,
    Expectation,
    Verdict,
    score,
    system_emissions,
)
from custody_watch.types import Flag, FlagLevel

ESTRANHO, DONO, VIZINHO, MALA = 3, 1, 6, 101


def devido(categoria: Category, alvo: int, ref: float, tol: float) -> DueSignal:
    return DueSignal(categoria, alvo, ref, tol, "caso de teste")


def emissao(categoria: Category, t: float, pessoa: int | None = None, mala: int | None = None):
    return Emission(categoria, t, pessoa, mala, "teste")


def gabarito(*devidos: DueSignal, protegidos=(DONO,), silencio=False) -> Expectation:
    return Expectation(never_accuse=frozenset(protegidos), due=devidos, silence=silencio)


FURTO = gabarito(devido(Category.ACCUSATION, ESTRANHO, 50.0, 3.0))


def test_acusacao_devida_dentro_da_janela_e_correto():
    resultado = score(FURTO, [emissao(Category.ACCUSATION, 51.5, ESTRANHO, MALA)])

    assert resultado.verdict is Verdict.CORRECT
    assert resultado.due[0].emitted is True
    assert resultado.due[0].delay_s == 1.5
    assert resultado.undue_accused == ()
    assert resultado.silence is False


def test_atraso_exatamente_no_limite_ainda_conta():
    """Referência e tolerância como o gabarito calcula, com ponto flutuante de verdade."""
    ref, tol = 50.77, 3.0766666666666667
    caso = gabarito(devido(Category.ACCUSATION, ESTRANHO, ref, tol))

    resultado = score(caso, [emissao(Category.ACCUSATION, ref + tol, ESTRANHO, MALA)])

    assert resultado.verdict is Verdict.CORRECT
    assert resultado.due[0].emitted is True


def test_um_quadro_alem_do_limite_e_perdeu_mas_o_atraso_e_reportado():
    quadro = 1.0 / 25.0
    resultado = score(FURTO, [emissao(Category.ACCUSATION, 53.0 + quadro, ESTRANHO, MALA)])

    assert resultado.verdict is Verdict.MISSED
    assert resultado.due[0].emitted is False
    assert abs(resultado.due[0].delay_s - (3.0 + quadro)) < 1e-9


def test_emissao_antes_do_ato_nao_cumpre_o_item_devido():
    resultado = score(FURTO, [emissao(Category.ACCUSATION, 49.0, ESTRANHO, MALA)])

    assert resultado.verdict is Verdict.MISSED
    assert resultado.due[0].delay_s is None
    assert resultado.undue_accused == ()


def test_acusar_quem_o_gabarito_protege_e_acusacao_indevida():
    caso = gabarito(protegidos=(DONO, VIZINHO))

    resultado = score(caso, [emissao(Category.ACCUSATION, 20.0, VIZINHO, MALA)])

    assert resultado.verdict is Verdict.UNDUE_ACCUSATION
    assert resultado.undue_accused == (VIZINHO,)


def test_acusacao_que_nenhum_item_justifica_e_indevida():
    """Quem não está na lista de protegidos também não pode ser acusado sem motivo."""
    resultado = score(FURTO, [emissao(Category.ACCUSATION, 51.0, 99, MALA)])

    assert resultado.verdict is Verdict.UNDUE_ACCUSATION
    assert resultado.undue_accused == (99,)


def test_empate_no_mesmo_instante_acusacao_devida_e_indevida():
    """Ladrão e vizinho acusados no mesmo quadro: o item devido saiu, mas o
    veredito é o mais grave."""
    emissoes = [
        emissao(Category.ACCUSATION, 51.0, ESTRANHO, MALA),
        emissao(Category.ACCUSATION, 51.0, VIZINHO, MALA),
    ]

    resultado = score(FURTO, emissoes)

    assert resultado.verdict is Verdict.UNDUE_ACCUSATION
    assert resultado.undue_accused == (VIZINHO,)
    assert resultado.due[0].emitted is True


def test_empate_entre_perdeu_e_acusacao_indevida_fica_com_a_mais_grave():
    resultado = score(FURTO, [emissao(Category.ACCUSATION, 51.0, DONO, MALA)])

    assert resultado.verdict is Verdict.UNDUE_ACCUSATION
    assert resultado.due[0].emitted is False


def test_duas_emissoes_casadas_mede_o_atraso_pela_primeira():
    emissoes = [
        emissao(Category.ACCUSATION, 52.5, ESTRANHO, MALA),
        emissao(Category.ACCUSATION, 51.0, ESTRANHO, MALA),
    ]

    assert score(FURTO, emissoes).due[0].delay_s == 1.0


def test_silencio_devido_e_observado_e_mudo_correto():
    resultado = score(gabarito(silencio=True), [])

    assert resultado.verdict is Verdict.SILENT_CORRECT
    assert resultado.silence is True


def test_sinal_brando_quando_o_silencio_e_devido_e_sinal_indevido():
    caso = gabarito(protegidos=(DONO, 8), silencio=True)
    n1 = emissao(Category.PERSON_SIGNAL, 203.7, 8, MALA)

    resultado = score(caso, [n1])

    assert resultado.verdict is Verdict.UNDUE_SIGNAL
    assert resultado.undue_signals == (n1,)
    assert resultado.undue_accused == ()


def test_sinal_brando_e_permitido_quando_o_silencio_nao_e_exigido():
    caso = gabarito(protegidos=(DONO, VIZINHO))

    resultado = score(caso, [emissao(Category.PERSON_SIGNAL, 17.3, VIZINHO, MALA)])

    assert resultado.verdict is Verdict.CORRECT
    assert resultado.silence is False


def test_item_devido_para_outra_mala_nao_conta():
    caso = gabarito(devido(Category.UNATTENDED, MALA, 13.0, 26.7))

    resultado = score(caso, [emissao(Category.UNATTENDED, 38.0, mala=102)])

    assert resultado.verdict is Verdict.MISSED


def test_categoria_fora_do_escopo_nao_pesa_no_veredito():
    """A baseline não tem sinal de pessoa: o item fica registrado, fora da conta."""
    caso = gabarito(
        devido(Category.PERSON_SIGNAL, ESTRANHO, 98.5, 1.7),
        devido(Category.ACCUSATION, ESTRANHO, 101.4, 3.1),
    )
    escopo = frozenset({Category.ACCUSATION, Category.UNATTENDED})

    resultado = score(caso, [emissao(Category.ACCUSATION, 101.7, ESTRANHO, MALA)], escopo)

    assert resultado.verdict is Verdict.CORRECT
    assert [o.in_scope for o in resultado.due] == [False, True]
    assert resultado.due[0].emitted is False


def test_emissao_fora_do_escopo_nao_quebra_o_silencio():
    escopo = frozenset({Category.ACCUSATION})

    resultado = score(
        gabarito(silencio=True), [emissao(Category.UNATTENDED, 10.0, mala=MALA)], escopo
    )

    assert resultado.verdict is Verdict.SILENT_CORRECT


def _flag(nivel: FlagLevel, tipo: str, pessoa: int, t: float) -> Flag:
    return Flag(tipo, nivel, pessoa, MALA, t, 1.0, "explicação")


def _evento(tipo: EventKind, t: float, sujeito: int | None = None) -> Event:
    return Event(tipo, t, t, sujeito, MALA, 1, {})


def test_saida_do_sistema_classificada_num_lugar_so():
    """N3 é acusação; N1 e N2 são sinal de pessoa; só BAG_UNATTENDED é sinal de
    bagagem. Retirada, ambiguidade e oclusão são registro, não sinal."""
    flags = FlagStore()
    flags.add(_flag(FlagLevel.N1, "permanencia_bagagem_alheia", 8, 90.0))
    flags.add(_flag(FlagLevel.N2, "contato_bagagem_alheia", ESTRANHO, 48.7))
    flags.add(_flag(FlagLevel.N3, "retirada_por_estranho", ESTRANHO, 52.2))
    eventos = [
        _evento(EventKind.BAG_UNATTENDED, 38.5),
        _evento(EventKind.BAG_REMOVED_BY_STRANGER, 52.2, ESTRANHO),
        _evento(EventKind.BAG_AMBIGUOUS, 60.0),
        _evento(EventKind.BAG_OCCLUDED, 61.0),
        _evento(EventKind.BAG_REATTENDED, 62.0),
    ]

    emissoes = system_emissions(flags, eventos)

    assert [(e.category, e.t, e.person, e.bag) for e in emissoes] == [
        (Category.UNATTENDED, 38.5, None, MALA),
        (Category.PERSON_SIGNAL, 48.7, ESTRANHO, MALA),
        (Category.ACCUSATION, 52.2, ESTRANHO, MALA),
        (Category.PERSON_SIGNAL, 90.0, 8, MALA),
    ]


def test_relatorio_tem_os_campos_pedidos():
    relatorio = score(FURTO, [emissao(Category.ACCUSATION, 51.0, ESTRANHO, MALA)]).to_dict()

    assert relatorio["veredito"] == "correto"
    assert relatorio["acusacao_indevida"] == {"ocorreu": False, "quem": []}
    assert relatorio["evento_devido_emitido"]["todos"] is True
    assert relatorio["evento_devido_emitido"]["itens"][0]["atraso_s"] == 1.0
    assert relatorio["silencio"] is False
    assert relatorio["sinal_indevido"]["ocorreu"] is False

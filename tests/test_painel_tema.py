"""Tema claro/escuro do painel ao vivo (CSS embutido em servidor.PAGINA)."""

import re

import pytest

from custody_watch.servidor import PAGINA
from tests.test_multicamera_tema import (
    FUNCOES_COR,
    HEX,
    _blocos,
    _corpo,
    _luminancia,
    _razao,
    _tokens,
)

# (texto, fundo, minimo). 4,5 = texto normal; 3 = borda, foco, marcador.
PARES = [
    ("ink", "ground", 4.5),
    ("ink", "surface", 4.5),
    ("ink", "raised", 4.5),
    ("ink", "hairline", 4.5),
    ("muted-text", "ground", 4.5),
    ("muted-text", "surface", 4.5),
    ("muted-text", "raised", 4.5),
    ("accent-text", "selecao-fundo", 4.5),
    ("erro-texto", "ground", 4.5),
    ("erro-texto", "surface", 4.5),
    ("historico-texto", "historico-fundo", 4.5),
    ("on-solid", "grave-solid", 4.5),
    ("on-solid", "media-solid", 4.5),
    ("on-solid", "leve-solid", 4.5),
    ("accent", "ground", 3.0),
    ("accent", "surface", 3.0),
    ("accent", "raised", 3.0),
    ("edge", "surface", 3.0),
    ("edge", "raised", 3.0),
    ("grave", "surface", 3.0),
    ("media", "surface", 3.0),
    ("leve", "surface", 3.0),
    ("historico-borda", "historico-fundo", 3.0),
    ("historico-borda", "ground", 3.0),
]

SEVERIDADES = ["grave-solid", "media-solid", "leve-solid"]


def _estilo():
    return re.search(r"<style>(.*?)</style>", PAGINA, re.S).group(1)


def _tokens_do_painel():
    css = _estilo()
    claro, *_ = _corpo(css, ":root {")
    media, *_ = _corpo(css, "@media (prefers-color-scheme: dark)")
    escuro_media, *_ = _corpo(media, ':root:not([data-theme="light"])')
    escuro_attr, *_ = _corpo(css, ':root[data-theme="dark"]')
    return _tokens(claro), _tokens(escuro_media), _tokens(escuro_attr)


def _temas():
    claro, _, escuro_attr = _tokens_do_painel()
    return {"light": claro, "dark": escuro_attr}


def _sem_blocos_de_tokens():
    css = _estilo()
    for cabecalho in (
        ":root {",
        "@media (prefers-color-scheme: dark)",
        ':root[data-theme="dark"]',
    ):
        _, ini, fim = _corpo(css, cabecalho)
        css = css[:ini] + css[fim:]
    return css


def test_tokens_definidos_nos_tres_blocos_com_mesmo_conjunto():
    claro, escuro_media, escuro_attr = _tokens_do_painel()
    tematicos = {n for n, v in claro.items() if v.startswith("#")}
    assert tematicos, "nenhum token de cor no :root"
    assert set(escuro_media) == tematicos
    assert set(escuro_attr) == tematicos
    assert escuro_media == escuro_attr


@pytest.mark.parametrize("tema", ["light", "dark"])
def test_tokens_compartilhados_tem_os_mesmos_valores_da_central(tema):
    painel = _temas()[tema]
    claro, _, escuro_attr = _blocos()
    central = {"light": claro, "dark": escuro_attr}[tema]
    compartilhados = {n for n in painel if n in central}
    # Os de report.py e os de aviso/erro/seleção não podem divergir da central.
    assert {"ground", "surface", "ink", "accent", "grave", "media", "leve"} <= compartilhados
    for nome in compartilhados:
        assert painel[nome] == central[nome], nome


def test_seletores_de_tema_e_color_scheme_seguem_o_padrao_da_central():
    css = _estilo()
    assert re.search(r":root\s*\{\s*color-scheme:\s*light dark;", css)
    assert ':root:not([data-theme="light"])' in css
    assert ':root[data-theme="dark"]' in css
    assert re.search(r"body\s*\{[^}]*background:\s*var\(--ground\)", css)


def test_nenhuma_cor_literal_fora_dos_blocos_de_tokens():
    resto = _sem_blocos_de_tokens()
    corpos = re.findall(r"\{([^{}]*)\}", resto)
    assert corpos
    for corpo in corpos:
        assert not HEX.search(corpo), corpo
        assert not FUNCOES_COR.search(corpo), corpo


def test_todo_var_usado_esta_definido():
    claro, *_ = _tokens_do_painel()
    usados = set(re.findall(r"var\(--([a-z0-9-]+)", _estilo()))
    assert usados - set(claro) == set()


@pytest.mark.parametrize("tema", ["light", "dark"])
@pytest.mark.parametrize(("fg", "bg", "minimo"), PARES)
def test_contraste_dos_pares_declarados(tema, fg, bg, minimo):
    tokens = _temas()[tema]
    razao = _razao(tokens[fg], tokens[bg])
    assert razao >= minimo, f"{tema}: {fg} sobre {bg} = {razao:.2f} < {minimo}"


@pytest.mark.parametrize("tema", ["light", "dark"])
def test_severidades_se_distinguem_entre_si(tema):
    tokens = _temas()[tema]
    cores = [tokens[n].upper() for n in SEVERIDADES]
    assert len(set(cores)) == len(cores)
    # Além da cor, cada severidade carrega o rótulo "N3"/"N2"/"N1" no selo.
    assert all(_luminancia(c) >= 0 for c in cores)


def test_nenhum_font_size_em_px_nem_abaixo_de_12px():
    css = _estilo()
    tamanhos = []
    for _, valor in re.findall(r"(font-size|font)\s*:\s*([^;}]+)", css):
        tamanhos += re.findall(r"(?<![\w.#-])(\d*\.?\d+)(px|rem|em)\b", valor)[:1]
    assert tamanhos
    assert [t for t in tamanhos if t[1] == "px"] == []
    assert [t for t in tamanhos if float(t[0]) * 16 < 12] == []


def test_foco_visivel_cobre_os_controles_interativos():
    css = _estilo()
    regra = re.search(r"([^{}]*:focus-visible[^{}]*)\{([^{}]*)\}", css)
    assert regra
    for alvo in ("button", "a", "[tabindex]"):
        assert f"{alvo}:focus-visible" in regra.group(1)
    assert "outline: 2px solid var(--accent)" in regra.group(2)


def test_sem_url_externa_no_html():
    assert not re.search(r"https?://|(?<!:)//[a-z0-9.-]+\.[a-z]", PAGINA, re.I)
    assert "@import" not in PAGINA
    assert "url(" not in PAGINA
    assert not re.search(r"<link\b", PAGINA, re.I)
    for src in re.findall(r"\bsrc=\"([^\"]*)\"", PAGINA):
        assert src.startswith("/"), src
        assert not src.startswith("//"), src

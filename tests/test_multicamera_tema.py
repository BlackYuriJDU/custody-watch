"""Tema claro/escuro da central multicâmera (assets/multicamera.css)."""

import re
from pathlib import Path

import pytest

import custody_watch

CSS = Path(custody_watch.__file__).parent / "assets" / "multicamera.css"

# Tokens de report.py: nomes e valores devem ser os mesmos na central.
TOKENS_REPORT = {
    "light": {
        "ground": "#F1F3F5",
        "surface": "#FFFFFF",
        "raised": "#E8ECEF",
        "ink": "#14181C",
        "muted": "#667484",
        "hairline": "#D3DAE0",
        "accent": "#C4551F",
        "grave": "#B03A2E",
        "media": "#B5762A",
        "leve": "#5E7387",
    },
    "dark": {
        "ground": "#0E1216",
        "surface": "#161C22",
        "raised": "#1E262E",
        "ink": "#E2E8EE",
        "muted": "#8A98A6",
        "hairline": "#2A343D",
        "accent": "#E0662E",
        "grave": "#D9584B",
        "media": "#D2913C",
        "leve": "#7E95A8",
    },
}

# (texto, fundo, minimo). 4,5 = texto normal; 3 = texto grande, ícone, borda, foco.
PARES = [
    # texto normal sobre superfícies
    ("ink", "ground", 4.5),
    ("ink", "surface", 4.5),
    ("ink", "raised", 4.5),
    ("ink", "hairline", 4.5),
    ("muted-text", "ground", 4.5),
    ("muted-text", "surface", 4.5),
    ("muted-text", "raised", 4.5),
    ("accent-text", "ground", 4.5),
    ("accent-text", "surface", 4.5),
    ("accent-text", "raised", 4.5),
    ("accent-text", "selecao-fundo", 4.5),
    ("media-text", "ground", 4.5),
    ("media-text", "surface", 4.5),
    ("media-text", "raised", 4.5),
    ("observado", "surface", 4.5),
    ("observado", "raised", 4.5),
    ("erro-texto", "erro-fundo", 4.5),
    ("historico-texto", "historico-fundo", 4.5),
    ("nota-titulo", "nota-fundo", 4.5),
    ("nota-texto", "nota-fundo", 4.5),
    ("on-accent", "accent-solid", 4.5),
    ("on-accent", "accent-hover", 4.5),
    ("sobre-quadro", "quadro", 4.5),
    # ícones, bordas de controle, anel de foco e marcadores
    ("accent", "ground", 3.0),
    ("accent", "surface", 3.0),
    ("accent", "raised", 3.0),
    ("accent", "quadro", 3.0),
    ("accent-solid", "surface", 3.0),
    ("accent-solid", "raised", 3.0),
    ("edge", "ground", 3.0),
    ("edge", "surface", 3.0),
    ("edge", "raised", 3.0),
    ("leve", "surface", 3.0),
    ("observado", "ground", 3.0),
    ("camera-0", "surface", 3.0),
    ("camera-1", "surface", 3.0),
    ("camera-2", "surface", 3.0),
    ("camera-3", "surface", 3.0),
    ("erro-borda", "erro-fundo", 3.0),
    ("historico-borda", "historico-fundo", 3.0),
]

# Ids de elementos não confundem a busca de cores: só olhamos corpos de regra.
HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
FUNCOES_COR = re.compile(r"\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(", re.I)


def _css():
    return CSS.read_text(encoding="utf-8")


def _corpo(css, cabecalho):
    """Conteúdo entre as chaves do primeiro bloco cujo cabeçalho é `cabecalho`."""
    inicio = css.index(cabecalho)
    abre = css.index("{", inicio)
    nivel = 0
    for i in range(abre, len(css)):
        if css[i] == "{":
            nivel += 1
        elif css[i] == "}":
            nivel -= 1
            if nivel == 0:
                return css[abre + 1 : i], inicio, i + 1
    raise AssertionError(f"bloco sem fechamento: {cabecalho}")


def _tokens(corpo):
    return {
        nome: valor.strip() for nome, valor in re.findall(r"--([a-z0-9-]+)\s*:\s*([^;]+);", corpo)
    }


def _blocos():
    css = _css()
    claro, *_ = _corpo(css, ":root {")
    media, *_ = _corpo(css, "@media (prefers-color-scheme: dark)")
    escuro_media, *_ = _corpo(media, ':root:not([data-theme="light"])')
    escuro_attr, *_ = _corpo(css, ':root[data-theme="dark"]')
    return _tokens(claro), _tokens(escuro_media), _tokens(escuro_attr)


def _sem_blocos_de_tokens():
    css = _css()
    for cabecalho in (
        ":root {",
        "@media (prefers-color-scheme: dark)",
        ':root[data-theme="dark"]',
    ):
        _, ini, fim = _corpo(css, cabecalho)
        css = css[:ini] + css[fim:]
    return css


def _luminancia(hexa):
    hexa = hexa.lstrip("#")[:6]
    canais = [int(hexa[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in canais]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _razao(a, b):
    mais, menos = sorted((_luminancia(a), _luminancia(b)), reverse=True)
    return (mais + 0.05) / (menos + 0.05)


def _temas():
    claro, escuro_media, escuro_attr = _blocos()
    return {"light": claro, "dark": escuro_attr}


def test_tokens_definidos_nos_tres_blocos_com_mesmo_conjunto():
    claro, escuro_media, escuro_attr = _blocos()
    tematicos = {n for n, v in claro.items() if v.startswith("#")}
    assert tematicos, "nenhum token de cor no :root"
    assert set(escuro_media) == tematicos
    assert set(escuro_attr) == tematicos
    assert escuro_media == escuro_attr


@pytest.mark.parametrize("tema", ["light", "dark"])
def test_tokens_compartilhados_com_report_tem_mesmos_valores(tema):
    tokens = _temas()[tema]
    for nome, valor in TOKENS_REPORT[tema].items():
        assert tokens[nome].upper() == valor, nome


def test_seletores_de_tema_e_color_scheme_seguem_o_padrao_do_report():
    css = _css()
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
    claro, *_ = _blocos()
    locais = {"camera-color"}  # definido por regra .camera-card
    usados = set(re.findall(r"var\(--([a-z0-9-]+)", _css()))
    assert usados - set(claro) - locais == set()


@pytest.mark.parametrize("tema", ["light", "dark"])
@pytest.mark.parametrize(("fg", "bg", "minimo"), PARES)
def test_contraste_dos_pares_declarados(tema, fg, bg, minimo):
    tokens = _temas()[tema]
    razao = _razao(tokens[fg], tokens[bg])
    assert razao >= minimo, f"{tema}: {fg} sobre {bg} = {razao:.2f} < {minimo}"


def test_nenhum_font_size_abaixo_de_12px():
    css = _css()
    achados = []
    for prop, valor in re.findall(r"(font-size|font)\s*:\s*([^;}]+)", css):
        # No shorthand `font` só o primeiro tamanho é font-size (o resto é line-height).
        tamanhos = re.findall(r"(?<![\w.#-])(\d*\.?\d+)(px|rem|em)\b", valor)[:1]
        for numero, unidade in tamanhos:
            px = float(numero) * (1 if unidade == "px" else 16)
            achados.append((prop, valor.strip(), px))
    assert achados
    pequenos = [a for a in achados if a[2] < 12]
    assert pequenos == []


def test_font_size_em_px_nao_e_usado_para_texto_da_interface():
    # Texto em rem escala com a preferência do usuário; só o SVG da planta usa px.
    css = _css()
    em_px = re.findall(r"([^{}]+)\{[^{}]*font(?:-size)?\s*:[^;}]*\d+px", css)
    assert [s.strip() for s in em_px] == ["#map text"]


def test_foco_visivel_cobre_controles_interativos():
    css = _css()
    regra = re.search(r"([^{}]*:focus-visible[^{}]*)\{([^{}]*)\}", css)
    assert regra
    seletores = regra.group(1)
    for alvo in ("button", "select", "input", "textarea", "summary", "a", "[tabindex]"):
        assert f"{alvo}:focus-visible" in seletores
    assert "outline: 2px solid var(--accent)" in regra.group(2)


def test_animacoes_so_com_reduced_motion_no_preference():
    css = _css()
    sem_no_pref = css
    for cabecalho in (
        "@media (prefers-reduced-motion: no-preference)",
        "@media (prefers-reduced-motion: reduce)",
    ):
        _, ini, fim = _corpo(sem_no_pref, cabecalho)
        sem_no_pref = sem_no_pref[:ini] + sem_no_pref[fim:]
    assert not re.search(r"\b(?:transition|animation)\b", sem_no_pref)
    assert not re.search(r"scroll-behavior\s*:\s*smooth", sem_no_pref)

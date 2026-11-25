"""Quem pode falar com a tela ao vivo: Host, Origin, tipo do corpo e prazo.

A tela escuta na máquina do operador, e o navegador dele abre outras páginas
ao mesmo tempo. Qualquer uma delas pode mandar um POST para 127.0.0.1 -- um
"pedido simples", com `text/plain`, nem passa por preflight -- e uma âncora
fantasma gera alerta contra quem passar ali. Um domínio que passa a resolver
para 127.0.0.1 (DNS rebinding) leria o estado e o vídeo com o Host dele.
"""

import http.client
import json
import socket
import threading
import time

import numpy as np
import pytest

from custody_watch.painel import Painel
from custody_watch.servidor import Servidor, host_aceito, nomes_aceitos
from tests.test_servidor import _sessao_viva


@pytest.fixture
def com_sessao():
    painel = Painel()
    sessao = _sessao_viva()
    painel.atualiza(np.zeros((2000, 2000, 3), np.uint8), {})
    servidor = Servidor(painel, "127.0.0.1", 0, sessao=sessao)
    thread = threading.Thread(target=servidor.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    yield sessao, servidor
    servidor.encerra()
    thread.join(5.0)


def _pede(
    porta: int,
    metodo: str,
    rota: str,
    corpo: bytes | None = None,
    cabecalhos: dict | None = None,
) -> tuple[int, str, bytes]:
    """Pedido cru: `http.client` respeita o Host que o teste escolher."""
    cabecalhos = {"Host": f"127.0.0.1:{porta}", **(cabecalhos or {})}
    conexao = http.client.HTTPConnection("127.0.0.1", porta, timeout=5.0)
    try:
        conexao.request(metodo, rota, body=corpo, headers=cabecalhos)
        resposta = conexao.getresponse()
        return resposta.status, resposta.getheader("Content-Type", ""), resposta.read()
    finally:
        conexao.close()


def _json(carga: dict) -> bytes:
    return json.dumps(carga).encode("utf-8")


JSON = {"Content-Type": "application/json"}
CLIQUE = _json({"x": 0.525, "y": 0.5})


def _erro_em_json(tipo: str, corpo: bytes) -> str:
    assert tipo.startswith("application/json")
    motivo = json.loads(corpo)["erro"]
    assert isinstance(motivo, str) and motivo
    return motivo


# --- Origin: só a própria página aponta -------------------------------------


@pytest.mark.parametrize(
    "origem",
    [
        "http://evil.example",
        "null",
        "https://127.0.0.1:{porta}",  # outro esquema
        "http://127.0.0.1:1",  # outra porta
        "http://localhost:{porta}",  # outro host que o do pedido
        "http://127.0.0.1:{porta}/caminho",  # Origin nunca tem caminho
        "http://[::1",  # colchete sem par
    ],
)
@pytest.mark.parametrize("tipo", ["application/json", "text/plain"])
def test_post_de_outra_origem_e_recusado_sem_tocar_a_sessao(com_sessao, origem, tipo):
    """O ataque real é o pedido simples (`text/plain`), que o navegador manda
    de qualquer página sem perguntar nada ao servidor."""
    sessao, servidor = com_sessao
    cabecalhos = {"Content-Type": tipo, "Origin": origem.format(porta=servidor.porta)}

    status, tipo_resposta, corpo = _pede(servidor.porta, "POST", "/ancora", CLIQUE, cabecalhos)

    assert status == 403
    _erro_em_json(tipo_resposta, corpo)
    assert sessao.anchors() == []
    assert sessao.apontadas() == {}


def test_dono_de_outra_origem_e_recusado(com_sessao):
    sessao, servidor = com_sessao
    status, _, corpo = _pede(servidor.porta, "POST", "/ancora", CLIQUE, JSON)
    assert status == 200
    bagagem = json.loads(corpo)["bagagem"]

    status, _, _ = _pede(
        servidor.porta,
        "POST",
        "/dono",
        _json({"bagagem": bagagem, "x": 0.5, "y": 0.49}),
        {"Content-Type": "text/plain", "Origin": "http://evil.example"},
    )

    assert status == 403
    assert next(b for b in sessao.anchors() if b.bag_id == bagagem).owner_party is None


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
def test_post_da_propria_pagina_funciona(com_sessao, host):
    """Mesmo esquema, host e porta do Host: é a página que o servidor serviu."""
    sessao, servidor = com_sessao
    autoridade = f"{host}:{servidor.porta}"

    status, _, corpo = _pede(
        servidor.porta,
        "POST",
        "/ancora",
        CLIQUE,
        {**JSON, "Host": autoridade, "Origin": f"http://{autoridade}"},
    )

    assert status == 200
    assert [b.bag_id for b in sessao.anchors()] == [json.loads(corpo)["bagagem"]]


@pytest.mark.parametrize(
    "tipo", ["application/json", "application/json; charset=utf-8", "Application/JSON"]
)
def test_post_sem_origin_de_cliente_fora_do_navegador_funciona(com_sessao, tipo):
    """curl e scripts não mandam Origin; quem manda é o navegador, sempre."""
    sessao, servidor = com_sessao

    status, _, _ = _pede(servidor.porta, "POST", "/ancora", CLIQUE, {"Content-Type": tipo})

    assert status == 200
    assert len(sessao.anchors()) == 1


# --- Content-Type: só JSON ----------------------------------------------------


@pytest.mark.parametrize(
    "tipo",
    [None, "text/plain", "application/x-www-form-urlencoded", "multipart/form-data", "json"],
)
def test_post_sem_tipo_json_e_recusado_sem_tocar_a_sessao(com_sessao, tipo):
    """Exigir JSON força o navegador a perguntar antes (preflight) a qualquer
    página de outra origem -- e esta tela não responde que sim."""
    sessao, servidor = com_sessao
    cabecalhos = {} if tipo is None else {"Content-Type": tipo}

    status, tipo_resposta, corpo = _pede(servidor.porta, "POST", "/ancora", CLIQUE, cabecalhos)

    assert status == 415
    _erro_em_json(tipo_resposta, corpo)
    assert sessao.anchors() == []


def test_pagina_aponta_com_corpo_json():
    """A recusa acima só não quebra a tela porque a página manda JSON."""
    servidor = Servidor(Painel(), "127.0.0.1", 0)
    thread = threading.Thread(target=servidor.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        status, _, corpo = _pede(servidor.porta, "GET", "/")
    finally:
        servidor.encerra()
        thread.join(5.0)

    pagina = corpo.decode("utf-8")
    assert status == 200
    assert 'method: "POST"' in pagina
    assert 'headers: { "Content-Type": "application/json" }' in pagina


# --- Host: só os nomes desta tela -------------------------------------------


HOSTS_ESTRANHOS = [
    "evil.example",
    "evil.example:{porta}",
    "127.0.0.1.evil.example:{porta}",
    "localhost.:{porta}",
    "127.0.0.1:1",  # porta errada
    "127.0.0.1",  # sem porta, e a porta do servidor não é a 80
    "usuario@127.0.0.1:{porta}",
    "127.0.0.1:{porta}/x",
    "[::1:{porta}",  # colchete sem par
    "",
]


@pytest.mark.parametrize("host", HOSTS_ESTRANHOS)
@pytest.mark.parametrize("rota", ["/", "/estado", "/video", "/quadro.jpg"])
def test_get_com_host_estranho_e_recusado(com_sessao, host, rota):
    """DNS rebinding: a página do atacante passa a resolver para 127.0.0.1 e
    lê o estado e o vídeo -- mas o navegador manda o Host dela."""
    _, servidor = com_sessao

    status, tipo, corpo = _pede(
        servidor.porta, "GET", rota, cabecalhos={"Host": host.format(porta=servidor.porta)}
    )

    assert status == 421
    _erro_em_json(tipo, corpo)


@pytest.mark.parametrize("host", HOSTS_ESTRANHOS)
def test_post_com_host_estranho_e_recusado_sem_tocar_a_sessao(com_sessao, host):
    sessao, servidor = com_sessao

    status, _, _ = _pede(
        servidor.porta,
        "POST",
        "/ancora",
        CLIQUE,
        {**JSON, "Host": host.format(porta=servidor.porta)},
    )

    assert status == 421
    assert sessao.anchors() == []


def _ate_fechar(cru: socket.socket) -> bytes:
    """Lê a resposta inteira. Fechar no meio dela faria o servidor escrever
    num socket já fechado, e o teste mediria o próprio atropelo."""
    partes = []
    while parte := cru.recv(4096):
        partes.append(parte)
    return b"".join(partes)


def test_pedido_sem_host_e_recusado(com_sessao):
    _, servidor = com_sessao
    with socket.create_connection(("127.0.0.1", servidor.porta), timeout=3.0) as cru:
        cru.sendall(b"GET /estado HTTP/1.0\r\n\r\n")
        resposta = _ate_fechar(cru)

    assert resposta.startswith(b"HTTP/1.0 421")


def test_host_repetido_e_recusado(com_sessao):
    _, servidor = com_sessao
    porta = servidor.porta
    with socket.create_connection(("127.0.0.1", porta), timeout=3.0) as cru:
        cabecalhos = f"Host: 127.0.0.1:{porta}\r\nHost: evil.example\r\n"
        cru.sendall(f"GET /estado HTTP/1.1\r\n{cabecalhos}\r\n".encode())
        resposta = _ate_fechar(cru)

    assert resposta.startswith(b"HTTP/1.0 421")


@pytest.mark.parametrize(
    "host", ["127.0.0.1:{porta}", "localhost:{porta}", "LocalHost:{porta}", "[::1]:{porta}"]
)
def test_host_desta_maquina_e_aceito(com_sessao, host):
    _, servidor = com_sessao

    status, _, _ = _pede(
        servidor.porta, "GET", "/estado", cabecalhos={"Host": host.format(porta=servidor.porta)}
    )

    assert status == 200


def test_nomes_extras_sao_aceitos_so_quando_declarados():
    """Um nome de rede para a tela é decisão de quem a sobe, como o próprio
    endereço de escuta: parâmetro explícito, nunca deduzido do pedido."""
    painel = Painel()
    servidor = Servidor(painel, "127.0.0.1", 0, nomes_permitidos=["Tela.Local"])
    thread = threading.Thread(target=servidor.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        porta = servidor.porta
        aceito, *_ = _pede(porta, "GET", "/estado", cabecalhos={"Host": f"tela.local:{porta}"})
        outro, *_ = _pede(porta, "GET", "/estado", cabecalhos={"Host": f"outra.local:{porta}"})
    finally:
        servidor.encerra()
        thread.join(5.0)

    assert aceito == 200
    assert outro == 421


def test_nomes_aceitos_incluem_o_endereco_de_escuta_de_rede():
    assert {"localhost", "127.0.0.1", "::1"} <= nomes_aceitos("127.0.0.1", ())
    assert "192.168.0.10" in nomes_aceitos("192.168.0.10", ())
    assert "camera.aeroporto.local" in nomes_aceitos("Camera.Aeroporto.Local", ())
    assert "fe80::1" in nomes_aceitos("[fe80::1]", ())
    assert "painel.local" in nomes_aceitos("127.0.0.1", ["Painel.Local"])
    # Curinga não é nome de ninguém: o navegador nunca o manda como Host.
    assert "0.0.0.0" not in nomes_aceitos("0.0.0.0", ())
    assert "::" not in nomes_aceitos("::", ())


def test_porta_padrao_do_esquema_pode_faltar_no_host():
    nomes = nomes_aceitos("127.0.0.1", ())

    assert host_aceito("localhost", nomes, porta=80, porta_padrao=80)
    assert host_aceito("localhost:80", nomes, porta=80, porta_padrao=80)
    assert host_aceito("localhost", nomes, porta=443, porta_padrao=443)
    assert not host_aceito("localhost", nomes, porta=8765, porta_padrao=80)
    assert not host_aceito("localhost:abc", nomes, porta=80, porta_padrao=80)
    assert not host_aceito(None, nomes, porta=80, porta_padrao=80)


def test_escuta_em_curinga_aceita_ip_literal_mas_nao_nome_desconhecido():
    """Em 0.0.0.0 o operador abre a tela pelo IP da máquina, que o servidor
    não conhece de antemão. IP literal não passa por DNS, então não serve a
    rebinding; nome desconhecido continua recusado."""
    nomes = nomes_aceitos("0.0.0.0", ())

    assert host_aceito("10.1.2.3:8765", nomes, porta=8765, porta_padrao=443, qualquer_ip=True)
    assert host_aceito("[fe80::1]:8765", nomes, porta=8765, porta_padrao=443, qualquer_ip=True)
    assert not host_aceito(
        "evil.example:8765", nomes, porta=8765, porta_padrao=443, qualquer_ip=True
    )
    assert not host_aceito("10.1.2.3:8765", nomes, porta=8765, porta_padrao=443)


# --- Prazo de leitura -------------------------------------------------------


def test_corpo_incompleto_nao_prende_a_conexao(com_sessao):
    """Content-Length maior que o corpo: sem prazo, a thread esperaria para
    sempre, e cada conexão assim é uma thread a menos no processo."""
    sessao, servidor = com_sessao
    servidor.prazo_leitura_s = 0.3
    porta = servidor.porta
    inicio = time.monotonic()
    with socket.create_connection(("127.0.0.1", porta), timeout=3.0) as cru:
        cru.sendall(
            (
                f"POST /ancora HTTP/1.1\r\nHost: 127.0.0.1:{porta}\r\n"
                "Content-Type: application/json\r\nContent-Length: 100\r\n\r\n{"
            ).encode()
        )
        try:
            resposta = _ate_fechar(cru)
        except TimeoutError:
            pytest.fail("servidor não respondeu em 3 s a um corpo incompleto")

    assert time.monotonic() - inicio < 3.0
    assert resposta.startswith(b"HTTP/1.0 408")
    assert sessao.anchors() == []
    assert _pede(porta, "GET", "/estado")[0] == 200


def test_conexao_muda_e_encerrada_no_prazo(com_sessao):
    _, servidor = com_sessao
    servidor.prazo_leitura_s = 0.3
    inicio = time.monotonic()
    with socket.create_connection(("127.0.0.1", servidor.porta), timeout=3.0) as cru:
        try:
            resto = cru.recv(200)
        except TimeoutError:
            pytest.fail("conexão muda continuou aberta por 3 s")

    assert time.monotonic() - inicio < 3.0
    assert resto == b""


def test_conexao_muda_no_prazo_nao_escreve_no_terminal(com_sessao, capsys):
    """Navegador abre conexão especulativa e nunca a usa. Uma linha por
    conexão dessas a cada prazo encheria o terminal do operador no caminho
    normal -- o mesmo motivo de `log_request` ser silencioso."""
    _, servidor = com_sessao
    servidor.prazo_leitura_s = 0.3
    capsys.readouterr()
    with socket.create_connection(("127.0.0.1", servidor.porta), timeout=3.0) as cru:
        assert _ate_fechar(cru) == b""

    assert capsys.readouterr().err == ""


def test_erro_de_pedido_continua_no_terminal(com_sessao, capsys):
    """Silenciar a conexão muda não pode calar erro de verdade."""
    _, servidor = com_sessao
    capsys.readouterr()

    status, *_ = _pede(servidor.porta, "PUT", "/estado")

    assert status == 501
    assert "501" in capsys.readouterr().err


def test_prazo_de_leitura_padrao_e_finito():
    servidor = Servidor(Painel(), "127.0.0.1", 0)
    try:
        assert 0 < servidor.prazo_leitura_s <= 60
    finally:
        servidor.server_close()


# --- JSON malformado ---------------------------------------------------------


def test_json_aninhado_demais_devolve_400_em_json(com_sessao):
    """Cabe no teto de bytes e ainda assim estoura a pilha do parser:
    `RecursionError` fora do `except` fechava a conexão sem resposta."""
    sessao, servidor = com_sessao
    corpo = b"[" * 4000

    status, tipo, resposta = _pede(servidor.porta, "POST", "/ancora", corpo, JSON)

    assert status == 400
    _erro_em_json(tipo, resposta)
    assert sessao.anchors() == []


# --- Números não finitos ------------------------------------------------------


@pytest.mark.parametrize(
    ("rota", "corpo"),
    [
        ("/dono", b'{"x": 0.5, "y": 0.5, "bagagem": Infinity}'),
        ("/dono", b'{"x": 0.5, "y": 0.5, "bagagem": -Infinity}'),
        ("/dono", b'{"x": 0.5, "y": 0.5, "bagagem": 1e999}'),
        ("/dono", b'{"x": 0.5, "y": 0.5, "bagagem": NaN}'),
        ("/dono", b'{"x": 0.5, "y": 0.5, "bagagem": 1' + b"0" * 400 + b"}"),
        ("/ancora", b'{"x": Infinity, "y": 0.5}'),
        ("/ancora", b'{"x": 0.5, "y": -1e999}'),
        ("/ancora", b'{"x": NaN, "y": 0.5}'),
        ("/ancora", b'{"x": 1' + b"0" * 400 + b', "y": 0.5}'),
        ("/dono", b'{"x": 1' + b"0" * 400 + b', "y": 0.5, "bagagem": 1}'),
    ],
)
def test_numero_nao_finito_devolve_400_em_json(com_sessao, rota, corpo):
    """Antes, `int(inf)` estourava fora do `except` e a conexão fechava sem
    resposta: a tela mostrava "recusado" sem dizer por quê."""
    sessao, servidor = com_sessao

    status, tipo, resposta = _pede(servidor.porta, "POST", rota, corpo, JSON)

    assert status == 400
    _erro_em_json(tipo, resposta)
    assert sessao.anchors() == []

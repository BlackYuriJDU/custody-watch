"""Host do laboratório multicâmera: só os nomes desta máquina (DNS rebinding).

Um domínio do atacante que passa a resolver para 127.0.0.1 leria os dados do
laboratório com o Host dele; a recusa tem de sair antes de qualquer rota.
"""

import http.client
import importlib.util
import socket
import threading
from pathlib import Path

import pytest

from custody_watch.camera_registry import load_camera_registry
from custody_watch.multicamera_media import FrameImages
from custody_watch.multicamera_server import ReplayServer
from custody_watch.multicamera_view import ReplayView
from tests.multicamera_fixture import make_registry

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serve_multicamera.py"
SPEC = importlib.util.spec_from_file_location("custody_serve_multicamera_script", SCRIPT)
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)

ROTAS = ["/", "/api/config", "/api/state?t=0"]


def _sobe(tmp_path, **opcoes):
    registry = load_camera_registry(make_registry(tmp_path))
    server = ReplayServer(ReplayView(registry), FrameImages(registry), port=0, **opcoes)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    return server, thread


def _derruba(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(5)
    assert not thread.is_alive()


@pytest.fixture
def servidor(tmp_path):
    server, thread = _sobe(tmp_path)
    try:
        yield server
    finally:
        _derruba(server, thread)


@pytest.fixture
def servidor_com_nome(tmp_path):
    server, thread = _sobe(tmp_path, allowed_hosts=["Lab.Local"])
    try:
        yield server
    finally:
        _derruba(server, thread)


def _pede(porta: int, host: str | None, rota: str, metodo: str = "GET", corpo=None):
    conexao = http.client.HTTPConnection("127.0.0.1", porta, timeout=5.0)
    try:
        conexao.putrequest(metodo, rota, skip_host=True, skip_accept_encoding=True)
        if host is not None:
            conexao.putheader("Host", host)
        if corpo is not None:
            conexao.putheader("Content-Length", str(len(corpo)))
        conexao.endheaders(corpo)
        resposta = conexao.getresponse()
        return resposta.status, resposta.read()
    finally:
        conexao.close()


def _http10_sem_host(porta: int) -> bytes:
    with socket.create_connection(("127.0.0.1", porta), timeout=5.0) as cru:
        cru.sendall(b"GET /api/config HTTP/1.0\r\n\r\n")
        partes = []
        while parte := cru.recv(4096):
            partes.append(parte)
    return b"".join(partes)


@pytest.mark.parametrize("rota", ROTAS)
@pytest.mark.parametrize(
    "host",
    ["evil.example", "evil.example:{porta}", "127.0.0.1:1", "127.0.0.1.evil.example:{porta}"],
)
def test_host_estranho_e_421_sem_vazar_conteudo(servidor, host, rota):
    status, corpo = _pede(servidor.server_port, host.format(porta=servidor.server_port), rota)

    assert status == 421
    assert b"registry_sha256" not in corpo
    assert b"multicamera.js" not in corpo


@pytest.mark.parametrize("host", ["127.0.0.1:{porta}", "localhost:{porta}", "[::1]:{porta}"])
@pytest.mark.parametrize("rota", ROTAS)
def test_host_local_com_a_porta_certa_e_200(servidor, host, rota):
    status, corpo = _pede(servidor.server_port, host.format(porta=servidor.server_port), rota)

    assert status == 200
    assert corpo


def test_host_ausente_http10_e_421_como_no_painel(servidor):
    resposta = _http10_sem_host(servidor.server_port)

    assert resposta.startswith(b"HTTP/1.0 421")
    assert b"registry_sha256" not in resposta


def test_host_repetido_e_421(servidor):
    porta = servidor.server_port
    with socket.create_connection(("127.0.0.1", porta), timeout=5.0) as cru:
        pedido = (
            f"GET /api/config HTTP/1.1\r\nHost: 127.0.0.1:{porta}\r\nHost: evil.example\r\n\r\n"
        )
        cru.sendall(pedido.encode())
        resposta = cru.recv(4096)

    assert resposta.split(b" ", 2)[1] == b"421"


def test_nome_adicional_permitido_e_200_e_outro_continua_421(servidor_com_nome):
    porta = servidor_com_nome.server_port

    aceito, _ = _pede(porta, f"lab.local:{porta}", "/api/config")
    outro, _ = _pede(porta, f"outra.local:{porta}", "/api/config")
    porta_errada, _ = _pede(porta, "lab.local:1", "/api/config")

    assert aceito == 200
    assert outro == 421
    assert porta_errada == 421


def test_nome_adicional_nao_tira_os_locais(servidor_com_nome):
    porta = servidor_com_nome.server_port

    status, _ = _pede(porta, f"127.0.0.1:{porta}", "/api/config")

    assert status == 200


@pytest.mark.parametrize("metodo", ["HEAD", "POST", "PUT", "DELETE", "PATCH"])
def test_host_estranho_recusado_antes_do_metodo_e_sem_rst(servidor, metodo):
    """421 também para escrita: a recusa vem antes de decidir 501, e o corpo
    pequeno é drenado para o cliente receber a resposta em vez de RST."""
    corpo = None if metodo in {"HEAD"} else b'{"x": 1}'

    status, _ = _pede(servidor.server_port, "evil.example", "/api/config", metodo, corpo)

    assert status == 421


def test_escrita_com_host_certo_continua_501(servidor):
    porta = servidor.server_port

    status, _ = _pede(porta, f"127.0.0.1:{porta}", "/api/config", "POST", b"{}")

    assert status == 501


# --- scripts/serve_multicamera.py: --allowed-host ------------------------------


def test_flag_allowed_host_e_repetivel_e_vazia_por_padrao():
    assert lab._argumentos(["cadastro.json"]).allowed_host == []

    args = lab._argumentos(
        ["cadastro.json", "--allowed-host", "a.local", "--allowed-host", "b.local"]
    )

    assert args.allowed_host == ["a.local", "b.local"]


@pytest.mark.parametrize("nome", ["", "a b", "a/b", "u@a.local", "a.local:8766", "http://a.local"])
def test_flag_allowed_host_recusa_nome_invalido(nome):
    with pytest.raises(SystemExit) as erro:
        lab._argumentos(["cadastro.json", "--allowed-host", nome])

    assert erro.value.code == 2


def test_main_repassa_nomes_permitidos_ao_servidor(tmp_path, monkeypatch):
    recebido = {}

    class Falso:
        def __init__(self, view, images, **opcoes):
            recebido.update(opcoes)
            raise OSError("parada do teste")

    monkeypatch.setattr(lab, "ReplayServer", Falso)
    monkeypatch.setattr(
        lab.sys,
        "argv",
        [
            "serve_multicamera.py",
            str(make_registry(tmp_path)),
            "--allowed-host",
            "lab.local",
        ],
    )

    assert lab.main() == 1
    assert list(recebido["allowed_hosts"]) == ["lab.local"]

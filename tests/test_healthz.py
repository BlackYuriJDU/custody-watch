"""`GET /healthz`: sonda de monitoração da tela ao vivo e do laboratório.

O corpo é mínimo e estável: sem vídeo, pessoas, bagagens nem configuração. A
rota passa pelas mesmas defesas das outras (Host, 421) e não aceita escrita.
"""

import http.client
import json
import threading

import numpy as np
import pytest

from custody_watch.painel import Painel
from custody_watch.servidor import Servidor
from tests.test_multicamera_host import _derruba, _sobe

CORPO_ESPERADO = {"status": "ok"}


@pytest.fixture(params=["ao_vivo", "laboratorio"])
def porta(request, tmp_path):
    if request.param == "laboratorio":
        server, thread = _sobe(tmp_path)
        try:
            yield server.server_port
        finally:
            _derruba(server, thread)
        return
    painel = Painel()
    painel.atualiza(np.zeros((8, 8, 3), np.uint8), {})
    servidor = Servidor(painel, "127.0.0.1", 0)
    thread = threading.Thread(target=servidor.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        yield servidor.porta
    finally:
        servidor.encerra()
        thread.join(5.0)


def _pede(porta: int, metodo: str, rota: str, host: str | None = None, corpo: bytes | None = None):
    conexao = http.client.HTTPConnection("127.0.0.1", porta, timeout=5.0)
    try:
        conexao.putrequest(metodo, rota, skip_host=True, skip_accept_encoding=True)
        conexao.putheader("Host", host or f"127.0.0.1:{porta}")
        conexao.putheader("Content-Length", str(len(corpo or b"")))
        if corpo is not None:
            conexao.putheader("Content-Type", "application/json")
        conexao.endheaders(corpo)
        resposta = conexao.getresponse()
        return resposta.status, resposta.getheaders(), resposta.read()
    finally:
        conexao.close()


def _cabecalho(cabecalhos, nome: str) -> str:
    return next((v for k, v in cabecalhos if k.lower() == nome.lower()), "")


def test_healthz_responde_200_json_com_corpo_minimo(porta):
    status, cabecalhos, corpo = _pede(porta, "GET", "/healthz")
    assert status == 200
    assert _cabecalho(cabecalhos, "Content-Type").startswith("application/json")
    assert _cabecalho(cabecalhos, "Cache-Control") == "no-store"
    assert json.loads(corpo) == CORPO_ESPERADO


def test_healthz_so_traz_as_chaves_documentadas(porta):
    _, _, corpo = _pede(porta, "GET", "/healthz")
    assert set(json.loads(corpo)) == set(CORPO_ESPERADO)


def test_healthz_com_host_estranho_e_421(porta):
    status, _, corpo = _pede(porta, "GET", "/healthz", host="evil.example")
    assert status == 421
    assert b"ok" not in corpo


@pytest.mark.parametrize("metodo", ["POST", "PUT", "DELETE"])
def test_escrita_em_healthz_tem_o_tratamento_de_rota_nao_gravavel(porta, metodo):
    """Mesmo status que uma rota só de leitura já existente (`/api/config` no
    laboratório, `/estado` na tela ao vivo)."""
    referencia = "/estado" if "/estado" in _rotas_ao_vivo(porta) else "/api/config"
    esperado, _, _ = _pede(porta, metodo, referencia, corpo=b"{}")
    status, _, corpo = _pede(porta, metodo, "/healthz", corpo=b"{}")
    assert status == esperado
    assert status != 200
    assert json.dumps(CORPO_ESPERADO).encode() not in corpo


def _rotas_ao_vivo(porta: int) -> list[str]:
    status, _, _ = _pede(porta, "GET", "/estado")
    return ["/estado"] if status == 200 else []

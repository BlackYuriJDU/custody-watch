"""Método que a tela ao vivo não atende é recusado sem derrubar a conexão.

`BaseHTTPRequestHandler` responde 501 a método sem `do_*` e fecha sem ler o
corpo. No Windows, fechar com bytes não lidos no socket vira RST, e o cliente
perde a resposta (`ConnectionAbortedError`, WinError 10053). Era a falha
intermitente de `test_healthz` vista nos hooks de #124 e #128.
"""

import http.client
import threading

import numpy as np
import pytest

from custody_watch.painel import Painel
from custody_watch.servidor import Servidor

REPETICOES = 150
"""A corrida entre o RST e a leitura da resposta é rara por pedido; repetir a
torna visível sem depender de sorte numa única chamada."""
CORPO = b"x" * 2048


@pytest.fixture
def porta():
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


def _status(porta: int, metodo: str, rota: str) -> int:
    conexao = http.client.HTTPConnection("127.0.0.1", porta, timeout=5.0)
    try:
        conexao.request(metodo, rota, body=CORPO, headers={"Content-Type": "application/json"})
        resposta = conexao.getresponse()
        resposta.read()
        return resposta.status
    finally:
        conexao.close()


@pytest.mark.parametrize(
    ("metodo", "rota", "esperado"),
    [
        ("DELETE", "/healthz", 501),
        ("PUT", "/estado", 501),
        ("PATCH", "/", 501),
        ("POST", "/healthz", 404),
    ],
)
def test_recusa_com_corpo_chega_inteira_ao_cliente(porta, metodo, rota, esperado):
    recebidos = [_status(porta, metodo, rota) for _ in range(REPETICOES)]
    assert recebidos == [esperado] * REPETICOES

import argparse
import sqlite3
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from custody_watch.config import Config
from custody_watch.events import EventLog
from custody_watch.journal_reader import iter_journal
from custody_watch.live_recovery import LiveRecovery, RecoveryPolicy
from custody_watch.painel import Painel
from custody_watch.source_health import SourceHealth
from custody_watch.types import BagState, Observation, Point
from tests.test_live_recovery import Capture
from tests.test_occlusion import PLANO
from tests.test_serve import serve


def args_for(root, *, reconnect=0):
    return argparse.Namespace(
        journal_dir=root,
        journal_part_records=10,
        journal_fsync=False,
        event_tail=8,
        archive_terminal_bags=True,
        reconnect_attempts=reconnect,
    )


def test_flag_exige_journal_e_default_fica_desativado(monkeypatch):
    monkeypatch.setattr("sys.argv", ["serve.py", "0"])
    assert serve._argumentos().archive_terminal_bags is False
    monkeypatch.setattr("sys.argv", ["serve.py", "0", "--archive-terminal-bags"])
    with pytest.raises(SystemExit) as caught:
        serve._argumentos()
    assert caught.value.code == 2
    monkeypatch.setattr(
        "sys.argv", ["serve.py", "0", "--archive-terminal-bags", "--journal-dir", "unused"]
    )
    assert serve._argumentos().archive_terminal_bags is True


def test_sessao_cria_arquivo_no_segmento_e_telemetria_sem_ler_historico(tmp_path, monkeypatch):
    session, writer = serve._session(args_for(tmp_path), PLANO, Config())
    try:
        path = writer.directory / f"terminal-{writer.segment_id}.sqlite"
        assert path.is_file()
        bag = session._s.registry.observe(Observation(1, "suitcase", Point(0, 0), 0))
        bag.owner_party = 44
        bag.state = BagState.RETIRADA_DONO
        session._s.registry.link_track(11, 1)
        session.feed(1, [])

        def forbidden(*a, **kw):
            pytest.fail("telemetria HTTP não pode consultar SQLite/histórico")

        monkeypatch.setattr(session._s.registry._archive, "counts", forbidden)
        monkeypatch.setattr(session._s.registry._archive, "all", forbidden)
        state = serve._journal_status(writer, session)["bag_retention"]
        assert state == {
            "enabled": True,
            "state": "ready",
            "resident_bags": 0,
            "resident_links": 0,
            "archived_bags": 1,
            "archived_links": 1,
        }
        assert session.bag_for_track(1).owner_party == 44
        session._s.registry.link_track(12, 1)
        session._s.registry.link_track(12, 1)
        session.feed(2, [])
        assert session.retention_status()["archived_links"] == 2
    finally:
        writer.close()


def test_sessao_provisoria_nao_ocupa_arquivo_da_primeira_captura(tmp_path):
    args = args_for(tmp_path, reconnect=2)
    provisional, writer = serve._session(args, PLANO, Config())
    try:
        assert list(writer.directory.glob("*.sqlite")) == []
        assert "bag_retention" not in serve._journal_status(writer, provisional)
        actual = serve._live_session(args, PLANO, Config(), EventLog(), writer)
        assert actual.retention_status()["enabled"] is True
        assert len(list(writer.directory.glob("*.sqlite"))) == 1
    finally:
        writer.close()


@pytest.mark.parametrize("stage", ["create", "claim"])
def test_falha_de_inicializacao_fecha_journal_sem_expor_caminho(tmp_path, monkeypatch, stage):
    def fail(*a, **kw):
        raise OSError("secret-path-in-error")

    monkeypatch.setattr(
        serve,
        "TerminalBagArchive",
        fail if stage == "create" else lambda *a: SimpleNamespace(claim=fail),
    )
    with pytest.raises(OSError, match="arquivo terminal") as caught:
        serve._session(args_for(tmp_path), PLANO, Config())
    assert "secret-path" not in str(caught.value)
    (directory,) = tmp_path.iterdir()
    records = list(iter_journal(directory))
    assert records[-1]["payload"]["reason"] == "initialization_failed"


def test_falha_de_leitura_e_telemetria_sem_reabrir_custodia(tmp_path):
    session, writer = serve._session(args_for(tmp_path), PLANO, Config())
    try:
        archive = session._s.registry._archive
        archive.path.rename(writer.directory / "preserved.sqlite")
        with pytest.raises(sqlite3.OperationalError):
            session.bag_for_track(123)
        state = serve._journal_status(writer, session)["bag_retention"]
        assert state["state"] == "failed"
        assert state["archived_bags"] == 0
        with pytest.raises(OSError, match="arquivo terminal"):
            session.feed(1, [])
        assert session._s.registry.resident() == []
    finally:
        writer.close()


def test_falha_de_gravacao_fica_visivel_e_sessao_nao_retoma_sozinha(tmp_path, monkeypatch):
    session, writer = serve._session(args_for(tmp_path), PLANO, Config())
    try:
        bag = session._s.registry.observe(Observation(1, "suitcase", Point(0, 0), 0))
        bag.state = BagState.RETIRADA_DONO

        def fail(*a, **kw):
            raise OSError("secret-disk-error")

        with monkeypatch.context() as patch:
            patch.setattr(session._s.registry._archive, "persist", fail)
            with pytest.raises(OSError):
                session.feed(1, [])
        state = serve._journal_status(writer, session)["bag_retention"]
        assert state["state"] == "failed"
        assert state["resident_bags"] == 1
        assert state["archived_bags"] == 0
        assert "secret" not in str(state)
        with pytest.raises(OSError, match="arquivo terminal"):
            session.feed(2, [])
        assert session._s.registry.get(1) is bag
    finally:
        writer.close()


def test_reconexao_usa_arquivo_novo_sem_herdar_id_ou_posse(tmp_path):
    args = args_for(tmp_path, reconnect=2)
    provisional, writer = serve._session(args, PLANO, Config())
    stop = threading.Event()
    first_done, second_done = threading.Event(), threading.Event()
    captures = [Capture(1, first_done), Capture(1, second_done)]
    sessions, opened = [], []

    def capture_factory():
        capture = captures[len(opened)]
        opened.append(capture)
        return capture

    def observed():
        session = recovery.session
        sessions.append((session, writer.segment_id))
        if len(sessions) == 1:
            bag = session._s.registry.observe(Observation(1, "suitcase", Point(0, 0), 0))
            bag.state = BagState.RETIRADA_DONO
            bag.owner_party = 44
            session._s.registry.archive_terminal_bags()
            first_done.set()
        else:
            assert session.bag_for_track(1) is None
            assert session._s.registry.archived_counts() == {"bags": 0, "bag_links": 0}
            stop.set()
            second_done.set()

    recovery = LiveRecovery(
        capture_factory=capture_factory,
        detector_factory=lambda: lambda image: [],
        session_factory=lambda log: serve._live_session(args, PLANO, Config(), log, writer),
        journal=writer,
        painel=Painel(),
        health=SourceHealth("live"),
        stop=stop,
        policy=RecoveryPolicy(attempts=2, backoff_s=0.001),
        on_processed=observed,
    )
    try:
        recovery.run()
        assert recovery.snapshot()["state"] == "stopped"
    finally:
        first_done.set()
        second_done.set()
        writer.close()
    assert len(sessions) == 2
    assert sessions[0][1] != sessions[1][1]
    assert len(list(writer.directory.glob("*.sqlite"))) == 2
    assert sessions[0][0].bag_for_track(1).owner_party == 44
    assert all(c.released for c in captures)


def test_telemetria_nao_espera_gravacao_bloqueada(tmp_path, monkeypatch):
    session, writer = serve._session(args_for(tmp_path), PLANO, Config())
    bag = session.aponta_bagagem(20, 20)
    bag.state = BagState.RETIRADA_DONO
    archive = session._s.registry._archive
    original = archive.persist
    entered, release, status_done = threading.Event(), threading.Event(), threading.Event()
    failures, states = [], []

    def blocked(*a):
        entered.set()
        assert release.wait(3)
        return original(*a)

    def feed():
        try:
            session.feed(1, [])
        except Exception as error:
            failures.append(error)

    def read_status():
        states.append(serve._journal_status(writer, session))
        status_done.set()

    monkeypatch.setattr(archive, "persist", blocked)
    worker = threading.Thread(target=feed)
    reader = threading.Thread(target=read_status)
    try:
        worker.start()
        assert entered.wait(1)
        reader.start()
        assert status_done.wait(1), "HTTP de estado aguardou a gravação do feed"
        assert states[0]["bag_retention"]["archived_bags"] == 0
        assert states[0]["bag_retention"]["resident_bags"] == 1
    finally:
        release.set()
        worker.join(3)
        if reader.ident is not None:
            reader.join(3)
        writer.close()
    assert not worker.is_alive() and not reader.is_alive()
    assert not failures
    assert session.retention_status()["archived_bags"] == 1


@pytest.mark.parametrize("stage", ["initialization", "write"])
def test_falha_de_arquivo_libera_captura_e_nao_reconecta(tmp_path, monkeypatch, stage):
    args = args_for(tmp_path, reconnect=2)
    provisional, writer = serve._session(args, PLANO, Config())
    gate = threading.Event()
    capture = Capture(1, gate)
    opened, sessions = [], []

    def capture_factory():
        opened.append(capture)
        return capture

    def fail(*a, **kw):
        gate.set()
        raise OSError("secret-storage-message")

    if stage == "initialization":
        monkeypatch.setattr(serve, "TerminalBagArchive", fail)

    def session_factory(log):
        session = serve._live_session(args, PLANO, Config(), log, writer)
        sessions.append(session)
        bag = session._s.registry.observe(Observation(1, "suitcase", Point(0, 0), 0))
        bag.state = BagState.RETIRADA_DONO
        monkeypatch.setattr(session._s.registry._archive, "persist", fail)
        return session

    recovery = LiveRecovery(
        capture_factory=capture_factory,
        detector_factory=lambda: lambda image: [],
        session_factory=session_factory,
        journal=writer,
        painel=Painel(),
        health=SourceHealth("live"),
        stop=threading.Event(),
        policy=RecoveryPolicy(attempts=2, backoff_s=0.001),
    )
    try:
        recovery.run()
        state = recovery.snapshot()
        assert state["reason"] == "archive_failed"
        assert state["retries"] == 0
        assert "secret" not in str(state)
        assert opened == [capture]
        assert capture.released
        assert recovery.camera.wait_closed(1)
        if sessions:
            retention = serve._journal_status(writer, sessions[0])["bag_retention"]
            assert retention["state"] == "failed"
            assert retention["resident_bags"] == 1
        else:
            assert stage == "initialization"
    finally:
        gate.set()
        writer.close()
    records = list(iter_journal(writer.directory))
    assert records[-1]["kind"] == "segment_interrupted"
    assert records[-1]["payload"]["reason"] == "processing_exception"


def test_main_de_arquivo_habilita_archive_e_encerra_journal(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    args.__dict__.update(
        fonte="synthetic.mp4",
        source_timeout=5,
        config=None,
        calibration=None,
        metres_per_pixel=0.01,
        cert=None,
        key=None,
        host="127.0.0.1",
        porta=0,
        weights="unused.pt",
        min_confidence=0.35,
    )
    monkeypatch.setattr(serve, "_argumentos", lambda: args)
    monkeypatch.setattr(
        serve, "_fonte", lambda *a, **kw: ([(0, np.zeros((24, 32, 3), np.uint8))], None)
    )
    monkeypatch.setattr(serve, "detector_yolo", lambda *a, **kw: lambda image: [])

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(serve.time, "sleep", interrupt)
    assert serve.main() == 0
    (folder,) = tmp_path.iterdir()
    assert len(list(folder.glob("*.sqlite"))) == 1
    records = list(iter_journal(folder))
    assert records[-1]["payload"]["reason"] == "end_of_file"


def test_main_falha_de_arquivo_retorna_erro_sem_mensagem_bruta(tmp_path, monkeypatch, capsys):
    args = args_for(tmp_path)
    args.__dict__.update(
        fonte="synthetic.mp4",
        source_timeout=5,
        config=None,
        calibration=None,
        metres_per_pixel=0.01,
        cert=None,
        key=None,
        host="127.0.0.1",
        porta=0,
        weights="unused.pt",
        min_confidence=0.35,
    )
    monkeypatch.setattr(serve, "_argumentos", lambda: args)
    monkeypatch.setattr(serve, "_fonte", lambda *a, **kw: ([], None))
    monkeypatch.setattr(serve, "detector_yolo", lambda *a, **kw: lambda image: [])

    def fail(*a, **kw):
        raise OSError("secret-storage-error")

    def process(frames, detector, session, panel, **kw):
        bag = session._s.registry.observe(Observation(1, "suitcase", Point(0, 0), 0))
        bag.state = BagState.RETIRADA_DONO
        monkeypatch.setattr(session._s.registry._archive, "persist", fail)
        session.feed(1, [])

    monkeypatch.setattr(serve, "processa", process)
    assert serve.main() == 1
    output = capsys.readouterr()
    assert "arquivo terminal" in output.err
    assert "secret-storage" not in output.err
    (folder,) = tmp_path.iterdir()
    records = list(iter_journal(folder))
    assert records[-1]["kind"] == "segment_interrupted"

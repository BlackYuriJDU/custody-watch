import json
import sqlite3
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from custody_watch.bag_registry import BagRegistry
from custody_watch.config import Config
from custody_watch.events import EventKind, EventLog
from custody_watch.ground_plane import GroundPlane
from custody_watch.orchestrator import LiveSession
from custody_watch.session_probe import SessionProbe, synthetic_frames
from custody_watch.terminal_bag_archive import TerminalBagArchive
from custody_watch.tracking import TrackedDetection
from custody_watch.types import Bag, BagState, Observation, Point


@pytest.mark.parametrize("state", [BagState.RETIRADA_DONO, BagState.RETIRADA_ESTRANHO])
def test_arquiva_somente_terminais_preservando_snapshot_posse_e_alias(tmp_path, state):
    archive = TerminalBagArchive(tmp_path / "terminal.sqlite")
    registry = BagRegistry(terminal_archive=archive)
    retired = registry.observe(Observation(-1, "suitcase", Point(1, 2), 3))
    retired.state = state
    retired.owner_party = 42
    retired.occlusion_candidates = {8, 7}
    registry.link_track(2**65, -1)
    current = registry.observe(Observation(2, "suitcase", Point(4, 5), 4))
    uncertain = registry.observe(Observation(3, "suitcase", Point(6, 7), 5))
    uncertain.state = BagState.AMBIGUA
    expected = asdict(retired)

    assert registry.archive_terminal_bags() == 1

    assert registry.resident() == [current, uncertain]
    assert registry.unresolved() == [current, uncertain]
    assert asdict(registry.get(-1)) == expected
    assert asdict(registry.get_by_track(2**65)) == expected
    assert registry.links() == {2**65: -1}
    assert registry.links(track_ids={999}) == {}
    assert registry.links(track_ids={2**65}) == {2**65: -1}
    assert registry.archived_counts() == {"bags": 1, "bag_links": 1}
    assert {bag.bag_id for bag in registry.all()} == {-1, 2, 3}
    assert registry.archive_terminal_bags() == 0


def test_id_terminal_reobservado_nao_renasce_nem_reabre_posse(tmp_path):
    registry = BagRegistry(terminal_archive=TerminalBagArchive(tmp_path / "terminal.sqlite"))
    bag = registry.observe(Observation(1, "suitcase", Point(0, 0), 0))
    bag.owner_party = 44
    bag.state = BagState.RETIRADA_ESTRANHO
    registry.link_track(11, 1)
    registry.archive_terminal_bags()
    events = EventLog()

    observed = registry.observe(Observation(11, "suitcase", Point(9, 9), 90), events=events)

    assert observed.state is BagState.RETIRADA_ESTRANHO
    assert observed.owner_party == 44
    assert observed.anchor == Point(0, 0)  # snapshot histórico congelado
    assert observed.last_seen == 0
    assert not events.of_kind(EventKind.BAG_APPEARED)
    assert registry.resident() == []
    with pytest.raises(ValueError, match="arquivada"):
        registry.assign_owner(1, 999)


def test_falha_de_gravacao_preserva_bagagem_e_alias_na_ram(tmp_path, monkeypatch):
    archive = TerminalBagArchive(tmp_path / "terminal.sqlite")
    registry = BagRegistry(terminal_archive=archive)
    bag = registry.observe(Observation(1, "suitcase", Point(0, 0), 0))
    bag.state = BagState.RETIRADA_DONO
    bag.owner_party = 44
    registry.link_track(11, 1)

    def failed_write(*args):
        raise OSError("falha de disco simulada")

    with monkeypatch.context() as patch:
        patch.setattr(archive, "persist", failed_write)
        with pytest.raises(OSError, match="falha de disco"):
            registry.archive_terminal_bags()
        assert registry.get(1) is bag
        assert registry.get_by_track(11) is bag
        assert registry.resident() == [bag]
        assert registry.archived_counts() == {"bags": 0, "bag_links": 0}

    assert registry.archive_terminal_bags() == 1
    assert registry.resident() == []


def test_conflito_de_alias_reverte_transacao_inteira(tmp_path):
    archive = TerminalBagArchive(tmp_path / "terminal.sqlite")
    first = Bag(1, Point(0, 0), state=BagState.RETIRADA_DONO)
    second = Bag(2, Point(1, 0), state=BagState.RETIRADA_ESTRANHO)
    archive.persist(first, {11: 1})

    with pytest.raises(sqlite3.IntegrityError):
        archive.persist(second, {11: 2})

    assert archive.get(2) is None
    assert archive.get(1) == first
    assert archive.links() == {11: 1}
    assert archive.counts() == {"bags": 1, "bag_links": 1}


@pytest.mark.parametrize("state", [BagState.NOVA, BagState.AMBIGUA, BagState.DESACOMPANHADA])
def test_arquivo_recusa_custodia_aberta(tmp_path, state):
    archive = TerminalBagArchive(tmp_path / "terminal.sqlite")
    bag = Bag(1, Point(0, 0), state=state)
    with pytest.raises(ValueError, match="terminal"):
        archive.persist(bag, {})
    assert archive.counts() == {"bags": 0, "bag_links": 0}


def test_arquivo_exclusivo_nao_reusa_namespace_ou_sobrescreve(tmp_path):
    path = tmp_path / "terminal.sqlite"
    archive = TerminalBagArchive(path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        TerminalBagArchive(path)
    assert path.read_bytes() == original
    BagRegistry(terminal_archive=archive)
    with pytest.raises(ValueError, match="registro"):
        BagRegistry(terminal_archive=archive)


def test_nome_com_caracteres_de_uri_e_id_canonico_grande(tmp_path):
    archive = TerminalBagArchive(tmp_path / "bag #1% ç.sqlite")
    bag = Bag(2**80, Point(0, 0), state=BagState.RETIRADA_DONO)
    archive.persist(bag, {-1: bag.bag_id})
    assert archive.get(bag.bag_id) == bag
    assert archive.canonical(-1) == bag.bag_id


def test_bagagem_nova_na_mesma_posicao_nao_e_engolida_pelo_historico(tmp_path):
    registry = BagRegistry(terminal_archive=TerminalBagArchive(tmp_path / "terminal.sqlite"))
    old = registry.observe(Observation(1, "suitcase", Point(0, 0), 0))
    old.state = BagState.RETIRADA_DONO
    old.occluded_since = 1
    registry.archive_terminal_bags()
    assert registry.occluded_near(Point(0, 0), 1) == []
    events = EventLog()
    new = registry.observe(Observation(2, "suitcase", Point(0, 0), 20), events=events)
    assert new.bag_id == 2
    assert new.state is BagState.NOVA
    assert len(events.of_kind(EventKind.BAG_APPEARED)) == 1
    assert registry.get(1).state is BagState.RETIRADA_DONO


@pytest.mark.parametrize("track_id", [1, 11])
def test_id_terminal_e_alias_nao_podem_ser_redirecionados_para_custodia_nova(tmp_path, track_id):
    registry = BagRegistry(terminal_archive=TerminalBagArchive(tmp_path / "terminal.sqlite"))
    old = registry.observe(Observation(1, "suitcase", Point(0, 0), 0))
    old.state = BagState.RETIRADA_ESTRANHO
    old.owner_party = 44
    registry.link_track(11, 1)
    registry.archive_terminal_bags()
    registry.observe(Observation(2, "suitcase", Point(5, 5), 5))
    with pytest.raises(ValueError, match="terminal"):
        registry.link_track(track_id, 2)
    assert registry.get_by_track(track_id).bag_id == 1
    assert registry.get_by_track(track_id).owner_party == 44


def test_falha_de_leitura_nao_converte_terminal_em_bagagem_nova(tmp_path):
    path = tmp_path / "terminal.sqlite"
    registry = BagRegistry(terminal_archive=TerminalBagArchive(path))
    bag = registry.observe(Observation(1, "suitcase", Point(0, 0), 0))
    bag.state = BagState.RETIRADA_DONO
    registry.archive_terminal_bags()
    path.rename(tmp_path / "preserved.sqlite")  # operações não deixam conexão aberta
    events = EventLog()

    with pytest.raises(sqlite3.OperationalError):
        registry.observe(Observation(1, "suitcase", Point(0, 0), 5), events=events)

    assert not path.exists()
    assert registry.resident() == []
    assert len(events) == 0


def test_arquivo_preenchido_nao_restaura_custodia_em_registro_novo(tmp_path):
    archive = TerminalBagArchive(tmp_path / "terminal.sqlite")
    archive.persist(Bag(1, Point(0, 0), state=BagState.RETIRADA_DONO), {})
    with pytest.raises(ValueError, match="registro"):
        BagRegistry(terminal_archive=archive)


def test_arquivo_trocado_nao_empresta_tombstones_de_outro_segmento(tmp_path):
    first = TerminalBagArchive(tmp_path / "first.sqlite")
    second = TerminalBagArchive(tmp_path / "second.sqlite")
    first.path.rename(tmp_path / "preserved.sqlite")
    second.path.rename(first.path)
    with pytest.raises(ValueError, match="namespace"):
        first.get(1)


def test_alias_arquivado_idempotente_e_snapshot_consultado_nao_altera_decisao(tmp_path):
    archive = TerminalBagArchive(tmp_path / "terminal.sqlite")
    registry = BagRegistry(terminal_archive=archive)
    bag = registry.observe(Observation(1, "suitcase", Point(0, 0), 0))
    bag.state = BagState.RETIRADA_ESTRANHO
    bag.owner_party = 44
    registry.archive_terminal_bags()
    registry.link_track(11, 1)
    registry.link_track(11, 1)
    detached = registry.get_by_track(11)
    detached.state = BagState.NOVA
    detached.owner_party = 999
    assert registry.get_by_track(11).state is BagState.RETIRADA_ESTRANHO
    assert registry.get(1).owner_party == 44
    assert registry.archived_counts() == {"bags": 1, "bag_links": 1}


@pytest.mark.parametrize("state", [BagState.RETIRADA_DONO, BagState.RETIRADA_ESTRANHO])
def test_reaparecimento_terminal_preserva_eventos_e_fila_da_sessao(tmp_path, state):
    plane = GroundPlane(np.diag([0.01, 0.01, 1.0]))
    baseline = LiveSession(plane, Config())
    archived = LiveSession(
        plane, Config(), terminal_archive=TerminalBagArchive(tmp_path / "terminal.sqlite")
    )
    for session in (baseline, archived):
        bag = session._s.registry.observe(Observation(1, "suitcase", Point(0, 0), 0))
        bag.owner_party = 44
        bag.state = state
        session._s.registry.link_track(11, 1)
        session.feed(1, [])
    for t in [2, 3, 10, 50]:
        tracked = [TrackedDetection(11, "suitcase", (400, 400, 440, 480))]
        baseline.feed(t, tracked)
        archived.feed(t, tracked)
    assert [e.to_dict() for e in archived.events] == [e.to_dict() for e in baseline.events]
    assert archived.queue() == baseline.queue()
    assert archived.bag_for_track(11).state is state
    assert archived.bag_for_track(11).owner_party == 44
    assert archived._s.registry.resident() == []


def test_terminal_orfa_reaparece_sem_tentar_atribuir_posse_posterior(tmp_path):
    plane = GroundPlane(np.diag([0.01, 0.01, 1.0]))
    session = LiveSession(
        plane, Config(), terminal_archive=TerminalBagArchive(tmp_path / "terminal.sqlite")
    )
    old = session._s.registry.observe(Observation(1, "suitcase", Point(0.2, 2), 0))
    old.state = BagState.RETIRADA_ESTRANHO
    session._s.registry.link_track(11, 1)
    session.feed(1, [])
    session.feed(
        2,
        [
            TrackedDetection(100, "person", (0, 0, 100, 200)),
            TrackedDetection(11, "suitcase", (0, 120, 40, 200)),
        ],
    )
    assert session.bag_for_track(11).owner_party is None
    assert session.bag_for_track(11).state is BagState.RETIRADA_ESTRANHO
    assert not session.events.of_kind(EventKind.BAG_OWNED)
    groups = len(session._s.parties._parties)
    with pytest.raises(ValueError, match="arquivada"):
        session.aponta_dono(1, 50, 100)
    assert len(session._s.parties._parties) == groups


def test_sessao_preserva_eventos_fila_posse_e_historico_sem_varrer_arquivo(tmp_path, monkeypatch):
    plane = GroundPlane(np.diag([0.01, 0.01, 1.0]))
    baseline = LiveSession(plane, Config())
    archive = TerminalBagArchive(tmp_path / "terminal.sqlite")
    measured = LiveSession(plane, Config(), terminal_archive=archive)
    for frame in synthetic_frames(1200, seed=7):
        if not frame.dropped:
            baseline.feed(frame.t, list(frame.tracked))
            measured.feed(frame.t, list(frame.tracked))

    assert [e.to_dict() for e in measured.events] == [e.to_dict() for e in baseline.events]
    assert measured.queue() == baseline.queue()
    assert sorted(
        (asdict(b) for b in measured._s.registry.all()), key=lambda b: b["bag_id"]
    ) == sorted((asdict(b) for b in baseline._s.registry.all()), key=lambda b: b["bag_id"])
    assert measured._s.registry.archived_counts()["bags"] > 0
    assert len(measured._s.registry.resident()) < len(baseline._s.registry.resident())

    def forbidden_scan():
        pytest.fail("feed, snapshot e probe não podem carregar o histórico inteiro")

    monkeypatch.setattr(archive, "all", forbidden_scan)
    monkeypatch.setattr(measured._s.registry, "all", forbidden_scan)
    measured.feed(240.0, [])
    assert measured.result().bag_links == {}
    with SessionProbe(measured) as probe:
        counts = probe.counts()
    assert counts["bags"] == len(measured._s.registry.resident())
    assert counts["terminal_bags"] == 0


def test_harness_separa_residentes_do_arquivo_e_preserva_journal(tmp_path):
    from custody_watch.journal_reader import iter_journal

    output = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts/soak_session.py"),
            "--output",
            str(output),
            "--frames",
            "800",
            "--seed",
            "7",
            "--archive-terminal-bags",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["terminal_archive"]["enabled"] is True
    assert summary["archived_terminal"]["bags"] > 0
    assert summary["last_snapshot"]["counts"]["terminal_bags"] == 0
    assert summary["last_snapshot"]["archived_terminal"] == summary["archived_terminal"]
    (journal_folder,) = (output / "journal").iterdir()
    records = list(iter_journal(journal_folder))
    assert len(records) == summary["journal_records"]
    assert summary["resources_ok"] is True
    assert summary["faults_ok"] is True

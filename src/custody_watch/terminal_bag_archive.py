"""Snapshots terminais locais; arquivo novo por registro, sem restaurar custódia."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .types import TERMINAL_BAG_STATES, Bag, BagState, Point


class TerminalBagArchive:
    """Arquivo opt-in; confirma bagagem e aliases antes de liberar os objetos.

    Conexões são fechadas em cada operação. Não mantém cache Python de registros;
    consultas completas são explícitas e não pertencem ao caminho por frame.
    O arquivo cresce no disco; não substitui o journal nem limita outras coleções.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).resolve()
        self._lock = threading.Lock()
        self._claimed = False
        self._namespace = uuid4().hex
        with self.path.open("xb"):
            pass
        with self._connection(initializing=True) as connection:
            connection.executescript(
                "CREATE TABLE bags (bag_id TEXT PRIMARY KEY, payload TEXT NOT NULL);"
                "CREATE TABLE aliases (track_id TEXT PRIMARY KEY, bag_id TEXT NOT NULL "
                "REFERENCES bags(bag_id));"
                "CREATE TABLE metadata (namespace TEXT NOT NULL);"
                "PRAGMA user_version=1;"
            )
            connection.execute("INSERT INTO metadata VALUES (?)", (self._namespace,))

    @contextmanager
    def _connection(self, *, initializing: bool = False) -> Iterator[sqlite3.Connection]:
        # mode=rw impede recriar silenciosamente um arquivo removido/trocado.
        with (
            self._lock,
            closing(sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True)) as connection,
        ):
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA cache_size=-256")
            connection.execute("PRAGMA synchronous=FULL")
            if not initializing:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                namespaces = connection.execute("SELECT namespace FROM metadata").fetchall()
                if version != 1 or namespaces != [(self._namespace,)]:
                    raise ValueError("namespace ou versão do arquivo terminal incompatível")
            with connection:
                yield connection

    def claim(self) -> None:
        """Proíbe compartilhar tombstones entre registros/segmentos novos."""
        with self._connection() as connection:
            count = connection.execute("SELECT COUNT(*) FROM bags").fetchone()[0]
            if self._claimed or count:
                raise ValueError("arquivo já vinculado a outro registro")
            self._claimed = True

    def persist(self, bag: Bag, aliases: dict[int, int]) -> None:
        if bag.state not in TERMINAL_BAG_STATES:
            raise ValueError("somente bagagem terminal pode ser arquivada")
        if any(value != bag.bag_id for value in aliases.values()):
            raise ValueError("alias aponta para outra bagagem")
        payload = asdict(bag)
        payload["state"] = bag.state.name
        payload["occlusion_candidates"] = sorted(bag.occlusion_candidates)
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self._connection() as connection:
            connection.execute("INSERT INTO bags VALUES (?, ?)", (str(bag.bag_id), text))
            connection.executemany(
                "INSERT INTO aliases VALUES (?, ?)",
                [(str(track), str(target)) for track, target in aliases.items()],
            )

    @staticmethod
    def _decode(text: str, bag_id: int) -> Bag:
        payload = json.loads(text)
        json.dumps(payload, allow_nan=False)
        payload["anchor"] = Point(**payload["anchor"])
        payload["state"] = BagState[payload["state"]]
        payload["occlusion_candidates"] = set(payload["occlusion_candidates"])
        bag = Bag(**payload)
        if bag.bag_id != bag_id or bag.state not in TERMINAL_BAG_STATES:
            raise ValueError("snapshot terminal inválido")
        return bag

    def get(self, bag_id: int) -> Bag | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM bags WHERE bag_id=?", (str(bag_id),)
            ).fetchone()
        return None if row is None else self._decode(row[0], bag_id)

    def canonical(self, track_id: int) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT bag_id FROM aliases WHERE track_id=?", (str(track_id),)
            ).fetchone()
        return track_id if row is None else int(row[0])

    def link(self, track_id: int, bag_id: int) -> None:
        with self._connection() as connection:
            previous = connection.execute(
                "SELECT bag_id FROM aliases WHERE track_id=?", (str(track_id),)
            ).fetchone()
            if previous is not None:
                if previous != (str(bag_id),):
                    raise ValueError("alias terminal não pode trocar de bagagem")
                return
            connection.execute("INSERT INTO aliases VALUES (?, ?)", (str(track_id), str(bag_id)))

    def links(self, track_ids: set[int] | None = None) -> dict[int, int]:
        with self._connection() as connection:
            if track_ids is None:
                rows = connection.execute("SELECT track_id, bag_id FROM aliases").fetchall()
            else:
                rows = []
                for track_id in track_ids:
                    row = connection.execute(
                        "SELECT track_id, bag_id FROM aliases WHERE track_id=?", (str(track_id),)
                    ).fetchone()
                    if row is not None:
                        rows.append(row)
        return {int(track): int(bag) for track, bag in rows}

    def all(self) -> list[Bag]:
        """Consulta histórica completa; pode materializar todo o arquivo em RAM."""
        with self._connection() as connection:
            rows = connection.execute("SELECT bag_id, payload FROM bags ORDER BY rowid").fetchall()
        return [self._decode(text, int(bag_id)) for bag_id, text in rows]

    def counts(self) -> dict[str, int]:
        with self._connection() as connection:
            bags = connection.execute("SELECT COUNT(*) FROM bags").fetchone()[0]
            links = connection.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
        return {"bags": bags, "bag_links": links}

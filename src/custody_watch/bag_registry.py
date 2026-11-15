"""Registro de bagagens por âncora espacial (spec §4.4).

Regra P2: bagagem parada não teleporta, então a posição é a identidade.
Aparência é inútil aqui — quarenta malas pretas idênticas derrotam qualquer
re-identificação, e é exatamente isso que faz um tracker trocar IDs sob
oclusão.

Regra P3: quando a reassociação é incerta, marcar AMBIGUA e calar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .config import RegistryConfig
from .events import Event, EventKind, EventLog
from .terminal_bag_archive import TerminalBagArchive
from .types import TERMINAL_BAG_STATES, Bag, BagState, Observation, Point

DEFAULT_REGISTRY = RegistryConfig()

MOVED_THRESHOLD_M = DEFAULT_REGISTRY.moved_threshold_m
AMBIGUITY_RADIUS_M = DEFAULT_REGISTRY.ambiguity_radius_m

_T = TypeVar("_T")


@dataclass(frozen=True)
class BagRetentionStatus:
    enabled: bool
    state: str
    resident_bags: int
    resident_links: int
    archived_bags: int
    archived_links: int


class BagRegistry:
    def __init__(
        self,
        config: RegistryConfig = DEFAULT_REGISTRY,
        *,
        terminal_archive: TerminalBagArchive | None = None,
    ) -> None:
        self._bags: dict[int, Bag] = {}
        self._by_track: dict[int, int] = {}
        self._unresolved: dict[int, Bag] = {}
        self._config = config
        self._archive = terminal_archive
        self._archive_failed = False
        self._archived_bags = 0
        self._archived_links = 0
        if self._archive is not None:
            self._archive.claim()

    def _archive_call(self, operation: Callable[..., _T], *args) -> _T:
        try:
            return operation(*args)
        except Exception:
            self._archive_failed = True
            raise

    @property
    def archive_failed(self) -> bool:
        return self._archive_failed

    def retention_status(self) -> BagRetentionStatus:
        """Contagens confirmadas nesta sessão; sem leitura de disco/histórico."""
        enabled = self._archive is not None
        state = "failed" if self._archive_failed else "ready" if enabled else "disabled"
        return BagRetentionStatus(
            enabled,
            state,
            len(self._bags),
            len(self._by_track),
            self._archived_bags,
            self._archived_links,
        )

    def get(self, bag_id: int) -> Bag | None:
        bag = self._bags.get(bag_id)
        if bag is not None or self._archive is None:
            return bag
        return self._archive_call(self._archive.get, bag_id)

    def get_by_track(self, track_id: int) -> Bag | None:
        """Bagagem que responde por este track.

        Depois de uma readoção, um mesmo `bag_id` responde por mais de um
        `track_id`. Enquanto não há religação os dois números são o mesmo, e é
        por isso que o default do mapa é o próprio `track_id` — nada precisa
        migrar.
        """
        bag_id = self._by_track.get(track_id)
        if bag_id is None:
            bag_id = (
                track_id
                if self._archive is None
                else self._archive_call(self._archive.canonical, track_id)
            )
        return self.get(bag_id)

    def link_track(self, track_id: int, bag_id: int) -> None:
        """Faz um track novo responder por uma bagagem já registrada."""
        if self._archive is not None:
            previous = self._archive_call(
                self._archive.get, self._archive_call(self._archive.canonical, track_id)
            )
            if previous is not None and previous.bag_id != bag_id:
                raise ValueError("ID terminal não pode trocar de bagagem")
        if bag_id not in self._bags:
            if self._archive is not None and self.get(bag_id) is not None:
                previous_links = self._archive_call(self._archive.links, {track_id})
                self._archive_call(self._archive.link, track_id, bag_id)
                self._archived_links += int(not previous_links)
                return
            raise KeyError(f"bagagem {bag_id} não registrada")
        self._by_track[track_id] = bag_id

    def links(self, *, track_ids: set[int] | None = None) -> dict[int, int]:
        """Mapa de track bruto para `bag_id` canônico, só onde houve
        religação. Espelha `TrackLinker.links()`: enquanto não há religação o
        track responde por si mesmo e não precisa aparecer aqui.
        """
        links = {} if self._archive is None else self._archive_call(self._archive.links, track_ids)
        links.update(
            {
                track: bag
                for track, bag in self._by_track.items()
                if track_ids is None or track in track_ids
            }
        )
        return links

    def occluded_near(self, position: Point, radius: float) -> list[Bag]:
        """Bagagens invisíveis cuja âncora está dentro do raio.

        Bagagem já resolvida não entra: uma âncora cuja custódia foi decidida
        deixa de ser alvo de readoção, senão uma bagagem genuinamente nova
        deixada no mesmo lugar é engolida pela antiga — sem `BAG_APPEARED`,
        sem posse, e invisível para sempre, porque a hospedeira é terminal.
        """
        return [
            bag
            for bag in self._bags.values()
            if bag.occluded_since is not None
            and bag.state not in TERMINAL_BAG_STATES
            and bag.state is not BagState.AMBIGUA
            and bag.anchor.distance_to(position) <= radius
        ]

    def all(self) -> list[Bag]:
        history = [] if self._archive is None else self._archive_call(self._archive.all)
        return [*self.resident(), *history]

    def resident(self) -> list[Bag]:
        """Somente objetos residentes; o histórico em disco exige all/get."""
        return list(self._bags.values())

    def archived_counts(self) -> dict[str, int]:
        return (
            {"bags": 0, "bag_links": 0}
            if self._archive is None
            else self._archive_call(self._archive.counts)
        )

    def is_archived(self, bag_id: int) -> bool:
        """Um snapshot histórico é somente leitura, inclusive ao reaparecer."""
        return (
            self._archive is not None and bag_id not in self._bags and self.get(bag_id) is not None
        )

    def _archive_terminal(self, bag: Bag) -> None:
        if self._archive is None:
            return
        aliases = {
            track: target for track, target in self._by_track.items() if target == bag.bag_id
        }
        self._archive_call(self._archive.persist, bag, aliases)
        self._archived_bags += 1
        self._archived_links += len(aliases)
        # A transação já foi confirmada. Em erro, todas as referências ficam em RAM.
        self._bags.pop(bag.bag_id)
        for track in aliases:
            self._by_track.pop(track)

    def archive_terminal_bags(self) -> int:
        """Arquiva somente custódias encerradas; sem arquivo é uma operação nula."""
        if self._archive is None:
            return 0
        count = 0
        for bag_id, bag in list(self._unresolved.items()):
            if bag.state in TERMINAL_BAG_STATES:
                self._archive_terminal(bag)
                self._unresolved.pop(bag_id)
                count += 1
        return count

    def unresolved(self) -> list[Bag]:
        """Índice de custódias não terminais, incluindo as ambíguas.

        Custódia terminal é irreversível no domínio. Como a resolução escreve
        em Bag.state, retirar do índice na próxima consulta observa também
        transições feitas por custody.py. Não varre tombstones de consultas
        anteriores. Com arquivo opt-in, confirma snapshot/aliases antes de
        retirar seus objetos da RAM; sem arquivo, conserva o registro completo.
        Posse e histórico permanecem consultáveis nos dois modos.
        Ambiguidade permanece: ela suprime sinais, mas não encerra custódia.
        """
        current = []
        for bag_id, bag in list(self._unresolved.items()):
            if bag.state in TERMINAL_BAG_STATES:
                self._archive_terminal(bag)
                self._unresolved.pop(bag_id, None)
            else:
                current.append(bag)
        return current

    def observe(
        self,
        observation: Observation,
        events: EventLog | None = None,
        evidence: dict | None = None,
    ) -> Bag:
        """Registra ou atualiza uma bagagem.

        Movimento sub-limiar não move a âncora: jitter de detector não deve ser
        lido como a bagagem sendo carregada.

        Emite `BAG_APPEARED` apenas para bagagem nova — uma já conhecida não
        emite, senão cada frame geraria um evento. `t` vem de `observation.t`.
        Quando `events` é `None`, nada é emitido.
        """
        bag = self.get_by_track(observation.track_id)
        if bag is None:
            bag = Bag(
                bag_id=observation.track_id,
                anchor=observation.position,
                last_seen=observation.t,
                anchored_at=observation.t,
            )
            self._bags[bag.bag_id] = bag
            self._unresolved[bag.bag_id] = bag

            if events is not None:
                events.emit(
                    Event(
                        kind=EventKind.BAG_APPEARED,
                        t_start=observation.t,
                        t_end=observation.t,
                        subject=None,
                        bag=bag.bag_id,
                        party=bag.owner_party,
                        evidence={"anchor": [bag.anchor.x, bag.anchor.y], **(evidence or {})},
                    )
                )
            return bag

        if self._archive is not None and bag.bag_id not in self._bags:
            return bag  # snapshot terminal congelado; não reabre custódia
        if bag.anchor.distance_to(observation.position) >= self._config.moved_threshold_m:
            bag.anchor = observation.position
        bag.last_seen = observation.t
        return bag

    def has_moved(self, observation: Observation) -> bool:
        bag = self.get_by_track(observation.track_id)
        if bag is None:
            return False
        return bag.anchor.distance_to(observation.position) >= self._config.moved_threshold_m

    def assign_owner(
        self,
        bag_id: int,
        party_id: int,
        t: float = 0.0,
        events: EventLog | None = None,
        evidence: dict | None = None,
    ) -> None:
        """Vinculação por back-tracing: quem depositou a bagagem.

        `t` é o instante do frame; o default existe só para não quebrar
        chamadas antigas. Quando `events` é `None`, nada é emitido.
        """
        if bag_id not in self._bags and self._archive is not None and self.get(bag_id) is not None:
            raise ValueError("custódia arquivada não pode receber outro dono")
        self._bags[bag_id].owner_party = party_id

        if events is not None:
            events.emit(
                Event(
                    kind=EventKind.BAG_OWNED,
                    t_start=t,
                    t_end=t,
                    subject=None,
                    bag=bag_id,
                    party=party_id,
                    evidence={"party": party_id, **(evidence or {})},
                )
            )

    def mark_ambiguous_neighbours(
        self,
        bag_id: int,
        t: float = 0.0,
        events: EventLog | None = None,
    ) -> list[int]:
        """Marca a bagagem e todas as vizinhas dentro do raio como AMBIGUA.

        Chamado quando uma oclusão se resolve com baixa confiança. Marcar todas
        as candidatas é deliberado: não há como saber qual é qual, e chutar
        corromperia o mapa de posse.

        Bagagem terminal não é rebaixada. Custódia já decidida -- inclusive um
        furto já reportado ao operador -- não pode ser apagada do registro por
        uma ambiguidade posterior nas vizinhanças: `removal_outcomes()` conta
        eventos e não mudaria, mas o registro deixaria de ser o registro do
        que aconteceu, e quem lesse `bag.state` leria a coisa errada.
        `occluded_near` já pula terminal pela mesma razão.

        Emite um único `BAG_AMBIGUOUS`, com o alvo em `bag` e a lista completa
        de afetados na evidência — não um evento por bagagem marcada. `t` é o
        instante do frame; o default existe só para não quebrar chamadas
        antigas. Quando `events` é `None`, nada é emitido.
        """
        target = self._bags[bag_id]
        affected: list[int] = []
        for other in self._bags.values():
            if other.state in TERMINAL_BAG_STATES:
                continue
            if target.anchor.distance_to(other.anchor) <= self._config.ambiguity_radius_m:
                other.state = BagState.AMBIGUA
                affected.append(other.bag_id)

        if events is not None:
            events.emit(
                Event(
                    kind=EventKind.BAG_AMBIGUOUS,
                    t_start=t,
                    t_end=t,
                    subject=None,
                    bag=bag_id,
                    party=target.owner_party,
                    evidence={
                        "neighbour_bags": affected,
                        "candidate_people": [],
                        "radius_m": self._config.ambiguity_radius_m,
                    },
                )
            )
        return affected

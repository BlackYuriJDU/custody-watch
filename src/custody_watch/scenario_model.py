"""O que é um cenário de custódia e os prazos que ele deriva da config.

O catálogo de cenas está em `scenarios.py` e nos módulos `scenarios_*`; as
premissas reutilizáveis, em `scenario_premises.py`. Aqui fica só a forma comum
a todas as cenas, para que o catálogo leia como roteiro e não como geometria.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .ground_plane import GroundPlane
from .orchestrator import MIN_MISSING_FRAMES
from .scenario_motion import Actor, Route, Waypoint, last_before, render
from .scenario_scoring import Category, DueSignal, Expectation
from .tracking import TrackedDetection, anchor_vetoes
from .types import Point

MIN_FPS = 3.0
"""Menor taxa de quadros para a qual as premissas temporais valem. Prazos que
dependem da cadência (fusão de grupos a cada `merge_every_frames` quadros,
o piso de quadros e o quadro em que a oclusão vence) são dimensionados para
ela, o pior caso."""
WALK_SPEED_MS = 1.2
"""Passo de pedestre; perto da mediana medida no CAVIAR (1,34 m/s, `tracking.py`)."""
QUEUE_SPEED_MS = 0.5
"""Fila andando: mais devagar que o passo, o que alonga o deslocamento que
`carry_confirm_s` precisa ver."""
ENTRY_DISTANCE_M = 8.0
"""De onde os atores entram: longe o bastante para não estar ao alcance de nada."""
EXIT_DISTANCE_M = 12.0
CARRY_OFFSET_M = 0.6
"""Distância entre quem leva e a bagagem levada; dentro do raio de contato."""
SLACK_S = 2.0
"""Folga de cena: pausa entre ações, para que uma não se sobreponha à outra."""
EVENT_SLACK_S = 1.0
"""Folga do gabarito sobre o prazo que a config impõe a cada sinal."""
DISTANCE_MARGIN_M = 2.0
"""Quanto além de um raio alguém vai para estar inequivocamente fora dele."""
BAG_BOX_EXTRA_PX = 0.5
"""A caixa de bagagem tem o piso de resolução da config mais isto: o piso
mede a confiabilidade do detector, que estes cenários não têm."""
PREMISE_STEP_S = 0.01
VETO_CHECK_FPS = 10.0
FOREVER = math.inf
"""Fim de visibilidade de quem fica em cena até o último quadro."""
OUTSIDE_AFTER_CROSSING_S = 2.0 * PREMISE_STEP_S
"""O cruzamento é limite inferior com erro de um passo; dois passos depois,
quem saiu do raio já está fora dele."""

INVARIANTS = ("P1", "P2", "P3", "P4", "base")
KINDS = ("legitimo", "furto", "ambiguo")
_SLUG = re.compile(r"^[a-z0-9_]+$")


def identity_plane() -> GroundPlane:
    """O plano dos cenários: pixel é metro, e o pé da caixa é a posição."""
    return GroundPlane(np.eye(3))


@dataclass(frozen=True)
class Premissa:
    descricao: str
    valida: bool
    medida: str


PremiseCheck = Callable[["Scenario", Config], tuple[Premissa, ...]]


@dataclass(frozen=True)
class Scenario:
    """Uma cena com gabarito. Construída para uma `Config`; ver `scenarios.py`."""

    name: str
    descricao: str
    invariant: str
    kind: str
    roles: tuple[tuple[str, int], ...]
    expected: Expectation
    actors: tuple[Actor, ...]
    duration_s: float
    bag_box_height: float
    marks: tuple[tuple[str, float], ...]
    checks: PremiseCheck = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not _SLUG.match(self.name):
            raise ValueError(f"nome deve ser slug em inglês, recebido {self.name!r}")
        if self.invariant not in INVARIANTS or self.kind not in KINDS:
            raise ValueError(f"{self.name}: invariante/tipo inválidos")
        ids = [track for _, track in self.roles]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.name}: dois papéis com o mesmo track")
        targets = {d.target for d in self.expected.due} | self.expected.never_accuse
        if not targets <= set(ids):
            raise ValueError(f"{self.name}: gabarito cita tracks sem papel {targets - set(ids)}")

    @property
    def papeis(self) -> dict[str, int]:
        return dict(self.roles)

    def role(self, name: str) -> int:
        return self.papeis[name]

    def mark(self, name: str) -> float:
        return dict(self.marks)[name]

    def actor(self, track_id: int) -> Actor:
        return next(a for a in self.actors if a.track_id == track_id)

    def frames(self, fps: float) -> Iterator[tuple[float, list[TrackedDetection]]]:
        if fps < MIN_FPS:
            raise ValueError(f"fps {fps} abaixo de {MIN_FPS}: as premissas temporais não valem")
        return render(self.actors, self.duration_s, fps, self.bag_box_height)

    def premissas(self, config: Config) -> tuple[Premissa, ...]:
        return (no_vetoes(self, config), tail(self, config), *self.checks(self, config))


# --- prazos derivados da config ------------------------------------------------


def occlusion_declared_s(config: Config) -> float:
    """Pior atraso, a MIN_FPS, entre a última observação e a oclusão aberta.

    As duas réguas do orquestrador: o limiar em segundos decide até um quadro
    depois de vencer, e nunca antes de `MIN_MISSING_FRAMES` quadros sem a
    bagagem. A 3 fps manda o piso: dois quadros, 0,67 s.
    """
    pelo_tempo = config.pipeline.missing_before_occluded_s + 1.0 / MIN_FPS
    return max(pelo_tempo, MIN_MISSING_FRAMES / MIN_FPS)


@dataclass(frozen=True)
class Timing:
    config: Config

    @property
    def settle_s(self) -> float:
        """Tempo parada para virar âncora, no pior quadro, com folga."""
        return self.config.custody.rest_confirm_s + 1.0 / MIN_FPS + SLACK_S

    @property
    def merge_period_s(self) -> float:
        return self.config.pipeline.merge_every_frames / MIN_FPS

    @property
    def weak_bond_wait_s(self) -> float:
        """Uma fusão para começar a contar, outra para concluir."""
        return self.config.party.weak_bond_s + 2.0 * self.merge_period_s + SLACK_S

    @property
    def strong_walk_s(self) -> float:
        return self.config.party.late_join_extent_m / WALK_SPEED_MS + self.merge_period_s + SLACK_S

    @property
    def occlusion_s(self) -> float:
        """Ausência declarada e registrada: o pior atraso de declaração a
        MIN_FPS, um quadro de folga e o piso do log."""
        declarada = occlusion_declared_s(self.config) + 1.0 / MIN_FPS
        return declarada + self.config.custody.min_occlusion_report_s

    @property
    def tail_s(self) -> float:
        """Toda decisão adiada (oclusão) se resolve dentro da cena."""
        return self.config.custody.max_occlusion_s + self.occlusion_s + SLACK_S

    @property
    def unattended_tolerance_s(self) -> float:
        return self.config.custody.unattended_time_s + 2.0 / MIN_FPS + EVENT_SLACK_S

    @property
    def contact_tolerance_s(self) -> float:
        return 2.0 / MIN_FPS + EVENT_SLACK_S

    def accusation_tolerance_s(self, speed_ms: float) -> float:
        reach = self.config.registry.moved_threshold_m / speed_ms
        return reach + self.config.custody.carry_confirm_s + 2.0 / MIN_FPS + EVENT_SLACK_S

    @property
    def bag_box_height(self) -> float:
        return self.config.pipeline.min_bag_height_px + BAG_BOX_EXTRA_PX


# --- montagem ------------------------------------------------------------------


def person(track_id: int, route: Route, start: float, end: float) -> Actor:
    return Actor(track_id, "person", route.waypoints(), ((start, end),))


def suitcase(
    track_id: int, path: list[Waypoint], visible: tuple[tuple[float, float], ...]
) -> Actor:
    return Actor(track_id, "suitcase", tuple(path), visible)


def arrive(t0: float, anchor: Point, offset: tuple[float, float], origin: Point) -> Route:
    """Quem chega trazendo a bagagem a `offset` de si e a deixa na âncora."""
    route = Route(t0, origin.x, origin.y)
    return route.walk_to(anchor.x - offset[0], anchor.y - offset[1], WALK_SPEED_MS)


def origin_behind(offset: tuple[float, float], anchor: Point) -> Point:
    """De onde vem quem traz a bagagem a `offset`: ENTRY_DISTANCE_M atrás."""
    norm = math.hypot(*offset)
    ux, uy = offset[0] / norm, offset[1] / norm
    return Point(
        anchor.x - offset[0] - ux * ENTRY_DISTANCE_M, anchor.y - offset[1] - uy * ENTRY_DISTANCE_M
    )


def crossing(actor: Actor, anchor: Point, radius: float, t0: float, t1: float) -> float:
    """Limite inferior do instante em que `actor` passa a estar a > `radius`."""
    found = last_before(
        lambda t: actor.position(t).distance_to(anchor) > radius, t0, t1, PREMISE_STEP_S
    )
    if found is None:
        raise ValueError(f"track {actor.track_id} nunca sai do raio {radius} m")
    return found


def entering(actor: Actor, anchor: Point, radius: float, t0: float, t1: float) -> float:
    """Limite inferior do instante em que `actor` entra no raio."""
    found = last_before(
        lambda t: actor.position(t).distance_to(anchor) <= radius, t0, t1, PREMISE_STEP_S
    )
    if found is None:
        raise ValueError(f"track {actor.track_id} nunca entra no raio {radius} m")
    return found


def unattended_due(bag_id: int, t_cross: float, tm: Timing) -> DueSignal:
    return DueSignal(
        Category.UNATTENDED,
        bag_id,
        t_cross,
        tm.unattended_tolerance_s,
        "bagagem desacompanhada depois que o dono cruza unattended_distance_m",
    )


def accusation_due(track: int, t_take: float, tm: Timing, speed: float) -> DueSignal:
    return DueSignal(
        Category.ACCUSATION,
        track,
        t_take,
        tm.accusation_tolerance_s(speed),
        "retirada por quem não é do grupo dono",
    )


# --- premissas comuns a toda cena ---------------------------------------------


def no_vetoes(scn: Scenario, config: Config) -> Premissa:
    floor = config.pipeline.min_bag_height_px
    vetados: set[str] = set()
    for _, tracked in render(scn.actors, scn.duration_s, VETO_CHECK_FPS, scn.bag_box_height):
        vetados.update(anchor_vetoes(tracked, floor).values())
    return Premissa(
        "nenhum veto de imagem (cortada, pequena, carregada) dispara: a cena não testa percepção",
        not vetados,
        f"vetos: {sorted(vetados) or 'nenhum'}",
    )


def tail(scn: Scenario, config: Config) -> Premissa:
    need = config.custody.max_occlusion_s + occlusion_declared_s(config)
    have = scn.duration_s - scn.mark("ultima_acao")
    return Premissa(
        "a cena roda ≥ max_occlusion_s após a última ação: nenhuma decisão fica adiada",
        have >= need,
        f"{have:.1f} s (mínimo {need:.1f} s)",
    )

"""Movimento e visibilidade dos atores de um cenário sintético, em metros.

Um ator é um track: pessoa ou bagagem, com uma trajetória linear por partes
e os intervalos em que o detector o vê. Nada aqui sabe o que é furto ou posse;
é só geometria e tempo, renderizados no mesmo formato que o rastreador
entrega ao `run_session`.

As caixas são desenhadas para o **plano identidade**: o pé da caixa é a
posição no chão em metros. Altura e largura da caixa não entram na lógica de
custódia, só nos vetos de imagem (`tracking.anchor_vetoes`), que são
propriedade da percepção e ficam fora do que estes cenários medem -- por isso
a altura da caixa de bagagem é recebida de quem constrói a cena, derivada do
piso de resolução da config.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from .tracking import TrackedDetection
from .types import PERSON_CLASS, Point

Waypoint = tuple[float, float, float]
"""(t, x, y): instante em segundos e posição no chão em metros."""

PERSON_HALF_WIDTH_M = 0.25
PERSON_HEIGHT_M = 1.7
BAG_HALF_WIDTH_M = 0.2


@dataclass(frozen=True)
class Actor:
    """Um track do cenário: trajetória e intervalos de visibilidade.

    `visible` é uma sequência de intervalos semiabertos `[início, fim)`. Fora
    deles o detector não entrega nada -- oclusão, lacuna ou saída de cena.
    """

    track_id: int
    cls: str
    path: tuple[Waypoint, ...]
    visible: tuple[tuple[float, float], ...]
    _times: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError(f"track {self.track_id}: trajetória vazia")
        times = tuple(w[0] for w in self.path)
        if any(b < a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError(f"track {self.track_id}: instantes da trajetória fora de ordem")
        object.__setattr__(self, "_times", times)

    @property
    def is_person(self) -> bool:
        return self.cls == PERSON_CLASS

    def position(self, t: float) -> Point:
        """Posição interpolada; antes do início e depois do fim, fica parada."""
        i = bisect_right(self._times, t)
        if i == 0:
            return Point(self.path[0][1], self.path[0][2])
        if i == len(self.path):
            return Point(self.path[-1][1], self.path[-1][2])
        t0, x0, y0 = self.path[i - 1]
        t1, x1, y1 = self.path[i]
        if t1 == t0:
            return Point(x1, y1)
        f = (t - t0) / (t1 - t0)
        return Point(x0 + f * (x1 - x0), y0 + f * (y1 - y0))

    def is_visible(self, t: float) -> bool:
        return any(start <= t < end for start, end in self.visible)

    def first_visible_t(self) -> float:
        return min(start for start, _ in self.visible)


class Route:
    """Construtor de trajetória: anda, espera, anda. Usado só na montagem."""

    def __init__(self, t: float, x: float, y: float) -> None:
        self._points: list[Waypoint] = [(t, x, y)]

    @property
    def t(self) -> float:
        return self._points[-1][0]

    @property
    def here(self) -> Point:
        _, x, y = self._points[-1]
        return Point(x, y)

    def walk_to(self, x: float, y: float, speed_ms: float) -> Route:
        if speed_ms <= 0.0:
            raise ValueError(f"velocidade deve ser positiva, recebida {speed_ms}")
        distance = self.here.distance_to(Point(x, y))
        self._points.append((self.t + distance / speed_ms, x, y))
        return self

    def stay(self, seconds: float) -> Route:
        if seconds < 0.0:
            raise ValueError(f"espera negativa: {seconds}")
        here = self.here
        self._points.append((self.t + seconds, here.x, here.y))
        return self

    def stay_until(self, t: float) -> Route:
        return self.stay(max(0.0, t - self.t))

    def waypoints(self) -> tuple[Waypoint, ...]:
        return tuple(self._points)


def carried(
    carrier: Sequence[Waypoint], t0: float, t1: float, offset: tuple[float, float]
) -> list[Waypoint]:
    """Trajetória de bagagem levada por alguém, com deslocamento fixo dele.

    Como a do portador é linear por partes e o deslocamento é constante, a da
    bagagem também é, com as mesmas quebras: basta copiar os pontos de quebra
    dentro de `[t0, t1]` e as duas pontas interpoladas.
    """
    guide = Actor(track_id=-1, cls=PERSON_CLASS, path=tuple(carrier), visible=((t0, t1),))
    times = [t0, *(w[0] for w in carrier if t0 < w[0] < t1), t1]
    dx, dy = offset
    out: list[Waypoint] = []
    for t in times:
        p = guide.position(t)
        out.append((t, p.x + dx, p.y + dy))
    return out


def detection(actor: Actor, t: float, bag_box_height: float) -> TrackedDetection:
    """A caixa que o rastreador entregaria, com o pé na posição do ator."""
    p = actor.position(t)
    if actor.is_person:
        bbox = (p.x - PERSON_HALF_WIDTH_M, p.y - PERSON_HEIGHT_M, p.x + PERSON_HALF_WIDTH_M, p.y)
    else:
        bbox = (p.x - BAG_HALF_WIDTH_M, p.y - bag_box_height, p.x + BAG_HALF_WIDTH_M, p.y)
    return TrackedDetection(track_id=actor.track_id, cls=actor.cls, bbox=bbox)


def render(
    actors: Sequence[Actor], duration_s: float, fps: float, bag_box_height: float
) -> Iterator[tuple[float, list[TrackedDetection]]]:
    """Quadros a `fps`, com `t = k / fps`: estritamente crescente e reprodutível.

    O instante vem do índice, nunca de soma acumulada, para que a mesma cena
    na mesma taxa produza exatamente os mesmos números.
    """
    if fps <= 0.0:
        raise ValueError(f"fps deve ser positivo, recebido {fps}")
    total = int(duration_s * fps) + 1
    for k in range(total):
        t = k / fps
        yield t, [detection(a, t, bag_box_height) for a in actors if a.is_visible(t)]


def sample_times(t0: float, t1: float, step: float) -> Iterator[float]:
    """Instantes `t0, t0+step, ...` até `t1` inclusive, sem soma acumulada."""
    n = int((t1 - t0) / step)
    for k in range(n + 1):
        yield t0 + k * step
    if t0 + n * step < t1:
        yield t1


def last_before(
    condition: Callable[[float], bool], t0: float, t1: float, step: float
) -> float | None:
    """Último instante amostrado antes de `condition` passar a valer.

    É um limite **inferior** do instante verdadeiro, com erro de até `step`.
    Gabarito usa limite inferior: um sinal emitido no quadro do cruzamento
    nunca cai antes da referência por erro de amostragem.
    """
    previous = t0
    for t in sample_times(t0, t1, step):
        if condition(t):
            return previous
        previous = t
    return None


__all__ = [
    "Actor",
    "Route",
    "Waypoint",
    "carried",
    "detection",
    "last_before",
    "render",
    "sample_times",
]

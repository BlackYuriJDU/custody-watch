"""Baseline ingênua de custódia: a referência mínima contra a qual P1–P3 se medem.

É a regra de distância/tempo da família PETS -- dono é quem estava junto
quando a bagagem parou; bagagem desacompanhada quando o dono fica longe por
tempo demais; alarme quando a bagagem sai do lugar com outra pessoa --
implementada aqui como referência mínima. **Não é reprodução de nenhum paper
específico**, nem dos sistemas que concorreram no PETS: é o que sobra dessa
família de regras quando se tira tudo que este projeto acrescentou.

O que ela deliberadamente não tem:

- **grupo**: dono é um track só. Acompanhante que leva a mala é estranho;
- **estado de incerteza**: o primeiro quadro em que a bagagem aparece longe da
  âncora decide, sem esperar confirmação e sem AMBIGUA;
- **re-ID** nem readoção: track novo é bagagem nova, e dono novo;
- **sinal de pessoa** (N1/N2): só acusa e só marca desacompanhada.

Usa os mesmos limiares da `Config` (raios, prazos) que o sistema recebe, para
que a diferença medida venha das regras e não de parâmetros diferentes.
Consome a mesma entrada -- quadros de `TrackedDetection` e um `GroundPlane` --
e devolve `Emission`, o formato comparável de `scenario_scoring`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .config import Config
from .ground_plane import GroundPlane
from .scenario_scoring import Category, Emission
from .tracking import TrackedDetection, to_observations
from .types import BAG_CLASSES, Observation, Point

SOURCE = "baseline"


@dataclass
class _NaiveBag:
    """Estado de um track de bagagem na baseline. Vive só durante a passagem."""

    candidate: Point
    candidate_since: float
    anchor: Point | None = None
    owner: int | None = None
    away_since: float | None = None
    unattended_reported: bool = False
    resolved: bool = False


def _nearest_within(
    people: Sequence[Observation], target: Point, radius: float
) -> Observation | None:
    perto = [p for p in people if p.position.distance_to(target) <= radius]
    return min(perto, key=lambda p: p.position.distance_to(target), default=None)


class NaiveBaseline:
    """Uma passagem da baseline sobre uma sequência de quadros."""

    def __init__(self, config: Config) -> None:
        self._rest_s = config.custody.rest_confirm_s
        self._moved_m = config.registry.moved_threshold_m
        self._owner_radius_m = config.pipeline.owner_search_radius_m
        self._unattended_m = config.custody.unattended_distance_m
        self._unattended_s = config.custody.unattended_time_s
        self._bags: dict[int, _NaiveBag] = {}
        self._emissions: list[Emission] = []

    @property
    def emissions(self) -> tuple[Emission, ...]:
        return tuple(self._emissions)

    def feed(self, t: float, observations: Sequence[Observation]) -> None:
        people = [o for o in observations if o.cls not in BAG_CLASSES]
        for bag in (o for o in observations if o.cls in BAG_CLASSES):
            self._observe(t, bag, people)

    def _observe(self, t: float, bag: Observation, people: Sequence[Observation]) -> None:
        state = self._bags.setdefault(
            bag.track_id, _NaiveBag(candidate=bag.position, candidate_since=t)
        )
        if state.resolved:
            return
        if state.anchor is None:
            self._settle(t, bag, state, people)
            return
        if bag.position.distance_to(state.anchor) >= self._moved_m:
            self._moved(t, bag, state, people)
            return
        self._attendance(t, bag, state, people)

    def _settle(
        self, t: float, bag: Observation, state: _NaiveBag, people: Sequence[Observation]
    ) -> None:
        """Parou? Então o dono é quem estiver mais perto agora."""
        if bag.position.distance_to(state.candidate) >= self._moved_m:
            state.candidate, state.candidate_since = bag.position, t
            return
        if t - state.candidate_since < self._rest_s:
            return
        # Onde ela está agora, e não onde a janela começou: o começo da janela
        # pode ser o último passo de quem a trazia, até meio metro antes.
        state.anchor = bag.position
        owner = _nearest_within(people, state.anchor, self._owner_radius_m)
        state.owner = None if owner is None else owner.track_id

    def _moved(
        self, t: float, bag: Observation, state: _NaiveBag, people: Sequence[Observation]
    ) -> None:
        """Saiu da âncora: com o dono é trânsito; com qualquer outro, alarme."""
        carrier = _nearest_within(people, bag.position, self._owner_radius_m)
        if carrier is None:
            return
        if carrier.track_id == state.owner:
            self._bags[bag.track_id] = _NaiveBag(candidate=bag.position, candidate_since=t)
            return
        state.resolved = True
        self._emissions.append(
            Emission(Category.ACCUSATION, t, carrier.track_id, bag.track_id, SOURCE)
        )

    def _attendance(
        self, t: float, bag: Observation, state: _NaiveBag, people: Sequence[Observation]
    ) -> None:
        """Regra de distância/tempo: dono longe (ou fora de quadro) por tempo demais."""
        if state.owner is None or state.anchor is None:
            return
        owner = next((p for p in people if p.track_id == state.owner), None)
        if owner is not None and owner.position.distance_to(state.anchor) <= self._unattended_m:
            state.away_since = None
            state.unattended_reported = False
            return
        if state.away_since is None:
            state.away_since = t
            return
        if not state.unattended_reported and t - state.away_since >= self._unattended_s:
            state.unattended_reported = True
            self._emissions.append(Emission(Category.UNATTENDED, t, None, bag.track_id, SOURCE))


def run_baseline(
    frames: Iterable[tuple[float, list[TrackedDetection]]],
    plane: GroundPlane,
    config: Config,
) -> tuple[Emission, ...]:
    """Roda a baseline sobre os mesmos quadros que `run_session` consome."""
    baseline = NaiveBaseline(config)
    for t, tracked in frames:
        baseline.feed(t, to_observations(tracked, plane, t))
    return baseline.emissions


__all__ = ["NaiveBaseline", "run_baseline"]

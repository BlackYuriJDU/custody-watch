"""Premissas reutilizáveis: o que cada cena mede na própria trajetória.

Cada função devolve `Premissa`s com a medida que decidiu, para que uma
premissa falha diga quanto faltou e não só que faltou.
"""

from __future__ import annotations

from collections.abc import Iterator

from .config import Config
from .scenario_model import MIN_FPS, PREMISE_STEP_S, Premissa, Scenario, merge_interval_s
from .scenario_motion import Actor, sample_times
from .types import Point


def _samples(t0: float, t1: float) -> Iterator[float]:
    return sample_times(t0, t1, PREMISE_STEP_S)


def depositor(scn: Scenario, bag_id: int, owner_id: int, config: Config) -> Premissa:
    bag = scn.actor(bag_id)
    t = bag.first_visible_t()
    where = bag.position(t)
    people = [a for a in scn.actors if a.is_person and a.is_visible(t)]
    nearest = min(people, key=lambda p, t=t, where=where: p.position(t).distance_to(where))
    d = nearest.position(t).distance_to(where)
    limit = config.pipeline.owner_search_radius_m
    return Premissa(
        f"o dono {owner_id} é o mais próximo da bagagem {bag_id} quando ela aparece, "
        "a ≤ owner_search_radius_m",
        nearest.track_id == owner_id and d <= limit,
        f"mais próximo {nearest.track_id} a {d:.2f} m (limite {limit:.2f} m)",
    )


def rests(scn: Scenario, bag_id: int, t0: float, t1: float, config: Config) -> Premissa:
    bag = scn.actor(bag_id)
    start = bag.position(t0)
    drift = max(bag.position(t).distance_to(start) for t in _samples(t0, t1))
    need = config.custody.rest_confirm_s + 1.0 / MIN_FPS
    return Premissa(
        f"a bagagem {bag_id} fica parada ≥ rest_confirm_s antes da ação seguinte",
        t1 - t0 >= need and drift < config.registry.moved_threshold_m,
        f"{t1 - t0:.1f} s parada (mínimo {need:.2f} s), deriva {drift:.3f} m",
    )


def within_stats(
    a: Actor, b: Actor, radius: float, t0: float, t1: float
) -> tuple[float, float, float]:
    """(maior trecho contínuo, total, distância mínima) com `a` visível."""
    longest = total = run = 0.0
    nearest = float("inf")
    for t in _samples(t0, t1):
        if not a.is_visible(t):
            run = 0.0
            continue
        d = a.position(t).distance_to(b.position(t))
        nearest = min(nearest, d)
        if d <= radius:
            run += PREMISE_STEP_S
            total += PREMISE_STEP_S
            longest = max(longest, run)
        else:
            run = 0.0
    return longest, total, nearest


def weak_bond(
    scn: Scenario, stranger: int, owner: int, bag: int, window: tuple[float, float], config: Config
) -> tuple[Premissa, ...]:
    """O estranho vira vínculo fraco sem encostar na bagagem nem co-mover."""
    t0, t1 = window
    s, o, b = scn.actor(stranger), scn.actor(owner), scn.actor(bag)
    longest, _, _ = within_stats(s, o, config.party.proximity_m, t0, t1)
    _, _, closest_bag = within_stats(s, b, config.pipeline.contact_radius_m, t0, t1)
    need = config.party.weak_bond_s + 2.0 * merge_interval_s(config)
    owner_drift = max(o.position(t).distance_to(o.position(t0)) for t in _samples(t0, t1))
    return (
        Premissa(
            f"o estranho {stranger} fica a ≤ party.proximity_m do dono {owner} por "
            "≥ weak_bond_s mais duas fusões: vira vínculo fraco",
            longest >= need,
            f"{longest:.1f} s contínuos (mínimo {need:.1f} s)",
        ),
        Premissa(
            f"enquanto isso o estranho {stranger} fica a > contact_radius_m da bagagem",
            closest_bag > config.pipeline.contact_radius_m,
            f"mínimo {closest_bag:.2f} m (raio {config.pipeline.contact_radius_m:.2f} m)",
        ),
        Premissa(
            f"o dono {owner} fica parado enquanto isso: não há co-movimento, só vínculo fraco",
            owner_drift < config.party.min_extent_m,
            f"deriva do dono {owner_drift:.3f} m (min_extent_m {config.party.min_extent_m})",
        ),
    )


def nearest_while_displaced(
    scn: Scenario,
    bag_id: int,
    expected: int,
    anchor: Point,
    window: tuple[float, float],
    config: Config,
) -> Premissa:
    """Com a bagagem fora da âncora, `expected` é quem está mais perto dela."""
    bag = scn.actor(bag_id)
    threshold = config.registry.moved_threshold_m
    limit = config.pipeline.owner_search_radius_m
    wrong = 0
    displaced = 0.0
    for t in _samples(*window):
        where = bag.position(t)
        if not bag.is_visible(t) or where.distance_to(anchor) < threshold:
            continue
        displaced += PREMISE_STEP_S
        people = [a for a in scn.actors if a.is_person and a.is_visible(t)]
        nearest = min(people, key=lambda p, t=t, where=where: p.position(t).distance_to(where))
        if nearest.track_id != expected or nearest.position(t).distance_to(where) > limit:
            wrong += 1
    need = config.custody.carry_confirm_s + 2.0 / MIN_FPS
    return Premissa(
        f"com a bagagem {bag_id} fora da âncora, o mais próximo dela é {expected}, "
        "a ≤ owner_search_radius_m, por ≥ carry_confirm_s",
        wrong == 0 and displaced >= need,
        f"{displaced:.1f} s deslocada (mínimo {need:.2f} s), {wrong} amostras divergentes",
    )


def windowed_quiet(
    scn: Scenario, pairs: list[tuple[int, int, tuple[float, float]]], cfg: Config
) -> tuple[Premissa, ...]:
    """`foreign_quiet` com janela por par: um track de mala muda de dono físico."""
    contact = cfg.pipeline.contact_radius_m
    near = cfg.pipeline.proximity_radius_m
    closest, longest = float("inf"), 0.0
    for who, bag, (t0, t1) in pairs:
        _, total, nearest = within_stats(scn.actor(who), scn.actor(bag), near, t0, t1)
        closest, longest = min(closest, nearest), max(longest, total)
    limit = cfg.pipeline.proximity_flag_s
    return (
        Premissa(
            "ninguém entra no raio de contato de bagagem alheia",
            closest > contact,
            f"mínimo {closest:.2f} m (raio {contact:.2f} m)",
        ),
        Premissa(
            "ninguém soma proximity_flag_s observado a ≤ proximity_radius_m de bagagem alheia",
            longest < limit,
            f"máximo {longest:.1f} s (limite {limit:.1f} s)",
        ),
    )


def foreign_quiet(
    scn: Scenario, pairs: tuple[tuple[int, int], ...], config: Config
) -> tuple[Premissa, ...]:
    """Ninguém encosta em bagagem alheia nem soma permanência perto dela."""
    whole = (0.0, scn.duration_s)
    return windowed_quiet(scn, [(who, bag, whole) for who, bag in pairs], config)


def stays_away(
    actor: Actor, anchor: Point, radius: float, window: tuple[float, float], label: str
) -> Premissa:
    """`actor` fica fora do raio (ou fora de quadro) durante a janela."""
    inside = sum(
        1
        for t in _samples(*window)
        if actor.is_visible(t) and actor.position(t).distance_to(anchor) <= radius
    )
    return Premissa(label, inside == 0, f"{inside} amostras dentro do raio {radius:.2f} m")


def stays_within(
    actor: Actor, anchor: Point, band: tuple[float, float], window: tuple[float, float], label: str
) -> Premissa:
    """`actor` visível e com distância à âncora dentro de `(band[0], band[1]]`."""
    low, high = band
    out = [
        actor.position(t).distance_to(anchor)
        for t in _samples(*window)
        if not actor.is_visible(t) or not low < actor.position(t).distance_to(anchor) <= high
    ]
    return Premissa(label, not out, f"{len(out)} amostras fora de ({low:.2f}, {high:.2f}] m")

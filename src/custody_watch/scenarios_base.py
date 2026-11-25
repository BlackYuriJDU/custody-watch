"""Cenas de base e de P4: depósito, abandono, furto simples, passante e lacuna."""

from __future__ import annotations

from .config import Config
from .scenario_model import (
    CARRY_OFFSET_M,
    DISTANCE_MARGIN_M,
    ENTRY_DISTANCE_M,
    EXIT_DISTANCE_M,
    FOREVER,
    MIN_FPS,
    OUTSIDE_AFTER_CROSSING_S,
    SLACK_S,
    WALK_SPEED_MS,
    Premissa,
    Scenario,
    Timing,
    accusation_due,
    arrive,
    crossing,
    entering,
    origin_behind,
    person,
    suitcase,
    unattended_due,
)
from .scenario_motion import Actor, Route, carried
from .scenario_premises import (
    depositor,
    foreign_quiet,
    nearest_while_displaced,
    rests,
    stays_away,
    stays_within,
    within_stats,
)
from .scenario_scoring import Expectation
from .types import Point

DETECTION_GAP_S = 150.0
"""Lacuna do cenário de permanência: contada como presença, cruzaria
`proximity_flag_s` com folga."""
OBSERVED_NEAR_FRACTION = 0.4
"""Fração de `proximity_flag_s` observada antes e depois da lacuna."""
PASSERBY_SPACING_FACTOR = 2.5
"""Malas do passante: vão em múltiplos de `proximity_radius_m`."""


def owner_returns(config: Config) -> Scenario:
    """base/legitimo: dono pousa, afasta-se além do prazo, volta e leva."""
    tm, owner_id, bag_id = Timing(config), 1, 101
    anchor, offset = Point(0.0, 0.0), (CARRY_OFFSET_M, 0.0)
    owner = arrive(0.0, anchor, offset, origin_behind(offset, anchor))
    t_dep = owner.t
    bag_in = carried(owner.waypoints(), 0.0, t_dep, offset)
    t_leave = owner.stay(tm.settle_s).t
    away = config.custody.unattended_distance_m + DISTANCE_MARGIN_M
    owner.walk_to(-away, 0.0, WALK_SPEED_MS).stay(tm.unattended_tolerance_s + SLACK_S)
    t_back = owner.t
    t_take = owner.walk_to(anchor.x - offset[0], 0.0, WALK_SPEED_MS).stay(SLACK_S).t
    t_exit = owner.walk_to(EXIT_DISTANCE_M, 0.0, WALK_SPEED_MS).t
    owner_actor = person(owner_id, owner, 0.0, t_exit)
    bag_path = [*bag_in, (t_take, anchor.x, anchor.y)]
    bag_path += carried(owner.waypoints(), t_take, t_exit, offset)
    radius = config.custody.unattended_distance_m
    t_cross = crossing(owner_actor, anchor, radius, t_leave, t_back)

    def checks(scn: Scenario, cfg: Config) -> tuple[Premissa, ...]:
        o = scn.actor(owner_id)
        back = entering(o, anchor, cfg.custody.unattended_distance_m, t_back, t_take)
        need = cfg.custody.unattended_time_s + 2.0 / MIN_FPS
        return (
            depositor(scn, bag_id, owner_id, cfg),
            rests(scn, bag_id, t_dep, t_leave, cfg),
            Premissa(
                "o dono fica a > unattended_distance_m por ≥ unattended_time_s",
                back - t_cross >= need,
                f"{back - t_cross:.1f} s fora (mínimo {need:.1f} s)",
            ),
            nearest_while_displaced(scn, bag_id, owner_id, anchor, (t_take, t_exit), cfg),
        )

    return Scenario(
        name="owner_returns",
        descricao="Dono pousa a mala, afasta-se além do prazo de desacompanhada, volta e leva.",
        invariant="base",
        kind="legitimo",
        roles=(("dono", owner_id), ("bagagem", bag_id)),
        expected=Expectation(
            never_accuse=frozenset({owner_id}),
            due=(unattended_due(bag_id, t_cross, tm),),
            silence=False,
        ),
        actors=(owner_actor, suitcase(bag_id, bag_path, ((0.0, t_exit),))),
        duration_s=t_exit + tm.tail_s,
        bag_box_height=tm.bag_box_height,
        marks=(
            ("deposito", t_dep),
            ("dono_cruza_raio", t_cross),
            ("retirada", t_take),
            ("ultima_acao", t_exit),
        ),
        checks=checks,
    )


def stranger_takes(config: Config) -> Scenario:
    """base/furto: dono sai de cena; depois do prazo, um estranho leva a mala."""
    tm, owner_id, stranger_id, bag_id = Timing(config), 1, 3, 101
    anchor, offset = Point(0.0, 0.0), (CARRY_OFFSET_M, 0.0)
    owner = arrive(0.0, anchor, offset, origin_behind(offset, anchor))
    t_dep = owner.t
    bag_in = carried(owner.waypoints(), 0.0, t_dep, offset)
    t_leave = owner.stay(tm.settle_s).t
    t_owner_exit = owner.walk_to(anchor.x - offset[0], -EXIT_DISTANCE_M, WALK_SPEED_MS).t
    owner_actor = person(owner_id, owner, 0.0, t_owner_exit)
    radius = config.custody.unattended_distance_m
    t_cross = crossing(owner_actor, anchor, radius, t_leave, t_owner_exit)

    take = (-CARRY_OFFSET_M, 0.0)
    grab = Point(anchor.x - take[0], anchor.y - take[1])
    t_x0 = t_cross + tm.unattended_tolerance_s + SLACK_S
    stranger = Route(t_x0, grab.x + ENTRY_DISTANCE_M, grab.y)
    t_take = stranger.walk_to(grab.x, grab.y, WALK_SPEED_MS).stay(SLACK_S).t
    t_exit = stranger.walk_to(grab.x + EXIT_DISTANCE_M, grab.y, WALK_SPEED_MS).t
    bag_path = [*bag_in, (t_take, anchor.x, anchor.y)]
    bag_path += carried(stranger.waypoints(), t_take, t_exit, take)

    def checks(scn: Scenario, cfg: Config) -> tuple[Premissa, ...]:
        due_by = t_cross + cfg.custody.unattended_time_s + 2.0 / MIN_FPS
        return (
            depositor(scn, bag_id, owner_id, cfg),
            rests(scn, bag_id, t_dep, t_leave, cfg),
            stays_away(
                scn.actor(owner_id),
                anchor,
                cfg.custody.unattended_distance_m,
                (t_cross + OUTSIDE_AFTER_CROSSING_S, scn.duration_s),
                "o dono fica a > unattended_distance_m, ou fora de quadro, até o fim",
            ),
            Premissa(
                "o estranho só aparece depois do prazo de desacompanhada",
                t_x0 >= due_by,
                f"chega em {t_x0:.1f} s (prazo {due_by:.1f} s)",
            ),
            nearest_while_displaced(scn, bag_id, stranger_id, anchor, (t_take, t_exit), cfg),
        )

    return Scenario(
        name="stranger_takes",
        descricao="Dono sai de cena; depois do prazo, um estranho leva a mala.",
        invariant="base",
        kind="furto",
        roles=(("dono", owner_id), ("estranho", stranger_id), ("bagagem", bag_id)),
        expected=Expectation(
            never_accuse=frozenset({owner_id}),
            due=(
                unattended_due(bag_id, t_cross, tm),
                accusation_due(stranger_id, t_take, tm, WALK_SPEED_MS),
            ),
            silence=False,
        ),
        actors=(
            owner_actor,
            person(stranger_id, stranger, t_x0, t_exit),
            suitcase(bag_id, bag_path, ((0.0, t_exit),)),
        ),
        duration_s=t_exit + tm.tail_s,
        bag_box_height=tm.bag_box_height,
        marks=(
            ("deposito", t_dep),
            ("dono_cruza_raio", t_cross),
            ("retirada", t_take),
            ("ultima_acao", t_exit),
        ),
        checks=checks,
    )


def passerby_no_luggage(config: Config) -> Scenario:
    """P4/legitimo: passante sem bagagem cruza perto de três malas sem parar."""
    tm, passer_id = Timing(config), 7
    owners, bags = (1, 2, 3), (101, 102, 103)
    spacing = PASSERBY_SPACING_FACTOR * config.pipeline.proximity_radius_m
    offset = (0.0, -CARRY_OFFSET_M)
    actors: list[Actor] = []
    deposits: list[tuple[float, float]] = []
    t = 0.0
    for i, (owner_id, bag_id) in enumerate(zip(owners, bags, strict=True)):
        anchor = Point(i * spacing, 0.0)
        route = arrive(t, anchor, offset, origin_behind(offset, anchor))
        bag_path = carried(route.waypoints(), t, route.t, offset)
        actors += [person(owner_id, route, t, FOREVER), suitcase(bag_id, bag_path, ((t, FOREVER),))]
        deposits.append((route.t, route.t + tm.settle_s))
        t = route.t + tm.settle_s
    lane = (config.pipeline.contact_radius_m + config.pipeline.proximity_radius_m) / 2.0
    passer = Route(t, -ENTRY_DISTANCE_M, -lane)
    t_exit = passer.walk_to(2.0 * spacing + ENTRY_DISTANCE_M, -lane, WALK_SPEED_MS).t
    actors.append(person(passer_id, passer, t, t_exit))

    def checks(scn: Scenario, cfg: Config) -> tuple[Premissa, ...]:
        pairs = [(passer_id, b) for b in bags]
        pairs += [(o, b) for i, o in enumerate(owners) for j, b in enumerate(bags) if i != j]
        nearest = max(
            within_stats(scn.actor(passer_id), scn.actor(b), 0.0, 0.0, scn.duration_s)[2]
            for b in bags
        )
        return (
            *(depositor(scn, b, o, cfg) for o, b in zip(owners, bags, strict=True)),
            *(rests(scn, b, t0, t1, cfg) for b, (t0, t1) in zip(bags, deposits, strict=True)),
            Premissa(
                "o passante cruza a ≤ proximity_radius_m de cada mala",
                nearest <= cfg.pipeline.proximity_radius_m,
                f"maior aproximação mínima {nearest:.2f} m",
            ),
            *foreign_quiet(scn, tuple(pairs), cfg),
        )

    return Scenario(
        name="passerby_no_luggage",
        descricao=(
            "Passante sem bagagem cruza perto de três malas sem parar: nenhum flag de "
            "permanência ou contato."
        ),
        invariant="P4",
        kind="legitimo",
        roles=(
            *((f"dono_{i + 1}", o) for i, o in enumerate(owners)),
            ("passante", passer_id),
            *((f"bagagem_{i + 1}", b) for i, b in enumerate(bags)),
        ),
        expected=Expectation(never_accuse=frozenset({*owners, passer_id}), due=(), silence=True),
        actors=tuple(actors),
        duration_s=t_exit + tm.tail_s,
        bag_box_height=tm.bag_box_height,
        marks=(("passante_entra", t), ("ultima_acao", t_exit)),
        checks=checks,
    )


def detection_gap_near_bag(config: Config) -> Scenario:
    """base/legitimo: pessoa some 150 s perto da mala e volta; sem permanência fabricada."""
    tm, owner_id, waiter_id, bag_id = Timing(config), 1, 8, 101
    anchor, offset = Point(0.0, 0.0), (CARRY_OFFSET_M, 0.0)
    owner = arrive(0.0, anchor, offset, origin_behind(offset, anchor))
    t_dep = owner.t
    bag_in = carried(owner.waypoints(), 0.0, t_dep, offset)
    spot = Point(anchor.x - config.custody.unattended_distance_m / 2.0, anchor.y)
    owner.walk_to(spot.x, spot.y, WALK_SPEED_MS)
    pipeline = config.pipeline
    seat = Point(anchor.x + (pipeline.contact_radius_m + pipeline.proximity_radius_m) / 2.0, 0.0)
    t_w0 = t_dep + tm.settle_s
    waiter = Route(t_w0, seat.x + ENTRY_DISTANCE_M, seat.y)
    stay = OBSERVED_NEAR_FRACTION * pipeline.proximity_flag_s
    t_gap0 = waiter.walk_to(seat.x, seat.y, WALK_SPEED_MS).stay(stay).t
    t_gap1 = waiter.stay(DETECTION_GAP_S).t
    t_exit = waiter.stay(stay).walk_to(seat.x + EXIT_DISTANCE_M, seat.y, WALK_SPEED_MS).t
    waiter_actor = Actor(
        waiter_id, "person", waiter.waypoints(), ((t_w0, t_gap0), (t_gap1, t_exit))
    )
    bag_path = [*bag_in, (t_exit, anchor.x, anchor.y)]

    def checks(scn: Scenario, cfg: Config) -> tuple[Premissa, ...]:
        w, bag = scn.actor(waiter_id), scn.actor(bag_id)
        _, observed, _ = within_stats(w, bag, cfg.pipeline.proximity_radius_m, 0.0, scn.duration_s)
        limit = cfg.pipeline.proximity_flag_s
        return (
            depositor(scn, bag_id, owner_id, cfg),
            rests(scn, bag_id, t_dep, t_w0, cfg),
            *foreign_quiet(scn, ((waiter_id, bag_id),), cfg),
            Premissa(
                "contada como presença, a lacuna levaria a permanência além de proximity_flag_s",
                observed + (t_gap1 - t_gap0) >= limit,
                f"{observed:.1f} s observados + {t_gap1 - t_gap0:.1f} s de lacuna "
                f"(limite {limit:.1f} s)",
            ),
            stays_away(
                w,
                scn.actor(owner_id).position(scn.duration_s),
                cfg.party.proximity_m,
                (t_w0, t_exit),
                "a pessoa nunca fica a ≤ party.proximity_m do dono: sem vínculo que mascare",
            ),
            stays_within(
                scn.actor(owner_id),
                anchor,
                (0.0, cfg.custody.unattended_distance_m),
                (t_dep, scn.duration_s),
                "o dono fica a ≤ unattended_distance_m: a mala nunca fica desacompanhada",
            ),
        )

    return Scenario(
        name="detection_gap_near_bag",
        descricao=(
            "Pessoa espera perto de uma mala alheia, some do detector por 150 s e volta "
            "ao mesmo lugar: o tempo não observado não vira permanência."
        ),
        invariant="base",
        kind="legitimo",
        roles=(("dono", owner_id), ("pessoa", waiter_id), ("bagagem", bag_id)),
        expected=Expectation(never_accuse=frozenset({owner_id, waiter_id}), due=(), silence=True),
        actors=(
            person(owner_id, owner, 0.0, FOREVER),
            waiter_actor,
            suitcase(bag_id, bag_path, ((0.0, FOREVER),)),
        ),
        duration_s=t_exit + tm.tail_s,
        bag_box_height=tm.bag_box_height,
        marks=(("lacuna_inicio", t_gap0), ("lacuna_fim", t_gap1), ("ultima_acao", t_exit)),
        checks=checks,
    )

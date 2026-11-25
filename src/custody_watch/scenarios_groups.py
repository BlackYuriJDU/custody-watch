"""Cenas de P1: posse só flui por vínculo forte."""

from __future__ import annotations

from .config import Config
from .scenario_model import (
    CARRY_OFFSET_M,
    ENTRY_DISTANCE_M,
    EXIT_DISTANCE_M,
    FOREVER,
    MIN_FPS,
    OUTSIDE_AFTER_CROSSING_S,
    PREMISE_STEP_S,
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
from .scenario_motion import Route, Waypoint, carried, sample_times
from .scenario_premises import (
    depositor,
    nearest_while_displaced,
    rests,
    stays_away,
    stays_within,
    weak_bond,
)
from .scenario_scoring import Category, DueSignal, Expectation
from .types import Point

SIT_BESIDE_FRACTION = 0.8
"""Quão perto do dono o estranho senta, em fração de `party.proximity_m`."""


def companion_takes(config: Config) -> Scenario:
    """P1/legitimo: quem CHEGOU JUNTO com o dono (vínculo forte) leva a mala."""
    tm, owner_id, companion_id, bag_id = Timing(config), 1, 2, 101
    anchor, offset = Point(0.0, 0.0), (CARRY_OFFSET_M, 0.0)
    side = config.party.proximity_m / 2.0
    start_x = anchor.x - offset[0] - WALK_SPEED_MS * tm.strong_walk_s
    owner = arrive(0.0, anchor, offset, Point(start_x, 0.0))
    companion = Route(0.0, start_x, side).walk_to(anchor.x - offset[0], side, WALK_SPEED_MS)
    t_dep = owner.t
    bag_in = carried(owner.waypoints(), 0.0, t_dep, offset)
    t_leave = owner.stay(tm.settle_s).t
    t_owner_exit = owner.walk_to(anchor.x - offset[0], -EXIT_DISTANCE_M, WALK_SPEED_MS).t
    companion.stay_until(t_owner_exit + SLACK_S)
    t_take = companion.walk_to(anchor.x - offset[0], 0.0, WALK_SPEED_MS).stay(SLACK_S).t
    t_exit = companion.walk_to(EXIT_DISTANCE_M, 0.0, WALK_SPEED_MS).t
    bag_path = [*bag_in, (t_take, anchor.x, anchor.y)]
    bag_path += carried(companion.waypoints(), t_take, t_exit, offset)

    def checks(scn: Scenario, cfg: Config) -> tuple[Premissa, ...]:
        o, c = scn.actor(owner_id), scn.actor(companion_id)
        gap = max(
            o.position(t).distance_to(c.position(t))
            for t in sample_times(0.0, t_dep, PREMISE_STEP_S)
        )
        extent = min(a.position(0.0).distance_to(a.position(t_dep)) for a in (o, c))
        late = cfg.party.late_join_extent_m
        need = late / WALK_SPEED_MS + cfg.pipeline.merge_every_frames / MIN_FPS
        return (
            depositor(scn, bag_id, owner_id, cfg),
            rests(scn, bag_id, t_dep, t_leave, cfg),
            Premissa(
                "dono e acompanhante chegam juntos, sempre a ≤ party.proximity_m",
                gap <= cfg.party.proximity_m,
                f"separação máxima {gap:.2f} m",
            ),
            Premissa(
                "juntos, cada um cobre ≥ late_join_extent_m por tempo de ao menos uma fusão",
                extent >= late and t_dep >= need,
                f"{extent:.1f} m em {t_dep:.1f} s (mínimo {late:.1f} m em {need:.1f} s)",
            ),
            Premissa(
                "history_window_s comporta late_join_extent_m de caminhada",
                cfg.pipeline.history_window_s * WALK_SPEED_MS >= late,
                f"{cfg.pipeline.history_window_s * WALK_SPEED_MS:.1f} m na janela",
            ),
            stays_within(
                c,
                anchor,
                (0.0, cfg.custody.unattended_distance_m),
                (t_dep, t_take),
                "o acompanhante fica a ≤ unattended_distance_m: a mala nunca fica sozinha",
            ),
            nearest_while_displaced(scn, bag_id, companion_id, anchor, (t_take, t_exit), cfg),
        )

    return Scenario(
        name="companion_takes",
        descricao="Acompanhante que chegou junto com o dono (vínculo forte) leva a mala.",
        invariant="P1",
        kind="legitimo",
        roles=(("dono", owner_id), ("acompanhante", companion_id), ("bagagem", bag_id)),
        expected=Expectation(
            never_accuse=frozenset({owner_id, companion_id}), due=(), silence=True
        ),
        actors=(
            person(owner_id, owner, 0.0, t_owner_exit),
            person(companion_id, companion, 0.0, t_exit),
            suitcase(bag_id, bag_path, ((0.0, t_exit),)),
        ),
        duration_s=t_exit + tm.tail_s,
        bag_box_height=tm.bag_box_height,
        marks=(("deposito", t_dep), ("retirada", t_take), ("ultima_acao", t_exit)),
        checks=checks,
    )


def _seated_owner_and_stranger(
    config: Config, tm: Timing
) -> tuple[Route, Route, list[Waypoint], float, float, float, Point]:
    """Dono deposita e fica; estranho senta ao lado dele até virar vínculo fraco.

    Devolve (dono, estranho, bagagem trazida, depósito, fim da espera,
    lugar do dono, lugar do estranho).
    """
    anchor, offset = Point(0.0, 0.0), (CARRY_OFFSET_M, 0.0)
    owner = arrive(0.0, anchor, offset, origin_behind(offset, anchor))
    t_dep = owner.t
    bag_in = carried(owner.waypoints(), 0.0, t_dep, offset)
    spot = owner.here
    seat = Point(spot.x - SIT_BESIDE_FRACTION * config.party.proximity_m, spot.y)
    stranger = Route(t_dep + tm.settle_s, seat.x, seat.y - ENTRY_DISTANCE_M)
    t_sit = stranger.walk_to(seat.x, seat.y, WALK_SPEED_MS).t
    t_bond = stranger.stay(tm.weak_bond_wait_s).t
    return owner, stranger, bag_in, t_dep, t_sit, t_bond, spot


def sit_beside_then_take(config: Config) -> Scenario:
    """P1/furto: senta ao lado do dono até virar vínculo fraco, encosta e leva."""
    tm, owner_id, stranger_id, bag_id = Timing(config), 1, 4, 101
    anchor = Point(0.0, 0.0)
    owner, stranger, bag_in, t_dep, t_sit, t_rise, spot = _seated_owner_and_stranger(config, tm)
    take = (-CARRY_OFFSET_M, 0.0)
    grab = Point(anchor.x - take[0], anchor.y - take[1])
    stranger.walk_to(spot.x, spot.y - 2.0 * CARRY_OFFSET_M, WALK_SPEED_MS)
    t_take = stranger.walk_to(grab.x, grab.y, WALK_SPEED_MS).stay(SLACK_S).t
    t_exit = stranger.walk_to(grab.x + EXIT_DISTANCE_M, grab.y, WALK_SPEED_MS).t
    stranger_actor = person(stranger_id, stranger, t_dep + tm.settle_s, t_exit)
    contact = config.pipeline.contact_radius_m
    t_touch = entering(stranger_actor, anchor, contact, t_rise, t_take)
    bag_path = [*bag_in, (t_take, anchor.x, anchor.y)]
    bag_path += carried(stranger.waypoints(), t_take, t_exit, take)

    def checks(scn: Scenario, cfg: Config) -> tuple[Premissa, ...]:
        s = scn.actor(stranger_id)
        return (
            depositor(scn, bag_id, owner_id, cfg),
            rests(scn, bag_id, t_dep, t_dep + tm.settle_s, cfg),
            *weak_bond(scn, stranger_id, owner_id, bag_id, (t_sit, t_rise), cfg),
            stays_away(
                s,
                anchor,
                cfg.pipeline.proximity_radius_m,
                (t_dep, t_rise),
                "sentado, o estranho fica a > proximity_radius_m da mala: nada depende de N1",
            ),
            Premissa(
                "o estranho encosta (entra no raio de contato) antes de levar",
                t_touch < t_take,
                f"encosta em {t_touch:.1f} s, leva em {t_take:.1f} s",
            ),
            nearest_while_displaced(scn, bag_id, stranger_id, anchor, (t_take, t_exit), cfg),
        )

    return Scenario(
        name="sit_beside_then_take",
        descricao=(
            "Estranho senta ao lado do dono até virar vínculo fraco, depois encosta na "
            "mala e a leva: deve continuar sinalizado."
        ),
        invariant="P1",
        kind="furto",
        roles=(("dono", owner_id), ("estranho", stranger_id), ("bagagem", bag_id)),
        expected=Expectation(
            never_accuse=frozenset({owner_id}),
            due=(
                DueSignal(
                    Category.PERSON_SIGNAL,
                    stranger_id,
                    t_touch,
                    tm.contact_tolerance_s,
                    "contato com bagagem alheia, mesmo com vínculo fraco",
                ),
                accusation_due(stranger_id, t_take, tm, WALK_SPEED_MS),
            ),
            silence=False,
        ),
        actors=(
            person(owner_id, owner, 0.0, FOREVER),
            stranger_actor,
            suitcase(bag_id, bag_path, ((0.0, t_exit),)),
        ),
        duration_s=t_exit + tm.tail_s,
        bag_box_height=tm.bag_box_height,
        marks=(
            ("deposito", t_dep),
            ("vinculo_fraco", t_rise),
            ("encosta", t_touch),
            ("retirada", t_take),
            ("ultima_acao", t_exit),
        ),
        checks=checks,
    )


def owner_leaves_weak_stays(config: Config) -> Scenario:
    """P1/furto: dono sai, fica só o vínculo fraco; a mala está desacompanhada."""
    tm, owner_id, stranger_id, bag_id = Timing(config), 1, 4, 101
    anchor = Point(0.0, 0.0)
    owner, stranger, bag_in, t_dep, t_sit, t_bond, spot = _seated_owner_and_stranger(config, tm)
    t_leave = owner.stay_until(t_bond).t
    t_owner_exit = owner.walk_to(spot.x, -EXIT_DISTANCE_M, WALK_SPEED_MS).t
    owner_actor = person(owner_id, owner, 0.0, t_owner_exit)
    radius = config.custody.unattended_distance_m
    t_cross = crossing(owner_actor, anchor, radius, t_leave, t_owner_exit)
    last = max(t_owner_exit, t_cross + tm.unattended_tolerance_s + SLACK_S)
    bag_path = [*bag_in, (last, anchor.x, anchor.y)]

    def checks(scn: Scenario, cfg: Config) -> tuple[Premissa, ...]:
        band = (cfg.pipeline.proximity_radius_m, cfg.custody.unattended_distance_m)
        return (
            depositor(scn, bag_id, owner_id, cfg),
            rests(scn, bag_id, t_dep, t_dep + tm.settle_s, cfg),
            *weak_bond(scn, stranger_id, owner_id, bag_id, (t_sit, t_leave), cfg),
            stays_within(
                scn.actor(stranger_id),
                anchor,
                band,
                (t_sit, scn.duration_s),
                "o estranho fica a ≤ unattended_distance_m da mala -- contá-lo como "
                "acompanhante mascara o abandono -- e a > proximity_radius_m (sem N1)",
            ),
            stays_away(
                scn.actor(owner_id),
                anchor,
                cfg.custody.unattended_distance_m,
                (t_cross + OUTSIDE_AFTER_CROSSING_S, scn.duration_s),
                "o dono não volta",
            ),
        )

    return Scenario(
        name="owner_leaves_weak_stays",
        descricao=(
            "Dono sai de cena e fica só o estranho com vínculo fraco ao lado da mala: "
            "ela deve ficar desacompanhada no prazo."
        ),
        invariant="P1",
        kind="furto",
        roles=(("dono", owner_id), ("estranho", stranger_id), ("bagagem", bag_id)),
        expected=Expectation(
            never_accuse=frozenset({owner_id, stranger_id}),
            due=(unattended_due(bag_id, t_cross, tm),),
            silence=False,
        ),
        actors=(
            owner_actor,
            person(stranger_id, stranger, t_dep + tm.settle_s, FOREVER),
            suitcase(bag_id, bag_path, ((0.0, FOREVER),)),
        ),
        duration_s=last + tm.tail_s,
        bag_box_height=tm.bag_box_height,
        marks=(
            ("deposito", t_dep),
            ("vinculo_fraco", t_bond),
            ("dono_cruza_raio", t_cross),
            ("ultima_acao", last),
        ),
        checks=checks,
    )

"""Cenas de P2 e P3: posição é identidade, e incerteza cala."""

from __future__ import annotations

from .config import Config
from .scenario_model import (
    CARRY_OFFSET_M,
    DISTANCE_MARGIN_M,
    ENTRY_DISTANCE_M,
    EXIT_DISTANCE_M,
    FOREVER,
    MIN_FPS,
    PREMISE_STEP_S,
    QUEUE_SPEED_MS,
    SLACK_S,
    WALK_SPEED_MS,
    Premissa,
    Scenario,
    Timing,
    arrive,
    entering,
    origin_behind,
    person,
    suitcase,
)
from .scenario_motion import Route, carried, sample_times
from .scenario_premises import (
    depositor,
    nearest_while_displaced,
    rests,
    stays_away,
    windowed_quiet,
    within_stats,
)
from .scenario_scoring import Expectation
from .types import Point

TWIN_SPACING_FACTOR = 1.5
"""Malas gêmeas: vão entre âncoras, em múltiplos do maior raio de registro."""
AMBIGUOUS_SPACING_FACTOR = 1.8
"""Malas ambíguas: vão em múltiplos do raio de readoção (o meio cai nas duas)."""
QUEUE_NEIGHBOUR_GAP_M = 0.35
"""Na fila, o vizinho fica colado atrás da mala, mais perto dela que o dono."""


def _twin_bags(config: Config, swapped: bool) -> Scenario:
    """P2/legitimo: duas malas iguais vizinhas, oclusão, os tracks voltam trocados."""
    tm = Timing(config)
    a_id, b_id, passer_id, bag_a, bag_b = 1, 2, 5, 101, 102
    after_a, after_b = (bag_b, bag_a) if swapped else (112, 111)
    registry = config.registry
    spacing = TWIN_SPACING_FACTOR * max(
        registry.reidentification_radius_m, registry.ambiguity_radius_m
    )
    anchor_a, anchor_b = Point(0.0, 0.0), Point(spacing, 0.0)
    off_a, off_b = (CARRY_OFFSET_M, 0.0), (-CARRY_OFFSET_M, 0.0)
    owner_a = arrive(0.0, anchor_a, off_a, origin_behind(off_a, anchor_a))
    t_dep_a = owner_a.t
    t_b0 = t_dep_a + tm.settle_s
    owner_b = arrive(t_b0, anchor_b, off_b, origin_behind(off_b, anchor_b))
    t_dep_b = owner_b.t
    lane = (config.pipeline.contact_radius_m + config.pipeline.proximity_radius_m) / 2.0
    t_p0 = t_dep_b + tm.settle_s
    passer = Route(t_p0, -ENTRY_DISTANCE_M, -lane)
    t_occ0 = passer.walk_to(spacing / 2.0, -lane, WALK_SPEED_MS).t
    t_occ1 = passer.stay(tm.occlusion_s).t
    t_p_exit = passer.walk_to(spacing + EXIT_DISTANCE_M, -lane, WALK_SPEED_MS).t
    t_take = t_occ1 + tm.settle_s
    owner_a.stay_until(t_take).walk_to(anchor_a.x - off_a[0] - EXIT_DISTANCE_M, 0.0, WALK_SPEED_MS)
    owner_b.stay_until(t_take).walk_to(anchor_b.x - off_b[0] + EXIT_DISTANCE_M, 0.0, WALK_SPEED_MS)
    t_exit_a, t_exit_b = owner_a.t, owner_b.t

    def before(route: Route, t0: float, t_dep: float, off, anchor: Point) -> list:
        return [*carried(route.waypoints(), t0, t_dep, off), (t_occ0, anchor.x, anchor.y)]

    def after(route: Route, t_exit: float, off, anchor: Point) -> list:
        rest = [(t_occ1, anchor.x, anchor.y), (t_take, anchor.x, anchor.y)]
        return rest + carried(route.waypoints(), t_take, t_exit, off)

    pa_before = before(owner_a, 0.0, t_dep_a, off_a, anchor_a)
    pb_before = before(owner_b, t_b0, t_dep_b, off_b, anchor_b)
    pa_after = after(owner_a, t_exit_a, off_a, anchor_a)
    pb_after = after(owner_b, t_exit_b, off_b, anchor_b)
    if swapped:
        bags = (
            suitcase(bag_a, pa_before + pb_after, ((0.0, t_occ0), (t_occ1, t_exit_b))),
            suitcase(bag_b, pb_before + pa_after, ((t_b0, t_occ0), (t_occ1, t_exit_a))),
        )
    else:
        bags = (
            suitcase(bag_a, pa_before, ((0.0, t_occ0),)),
            suitcase(bag_b, pb_before, ((t_b0, t_occ0),)),
            suitcase(after_a, pa_after, ((t_occ1, t_exit_a),)),
            suitcase(after_b, pb_after, ((t_occ1, t_exit_b),)),
        )
    t_last = max(t_exit_a, t_exit_b, t_p_exit)

    def checks(scn: Scenario, cfg: Config) -> tuple[Premissa, ...]:
        reg = cfg.registry
        gap = anchor_a.distance_to(anchor_b)
        missing = cfg.pipeline.missing_frames_before_occluded
        everything = (0.0, scn.duration_s)
        early, late = (0.0, t_occ0), (t_occ1, scn.duration_s)
        pairs = [
            (a_id, bag_b, early),
            (b_id, bag_a, early),
            (passer_id, bag_a, everything),
            (passer_id, bag_b, everything),
            (a_id, after_b, late),
            (b_id, after_a, late),
            (passer_id, after_a, late),
            (passer_id, after_b, late),
        ]
        back_a = scn.actor(after_a).position(t_occ1).distance_to(anchor_a)
        back_b = scn.actor(after_b).position(t_occ1).distance_to(anchor_b)
        return (
            depositor(scn, bag_a, a_id, cfg),
            depositor(scn, bag_b, b_id, cfg),
            rests(scn, bag_a, t_dep_a, t_b0, cfg),
            rests(scn, bag_b, t_dep_b, t_p0, cfg),
            Premissa(
                "as âncoras ficam a > reidentification_radius_m e > ambiguity_radius_m "
                "uma da outra: cada mala é readotável sem ambiguidade",
                gap > max(reg.reidentification_radius_m, reg.ambiguity_radius_m),
                f"vão {gap:.2f} m",
            ),
            Premissa(
                "a oclusão dura mais que missing_frames_before_occluded quadros a MIN_FPS",
                (t_occ1 - t_occ0) * MIN_FPS > missing + 1,
                f"{t_occ1 - t_occ0:.2f} s",
            ),
            Premissa(
                "cada mala reaparece exatamente na própria âncora",
                max(back_a, back_b) <= 0.0,
                f"desvios {back_a:.3f} m e {back_b:.3f} m",
            ),
            *windowed_quiet(scn, pairs, cfg),
            nearest_while_displaced(scn, after_a, a_id, anchor_a, (t_take, t_exit_a), cfg),
            nearest_while_displaced(scn, after_b, b_id, anchor_b, (t_take, t_exit_b), cfg),
        )

    roles = [
        ("dono_a", a_id),
        ("dono_b", b_id),
        ("passante", passer_id),
        ("bagagem_a", bag_a),
        ("bagagem_b", bag_b),
    ]
    if not swapped:
        roles += [("bagagem_a_novo_track", after_a), ("bagagem_b_novo_track", after_b)]
    how = "com os tracks uma da outra" if swapped else "com tracks novos"
    return Scenario(
        name="twin_bags_swapped_ids" if swapped else "twin_bags_new_ids",
        descricao=(
            f"Duas malas idênticas vizinhas somem atrás de um passante e voltam {how}; "
            "cada dono leva a sua: ninguém é acusado."
        ),
        invariant="P2",
        kind="legitimo",
        roles=tuple(roles),
        expected=Expectation(never_accuse=frozenset({a_id, b_id, passer_id}), due=(), silence=True),
        actors=(
            person(a_id, owner_a, 0.0, t_exit_a),
            person(b_id, owner_b, t_b0, t_exit_b),
            person(passer_id, passer, t_p0, t_p_exit),
            *bags,
        ),
        duration_s=t_last + tm.tail_s,
        bag_box_height=tm.bag_box_height,
        marks=(
            ("oclusao", t_occ0),
            ("reaparece", t_occ1),
            ("retirada", t_take),
            ("ultima_acao", t_last),
        ),
        checks=checks,
    )


def twin_bags_new_ids(config: Config) -> Scenario:
    return _twin_bags(config, swapped=False)


def twin_bags_swapped_ids(config: Config) -> Scenario:
    return _twin_bags(config, swapped=True)


def new_track_between_occluded(config: Config) -> Scenario:
    """P3/ambiguo: duas malas ocluídas e um track novo no meio das duas."""
    tm = Timing(config)
    a_id, b_id, handler_id, bag_a, bag_b, merged_id = 1, 2, 5, 101, 102, 120
    spacing = AMBIGUOUS_SPACING_FACTOR * config.registry.reidentification_radius_m
    anchor_a, anchor_b = Point(0.0, 0.0), Point(spacing, 0.0)
    middle = Point(spacing / 2.0, 0.0)
    off_a, off_b = (CARRY_OFFSET_M, 0.0), (-CARRY_OFFSET_M, 0.0)
    owner_a = arrive(0.0, anchor_a, off_a, origin_behind(off_a, anchor_a))
    t_dep_a = owner_a.t
    t_b0 = t_dep_a + tm.settle_s
    owner_b = arrive(t_b0, anchor_b, off_b, origin_behind(off_b, anchor_b))
    t_dep_b = owner_b.t
    t_go = t_dep_b + tm.settle_s
    away = config.pipeline.owner_search_radius_m + DISTANCE_MARGIN_M
    owner_a.stay_until(t_go).walk_to(anchor_a.x - away, 0.0, WALK_SPEED_MS)
    owner_b.stay_until(t_go).walk_to(anchor_b.x + away, 0.0, WALK_SPEED_MS)
    t_h0 = max(owner_a.t, owner_b.t) + SLACK_S
    stop = Point(middle.x, middle.y - CARRY_OFFSET_M)
    handler = Route(t_h0, stop.x, stop.y - ENTRY_DISTANCE_M)
    t_occ0 = handler.walk_to(stop.x, stop.y, WALK_SPEED_MS).t
    t_occ1 = handler.stay(tm.occlusion_s).t
    t_h_exit = handler.walk_to(stop.x, stop.y - EXIT_DISTANCE_M, WALK_SPEED_MS).t
    path_a = [*carried(owner_a.waypoints(), 0.0, t_dep_a, off_a), (t_occ0, 0.0, 0.0)]
    path_b = [*carried(owner_b.waypoints(), t_b0, t_dep_b, off_b), (t_occ0, spacing, 0.0)]

    def checks(scn: Scenario, cfg: Config) -> tuple[Premissa, ...]:
        radius = cfg.registry.reidentification_radius_m
        search = cfg.pipeline.owner_search_radius_m
        absence = (t_occ0, scn.duration_s)
        missing = cfg.pipeline.missing_frames_before_occluded
        _, _, handler_min = within_stats(
            scn.actor(handler_id), scn.actor(bag_b), search, t_occ0, t_occ1
        )
        owners_away = [
            stays_away(scn.actor(o), anchor, search, absence, "")
            for o in (a_id, b_id)
            for anchor in (anchor_a, anchor_b)
        ]
        return (
            depositor(scn, bag_a, a_id, cfg),
            depositor(scn, bag_b, b_id, cfg),
            rests(scn, bag_a, t_dep_a, t_b0, cfg),
            rests(scn, bag_b, t_dep_b, t_go, cfg),
            Premissa(
                "o meio fica a ≤ reidentification_radius_m das duas âncoras: readoção ambígua",
                middle.distance_to(anchor_a) <= radius and middle.distance_to(anchor_b) <= radius,
                f"{middle.distance_to(anchor_a):.2f} m de cada (raio {radius:.2f} m)",
            ),
            Premissa(
                "as âncoras ficam a > ambiguity_radius_m uma da outra: marcar só a "
                "vizinhança de uma deixa a outra de fora",
                spacing > cfg.registry.ambiguity_radius_m,
                f"vão {spacing:.2f} m (raio {cfg.registry.ambiguity_radius_m:.2f} m)",
            ),
            Premissa(
                "o track novo aparece com as duas já declaradas ocluídas",
                (t_occ1 - t_occ0) * MIN_FPS > missing + 1,
                f"{t_occ1 - t_occ0:.2f} s depois do sumiço",
            ),
            Premissa(
                "durante a ausência, só o funcionário passa a ≤ owner_search_radius_m",
                all(p.valida for p in owners_away) and handler_min <= search,
                f"funcionário a {handler_min:.2f} m; donos fora: "
                f"{sum(p.valida for p in owners_away)}/{len(owners_away)}",
            ),
        )

    return Scenario(
        name="new_track_between_occluded",
        descricao=(
            "Um funcionário para na frente de duas malas a 1,8 raio de readoção; as duas "
            "somem e volta uma caixa só, no meio: ninguém é acusado."
        ),
        invariant="P3",
        kind="ambiguo",
        roles=(
            ("dono_a", a_id),
            ("dono_b", b_id),
            ("funcionario", handler_id),
            ("bagagem_a", bag_a),
            ("bagagem_b", bag_b),
            ("track_do_meio", merged_id),
        ),
        expected=Expectation(
            never_accuse=frozenset({a_id, b_id, handler_id}), due=(), silence=False
        ),
        actors=(
            person(a_id, owner_a, 0.0, FOREVER),
            person(b_id, owner_b, t_b0, FOREVER),
            person(handler_id, handler, t_h0, t_h_exit),
            suitcase(bag_a, path_a, ((0.0, t_occ0),)),
            suitcase(bag_b, path_b, ((t_b0, t_occ0),)),
            suitcase(merged_id, [(t_occ1, middle.x, middle.y)], ((t_occ1, FOREVER),)),
        ),
        duration_s=t_h_exit + tm.tail_s,
        bag_box_height=tm.bag_box_height,
        marks=(("oclusao", t_occ0), ("track_novo", t_occ1), ("ultima_acao", t_h_exit)),
        checks=checks,
    )


def owner_pulls_neighbour_closer(config: Config) -> Scenario:
    """P3/ambiguo: na fila, o dono puxa a mala com o vizinho mais perto dela."""
    tm, owner_id, neighbour_id, bag_id = Timing(config), 1, 6, 101
    anchor, offset = Point(0.0, 0.0), (-CARRY_OFFSET_M, 0.0)
    owner = arrive(0.0, anchor, offset, origin_behind(offset, anchor))
    t_dep = owner.t
    bag_in = carried(owner.waypoints(), 0.0, t_dep, offset)
    behind = Point(anchor.x - QUEUE_NEIGHBOUR_GAP_M, anchor.y)
    t_n0 = t_dep + tm.settle_s
    neighbour = Route(t_n0, behind.x - ENTRY_DISTANCE_M, behind.y)
    t_pull = neighbour.walk_to(behind.x, behind.y, WALK_SPEED_MS).stay(SLACK_S).t
    custody = config.custody
    step = config.registry.moved_threshold_m + QUEUE_SPEED_MS * (
        custody.carry_confirm_s + 2.0 / MIN_FPS + SLACK_S
    )
    start = owner.here
    t_end = owner.stay_until(t_pull).walk_to(start.x + step, start.y, QUEUE_SPEED_MS).t
    neighbour.walk_to(behind.x + step, behind.y, QUEUE_SPEED_MS)
    bag_path = [*bag_in, (t_pull, anchor.x, anchor.y)]
    bag_path += carried(owner.waypoints(), t_pull, t_end, offset)

    def checks(scn: Scenario, cfg: Config) -> tuple[Premissa, ...]:
        n, o = scn.actor(neighbour_id), scn.actor(owner_id)
        search = cfg.pipeline.owner_search_radius_m
        reach = entering(n, anchor, search, t_n0, t_pull)
        anchored = t_dep + cfg.custody.rest_confirm_s + 1.0 / MIN_FPS
        longest, _, _ = within_stats(n, o, cfg.party.proximity_m, 0.0, scn.duration_s)
        bag = scn.actor(bag_id)
        owner_reach = max(
            o.position(t).distance_to(bag.position(t))
            for t in sample_times(t_pull, t_end, PREMISE_STEP_S)
        )
        return (
            depositor(scn, bag_id, owner_id, cfg),
            rests(scn, bag_id, t_dep, t_n0, cfg),
            Premissa(
                "o vizinho só chega ao alcance depois de a mala virar âncora do dono",
                reach >= anchored,
                f"alcance em {reach:.1f} s, âncora em {anchored:.1f} s",
            ),
            nearest_while_displaced(
                scn, bag_id, neighbour_id, anchor, (t_pull, scn.duration_s), cfg
            ),
            Premissa(
                "o dono puxa a mala: fica a ≤ owner_search_radius_m dela o tempo todo",
                owner_reach <= search,
                f"distância máxima {owner_reach:.2f} m (raio {search:.2f} m)",
            ),
            Premissa(
                "o co-movimento de vizinho e dono cobre < late_join_extent_m: sem vínculo forte",
                step < cfg.party.late_join_extent_m,
                f"{step:.2f} m (limite {cfg.party.late_join_extent_m:.1f} m)",
            ),
            Premissa(
                "o vizinho fica a ≤ party.proximity_m do dono por < weak_bond_s: sem vínculo fraco",
                longest < cfg.party.weak_bond_s,
                f"{longest:.1f} s contínuos (limite {cfg.party.weak_bond_s:.1f} s)",
            ),
        )

    return Scenario(
        name="owner_pulls_neighbour_closer",
        descricao=(
            "Na fila, o dono puxa a própria mala enquanto o vizinho de trás está mais perto "
            "dela: o vizinho não pode ser acusado."
        ),
        invariant="P3",
        kind="ambiguo",
        roles=(("dono", owner_id), ("vizinho", neighbour_id), ("bagagem", bag_id)),
        expected=Expectation(
            never_accuse=frozenset({owner_id, neighbour_id}), due=(), silence=False
        ),
        actors=(
            person(owner_id, owner, 0.0, FOREVER),
            person(neighbour_id, neighbour, t_n0, FOREVER),
            suitcase(bag_id, bag_path, ((0.0, FOREVER),)),
        ),
        duration_s=t_end + tm.tail_s,
        bag_box_height=tm.bag_box_height,
        marks=(("deposito", t_dep), ("puxa", t_pull), ("ultima_acao", t_end)),
        checks=checks,
    )

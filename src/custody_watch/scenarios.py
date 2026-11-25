"""Catálogo determinístico de cenários de custódia, com gabarito por construção.

A lógica de custódia só consome `(x, y, t)` depois do `GroundPlane`, então
pode ser avaliada com trajetórias geradas. Cada cenário é uma cena em METROS,
renderizada no plano identidade pelo mesmo caminho público que o vídeo usa
(`TrackedDetection` → `run_session`/`LiveSession`), e carrega o desfecho
correto escrito antes de rodar: quem não pode ser acusado, quem deve ser
sinalizado e até quando, e se o certo é ficar calado.

A geometria deriva dos limiares da `Config` recebida -- raios, prazos,
cadência de fusão --, e `premissas(config)` mede na própria trajetória as
pré-condições de que o gabarito depende. Se um limiar muda e a cena deixa de
exercitar o que diz exercitar, falha a premissa, não o gabarito.

As cenas moram em três módulos, por invariante: `scenarios_base` (base e P4),
`scenarios_groups` (P1) e `scenarios_anchors` (P2 e P3). Este módulo é o
índice: a ordem aqui é a ordem dos relatórios.

Alcance: mede a LÓGICA com posições por construção. Não demonstra detector,
rastreador nem desempenho em vídeo real.
"""

from __future__ import annotations

from collections.abc import Callable

from .config import Config
from .scenario_model import MIN_FPS, Premissa, Scenario, identity_plane
from .scenarios_anchors import (
    new_track_between_occluded,
    owner_pulls_neighbour_closer,
    twin_bags_new_ids,
    twin_bags_swapped_ids,
)
from .scenarios_base import (
    detection_gap_near_bag,
    owner_returns,
    passerby_no_luggage,
    stranger_takes,
)
from .scenarios_groups import companion_takes, owner_leaves_weak_stays, sit_beside_then_take

SCENARIO_BUILDERS: tuple[Callable[[Config], Scenario], ...] = (
    owner_returns,
    companion_takes,
    stranger_takes,
    sit_beside_then_take,
    owner_leaves_weak_stays,
    twin_bags_new_ids,
    twin_bags_swapped_ids,
    new_track_between_occluded,
    owner_pulls_neighbour_closer,
    passerby_no_luggage,
    detection_gap_near_bag,
)


def build_scenarios(config: Config | None = None) -> tuple[Scenario, ...]:
    """Todos os cenários, construídos para `config` (default: `Config()`)."""
    config = config or Config()
    return tuple(build(config) for build in SCENARIO_BUILDERS)


__all__ = [
    "MIN_FPS",
    "Premissa",
    "SCENARIO_BUILDERS",
    "Scenario",
    "build_scenarios",
    "identity_plane",
]

"""Pontuação de um cenário contra o gabarito: pura, sem estado, sem vídeo.

## O que conta como acusação (o único lugar onde isso é decidido)

Lido em `flags.py`, `alerts.py` e `events.py`:

- **acusação** (`Category.ACCUSATION`): flag de nível `N3`. Hoje o único tipo
  é `retirada_por_estranho`, emitido junto com `BAG_REMOVED_BY_STRANGER` --
  é o que põe a pessoa no topo da fila (`alerts.build_queue`). Conta-se a
  flag e não o evento porque a flag é o que o operador vê, com a pessoa já
  no id canônico.
- **sinal brando de pessoa** (`Category.PERSON_SIGNAL`): flags `N1`
  (`permanencia_bagagem_alheia`, não entra na fila) e `N2`
  (`contato_bagagem_alheia`, entra na fila abaixo do N3). Descrevem relação,
  não acusam.
- **sinal brando de bagagem** (`Category.UNATTENDED`): evento
  `BAG_UNATTENDED`. É da bagagem, não aponta pessoa.
- **não é sinal**: `BAG_AMBIGUOUS` (incerteza declarada: P3 manda calar),
  `BAG_OCCLUDED`, `BAG_REATTENDED`, `BAG_REMOVED_BY_OWNER`, eventos de grupo e
  de religação. São registro de auditoria, não algo dirigido ao operador.

A baseline (`scenario_baseline`) já emite nestas categorias, sem tradução.

## Veredito

Em ordem de gravidade, o primeiro que se aplica:

1. `acusacao_indevida` -- acusação contra quem o gabarito protege ou que
   nenhum item devido justifica;
2. `sinal_indevido` -- o gabarito pede silêncio e saiu sinal brando. Não
   está na lista original de vereditos: sem ele, um N1 fabricado contra um
   passante teria de ser chamado de acusação (não é) ou de acerto (também
   não);
3. `perdeu` -- algum item devido não saiu dentro da janela;
4. `mudo_correto` -- nada saiu e nada era devido;
5. `correto` -- o resto: tudo que era devido saiu, nada indevido.

Itens cuja categoria está fora do `scope` não pesam no veredito; a baseline
não tem sinal de pessoa e é pontuada sem ele.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .events import Event, EventKind
from .flags import FlagStore
from .types import FlagLevel

TIME_EPSILON_S = 1e-9
"""Folga de ponto flutuante nas bordas da janela: `k / fps` raramente cai
exatamente sobre `referência + tolerância`."""


class Category(StrEnum):
    ACCUSATION = "acusacao"
    PERSON_SIGNAL = "sinal_pessoa"
    UNATTENDED = "desacompanhada"


ALL_CATEGORIES = frozenset(Category)
PERSON_CATEGORIES = frozenset({Category.ACCUSATION, Category.PERSON_SIGNAL})


class Verdict(StrEnum):
    CORRECT = "correto"
    UNDUE_ACCUSATION = "acusacao_indevida"
    UNDUE_SIGNAL = "sinal_indevido"
    MISSED = "perdeu"
    SILENT_CORRECT = "mudo_correto"


@dataclass(frozen=True)
class DueSignal:
    """Um sinal que o gabarito exige, com janela `[referência, referência+tolerância]`.

    `target` é o track da pessoa nas categorias de pessoa e o track original
    da bagagem em `UNATTENDED`. `reference_t` é o instante físico do ato
    (alguém pega a bagagem, o dono cruza o raio de acompanhamento), e a
    tolerância carrega o prazo que a própria config impõe mais folga de
    quantização; o atraso é medido a partir do ato.
    """

    category: Category
    target: int
    reference_t: float
    tolerance_s: float
    descricao: str


@dataclass(frozen=True)
class Expectation:
    """Gabarito por construção de um cenário."""

    never_accuse: frozenset[int]
    due: tuple[DueSignal, ...]
    silence: bool
    """O desfecho correto é não emitir nada: nem acusação, nem sinal brando."""


@dataclass(frozen=True)
class Emission:
    """Saída comparável de qualquer sistema: o que foi dito, sobre quem, quando."""

    category: Category
    t: float
    person: int | None
    bag: int | None
    source: str

    def target(self) -> int | None:
        return self.person if self.category in PERSON_CATEGORIES else self.bag


@dataclass(frozen=True)
class DueOutcome:
    due: DueSignal
    in_scope: bool
    emitted: bool
    delay_s: float | None
    """Atraso da primeira emissão casada a partir da referência, mesmo fora
    da janela -- quem lê precisa saber se perdeu por um quadro ou por nada."""


@dataclass(frozen=True)
class Score:
    verdict: Verdict
    undue_accused: tuple[int, ...]
    undue_signals: tuple[Emission, ...]
    due: tuple[DueOutcome, ...]
    silence: bool
    counts: dict[str, int]

    @property
    def all_due_emitted(self) -> bool:
        return all(o.emitted for o in self.due if o.in_scope)

    def to_dict(self) -> dict[str, Any]:
        return {
            "veredito": self.verdict.value,
            "acusacao_indevida": {
                "ocorreu": bool(self.undue_accused),
                "quem": list(self.undue_accused),
            },
            "sinal_indevido": {
                "ocorreu": bool(self.undue_signals),
                "sinais": [_emission_dict(e) for e in self.undue_signals],
            },
            "evento_devido_emitido": {
                "todos": self.all_due_emitted,
                "itens": [_outcome_dict(o) for o in self.due],
            },
            "silencio": self.silence,
            "emissoes": dict(self.counts),
        }


def _emission_dict(e: Emission) -> dict[str, Any]:
    return {
        "categoria": e.category.value,
        "t": e.t,
        "pessoa": e.person,
        "bagagem": e.bag,
        "origem": e.source,
    }


def _outcome_dict(o: DueOutcome) -> dict[str, Any]:
    return {
        "categoria": o.due.category.value,
        "alvo": o.due.target,
        "referencia_t": o.due.reference_t,
        "tolerancia_s": o.due.tolerance_s,
        "descricao": o.due.descricao,
        "no_escopo": o.in_scope,
        "emitido": o.emitted,
        "atraso_s": o.delay_s,
    }


def _due_outcome(due: DueSignal, emissions: Sequence[Emission], in_scope: bool) -> DueOutcome:
    matches = [
        e
        for e in emissions
        if e.category is due.category
        and e.target() == due.target
        and e.t >= due.reference_t - TIME_EPSILON_S
    ]
    if not in_scope or not matches:
        return DueOutcome(due=due, in_scope=in_scope, emitted=False, delay_s=None)
    first = min(matches, key=lambda e: e.t)
    delay = first.t - due.reference_t
    return DueOutcome(
        due=due,
        in_scope=True,
        emitted=delay <= due.tolerance_s + TIME_EPSILON_S,
        delay_s=delay,
    )


def score(
    expected: Expectation,
    emissions: Iterable[Emission],
    scope: frozenset[Category] = ALL_CATEGORIES,
) -> Score:
    """Compara as emissões com o gabarito. Ver o docstring do módulo."""
    in_scope = sorted(
        (e for e in emissions if e.category in scope), key=lambda e: (e.t, e.category)
    )
    justified = {
        d.target for d in expected.due if d.category is Category.ACCUSATION and d.category in scope
    }
    undue_accused = tuple(
        sorted(
            {
                e.person
                for e in in_scope
                if e.category is Category.ACCUSATION
                and e.person is not None
                and (e.person in expected.never_accuse or e.person not in justified)
            }
        )
    )
    undue_signals = (
        tuple(e for e in in_scope if e.category is not Category.ACCUSATION)
        if expected.silence
        else ()
    )
    outcomes = tuple(_due_outcome(d, in_scope, d.category in scope) for d in expected.due)
    silence = not in_scope
    counts = {c.value: sum(1 for e in in_scope if e.category is c) for c in Category if c in scope}

    if undue_accused:
        verdict = Verdict.UNDUE_ACCUSATION
    elif undue_signals:
        verdict = Verdict.UNDUE_SIGNAL
    elif not all(o.emitted for o in outcomes if o.in_scope):
        verdict = Verdict.MISSED
    elif silence:
        verdict = Verdict.SILENT_CORRECT
    else:
        verdict = Verdict.CORRECT

    return Score(
        verdict=verdict,
        undue_accused=undue_accused,
        undue_signals=undue_signals,
        due=outcomes,
        silence=silence,
        counts=counts,
    )


def classify_flag_level(level: FlagLevel) -> Category:
    """N3 é acusação; N1 e N2 são sinal brando de pessoa."""
    return Category.ACCUSATION if level >= FlagLevel.N3 else Category.PERSON_SIGNAL


SIGNAL_EVENT_KINDS: dict[EventKind, Category] = {EventKind.BAG_UNATTENDED: Category.UNATTENDED}
"""Eventos que são sinal. Os demais são registro de auditoria."""


def system_emissions(flags: FlagStore, events: Iterable[Event]) -> tuple[Emission, ...]:
    """Traduz a saída do sistema (flags e eventos) em emissões comparáveis."""
    out: list[Emission] = []
    for person in flags.people():
        for flag in flags.for_person(person):
            out.append(
                Emission(
                    category=classify_flag_level(flag.level),
                    t=flag.t,
                    person=person,
                    bag=flag.bag,
                    source=f"flag:{flag.kind}",
                )
            )
    for event in events:
        category = SIGNAL_EVENT_KINDS.get(event.kind)
        if category is None:
            continue
        out.append(
            Emission(
                category=category,
                t=event.t_start,
                person=None,
                bag=event.bag,
                source=f"evento:{event.kind.value}",
            )
        )
    return tuple(sorted(out, key=lambda e: (e.t, e.category, e.source)))


__all__ = [
    "ALL_CATEGORIES",
    "Category",
    "DueOutcome",
    "DueSignal",
    "Emission",
    "Expectation",
    "Score",
    "Verdict",
    "classify_flag_level",
    "score",
    "system_emissions",
]

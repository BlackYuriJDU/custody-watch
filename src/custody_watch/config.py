"""Configuração com guarda-corpo (spec §3).

Tornar limiares ajustáveis por JSON reabre todos os ataques que três rodadas
de revisão adversarial fecharam. `late_join_extent_m: 0.5` num arquivo de
config devolve a posse da bagagem a dois cúmplices, silenciosamente, sem
quebrar teste nenhum — porque os testes usam os defaults.

Daí `SAFE_BOUNDS`: cada limiar perigoso tem faixa e uma razão escrita. Sair
da faixa exige `unsafe_override` com justificativa, e a mensagem de erro cita
o ataque que o limite previne. Erro de config é documentação.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

MIN_JUSTIFICATION_CHARS = 15


class UnsafeValueError(ValueError):
    """Valor fora da faixa segura, sem override justificado."""


@dataclass(frozen=True)
class Bounds:
    minimum: float
    maximum: float
    reason: str


@dataclass(frozen=True)
class PartyConfig:
    proximity_m: float = 2.0
    late_join_extent_m: float = 5.0
    min_extent_m: float = 0.5
    min_overlap_samples: int = 3
    min_overlap_s: float = 2.0
    max_gap_s: float = 2.0
    time_tolerance_s: float = 0.5
    weak_bond_s: float = 60.0


@dataclass(frozen=True)
class CustodyConfig:
    unattended_distance_m: float = 3.0
    unattended_time_s: float = 25.0
    max_occlusion_s: float = 30.0
    carry_confirm_s: float = 1.0
    rest_confirm_s: float = 2.0
    ownership_window_s: float = 10.0
    min_occlusion_report_s: float = 1.0


@dataclass(frozen=True)
class RegistryConfig:
    moved_threshold_m: float = 0.5
    ambiguity_radius_m: float = 1.0
    reidentification_radius_m: float = 1.0


@dataclass(frozen=True)
class FlagConfig:
    tau_s: float = 900.0
    weight_n1: float = 1.0
    weight_n2: float = 3.0
    weight_n3: float = 10.0


@dataclass(frozen=True)
class PipelineConfig:
    """Números do orquestrador. Antes eram constantes soltas no módulo."""

    owner_search_radius_m: float = 3.0
    missing_before_occluded_s: float = 0.18
    """Tempo sem observação até a bagagem contar como ocluída.

    Substitui `missing_frames_before_occluded`, 5 quadros: 0,2 s a 25 fps e
    1,67 s a 3 fps, a mesma cena esperando oito vezes mais conforme a fonte.
    A 25 fps, a taxa do CAVIAR em que os 5 quadros foram ajustados, qualquer
    valor em (4/25, 5/25] = (0,16; 0,20] s decide no mesmo quinto quadro de
    antes. 0,18 é o meio do intervalo: soma binária de 0,04 s e jitter de
    timestamp de até 20 ms não mudam o quadro da decisão.

    Vale junto com o piso de quadros seguidos sem a bagagem do orquestrador
    (`orchestrator.MIN_MISSING_FRAMES`, 2): falha de detector é evento por
    quadro, e a 3 fps uma só já dura 0,33 s. Com os dois, a oclusão abre em
    0,2 s a 25 e a 10 fps, e em 0,67 s (dois quadros) a 3 fps.
    """
    merge_every_s: float = 0.98
    """Intervalo entre duas fusões de grupos, em segundos.

    A fusão não é manutenção: é nela que co-movimento vira vínculo forte (quem
    conta como dono, regra P1), que o cronômetro do vínculo fraco começa e
    conclui, e que um afastamento o zera. Roda no primeiro quadro a pelo
    menos este tempo da fusão anterior -- ou do primeiro quadro da sessão,
    para a primeira --, então nunca mais que um quadro depois de vencer.

    Substitui `merge_every_frames`, 25 quadros: 1 s a 25 fps e 8,33 s a 3 fps,
    a mesma cena formando o grupo em tempos reais diferentes conforme a fonte.
    A 25 fps, a taxa do CAVIAR em que os 25 quadros foram ajustados, qualquer
    valor em (24/25, 25/25] = (0,96; 1,00] s funde a cada 25 quadros, como
    antes. 0,98 é o meio do intervalo: soma binária de k/25 e jitter de
    timestamp de até 20 ms não mudam o quadro da fusão. A 10 e a 3 fps, o
    mesmo 0,98 dá 10 e 3 quadros: 1 s nas três taxas.

    A fase anda um quadro a 25 fps: a contagem antiga começava em 1, e a
    primeira fusão vinha no 25º quadro, 0,96 s depois do primeiro; agora
    toda fusão, a primeira inclusive, espera 0,98 s e cai no 26º. O período
    segue 25 quadros.
    """
    history_window_s: float = 10.0
    contact_radius_m: float = 0.8
    proximity_radius_m: float = 2.0
    proximity_flag_s: float = 90.0
    max_observation_speed_ms: float = 25.0
    min_bag_height_px: float = 40.0


@dataclass(frozen=True)
class ReidConfig:
    min_similarity: float = 0.90
    min_margin: float = 0.03
    max_gap_s: float = 30.0
    max_speed_ms: float = 3.0
    position_slack_m: float = 2.0
    min_samples: int = 10
    max_samples: int = 200
    max_settle_s: float = 0.5


@dataclass(frozen=True)
class AlertConfig:
    clip_margin_s: float = 10.0
    operator_hourly_budget: int = 25


@dataclass(frozen=True)
class Config:
    party: PartyConfig = PartyConfig()
    custody: CustodyConfig = CustodyConfig()
    registry: RegistryConfig = RegistryConfig()
    flags: FlagConfig = FlagConfig()
    alerts: AlertConfig = AlertConfig()
    reid: ReidConfig = ReidConfig()
    pipeline: PipelineConfig = PipelineConfig()


SAFE_BOUNDS: dict[str, Bounds] = {
    "party.proximity_m": Bounds(
        1.0,
        3.0,
        "acima de 3m, pessoas em mesas vizinhas da praça de alimentação são "
        "tratadas como o mesmo grupo",
    ),
    "party.late_join_extent_m": Bounds(
        3.0,
        15.0,
        "abaixo de 3m, dois cúmplices obtêm posse perambulando ao lado da "
        "vítima sem ir a lugar algum",
    ),
    "party.min_extent_m": Bounds(
        0.3,
        2.0,
        "abaixo de 0.3m o ruído do detector satisfaz a guarda, e duas pessoas "
        "sentadas passam a co-mover",
    ),
    "party.time_tolerance_s": Bounds(
        0.1,
        2.0,
        "acima de 2s, observações de instantes não relacionados são casadas, "
        "e seguir alguém a oito metros vira andar junto",
    ),
    "party.min_overlap_s": Bounds(
        1.0,
        30.0,
        "abaixo de 1s, um cruzamento momentâneo conta como co-movimento sustentado",
    ),
    "party.max_gap_s": Bounds(
        0.5,
        10.0,
        "acima de 10s, três fotografias esparsas contam como sobreposição contínua",
    ),
    "custody.unattended_distance_m": Bounds(
        1.0,
        10.0,
        "acima de 10m o dono nunca é considerado ausente; abaixo de 1m qualquer "
        "passo dele dispara desacompanhamento",
    ),
    "custody.unattended_time_s": Bounds(
        5.0,
        300.0,
        "abaixo de 5s, ir ao balcão vira bagagem desacompanhada; acima de 300s "
        "o furto termina antes do alerta",
    ),
    "custody.max_occlusion_s": Bounds(
        5.0,
        120.0,
        "abaixo de 5s, alguém parado em frente à bagagem apaga a custódia e o "
        "falso alarme de retirada volta; acima de 120s o ladrão já saiu do "
        "prédio antes de o evento existir",
    ),
    "custody.carry_confirm_s": Bounds(
        0.2,
        10.0,
        "abaixo de 0.2s um único salto de projeção declara furto contra quem "
        "estiver por perto — é o defeito que este projeto fechou na porta do "
        "desaparecimento e reabriria na do movimento; acima de 10s o ladrão "
        "sai de quadro antes de o evento existir",
    ),
    "custody.min_occlusion_report_s": Bounds(
        0.0,
        10.0,
        "acima de 10s a oclusão de verdade some do log, e é ela que diz em que "
        "intervalo a custódia ficou sem observação -- quem revisa o log deixa "
        "de conseguir reconstruir o que o sistema não viu. Zero é permitido "
        "de propósito: significa registrar tudo, que é ruído de auditoria e "
        "não risco",
    ),
    "custody.rest_confirm_s": Bounds(
        0.5,
        30.0,
        "abaixo de 0.5s a bagagem 'para' entre dois passos de quem a puxa, e "
        "trânsito vira depósito: o viajante que entra em quadro com a própria "
        "mala ganha uma custódia que nunca existiu, e perdê-la é acusá-lo; "
        "acima de 30s um depósito real leva meio minuto para entrar no "
        "registro, e o furto por abandono termina antes disso",
    ),
    "custody.ownership_window_s": Bounds(
        1.0,
        60.0,
        "acima de 60s um estranho que se aproxima de uma bagagem sem dono vira "
        "dono legítimo por proximidade -- é o ataque que a regra P1 fecha, e a "
        "janela é o que impede a retentativa de posse de reabri-lo; abaixo de "
        "1s a retentativa não cobre nem uma rajada de falha do detector, que é "
        "o defeito que ela existe para consertar",
    ),
    "registry.reidentification_radius_m": Bounds(
        0.3,
        3.0,
        "abaixo de 0.3m a bagagem que volta empurrada por quem passou não é "
        "readotada e se registra como bagagem nova, enquanto a original segue "
        "ocluída -- duas entradas para uma bagagem física, e a identidade que "
        "a readoção existe para preservar se perde; acima de 3m duas bagagens "
        "distintas no mesmo balcão viram a mesma",
    ),
    "registry.moved_threshold_m": Bounds(
        0.2,
        2.0,
        "abaixo de 0.2m o jitter do detector é lido como a bagagem sendo carregada",
    ),
    "party.weak_bond_s": Bounds(
        10.0,
        600.0,
        "abaixo de 10s qualquer passagem casual vira vinculo fraco, e o flag de "
        "quem so passou perto e atenuado sem motivo",
    ),
    "pipeline.max_observation_speed_ms": Bounds(
        12.0,
        100.0,
        "abaixo de 12 m/s o portao comeca a cortar movimento real -- o p99.9 "
        "medido no CAVIAR e 12.7 m/s; acima de 100 m/s ele deixa passar o "
        "artefato de projecao correlacionado que produz extensao falsa",
    ),
    "pipeline.contact_radius_m": Bounds(
        0.3,
        2.0,
        "acima de 2m 'contato' passa a incluir quem so andou perto, e o flag N2 "
        "deixa de significar que alguem tocou na bagagem",
    ),
    "pipeline.owner_search_radius_m": Bounds(
        0.5,
        5.0,
        "acima de 5m o back-tracing atribui a bagagem a quem so passava pela "
        "regiao, e o dono errado torna toda a cadeia de custodia invalida",
    ),
    "pipeline.min_bag_height_px": Bounds(
        0.0,
        200.0,
        "acima de 200px nenhuma bagagem real de uma câmera de teto passa, e o "
        "sistema fica mudo sem avisar; o valor padrão, 40px, é o que uma mala de "
        "55cm ocupa a 73 px/m, a exigência medida contra o CAVIAR. Zero desliga "
        "o piso, o que só é honesto para caixa anotada à mão",
    ),
    "pipeline.missing_before_occluded_s": Bounds(
        0.08,
        2.0,
        "abaixo de 0.08s (dois quadros a 25 fps), toda rajada curta do detector "
        "numa fonte rápida abre um intervalo de oclusão -- o piso de dois "
        "quadros seguidos do orquestrador só barra a falha isolada --, e a "
        "lista de candidatos acumula quem apenas passou por perto até a "
        "atribuição não apontar mais ninguém em particular; acima de 2s "
        "(cinquenta quadros a 25 fps) a bagagem segue 'visível' na lógica por "
        "segundos depois de sumir da tela, e a janela de candidatos perde "
        "justamente quem estava em cima dela",
    ),
    "pipeline.merge_every_s": Bounds(
        0.0,
        5.0,
        "acima de 5s, quem andou junto só o mínimo (late_join_extent_m, 5 m, uns "
        "4 s a passo de pedestre) tem essa caminhada saindo da janela de "
        "histórico (history_window_s, 10 s) antes da fusão seguinte, e o "
        "acompanhante que chegou com o dono fica sem vínculo forte: levar a mala "
        "do grupo vira acusação (regra P1). É o defeito dos 25 quadros a 3 fps, "
        "8,3 s. E afastamentos mais curtos que a cadência não zeram o cronômetro "
        "do vínculo fraco, e quem saiu e voltou a sentar perto vira vínculo fraco "
        "do grupo do dono sem ter ficado weak_bond_s seguidos. Zero é permitido: "
        "funde a cada quadro, o que custa CPU e não segurança",
    ),
    "reid.min_similarity": Bounds(
        0.70,
        0.999,
        "e apenas um piso de sanidade, nao o discriminador: medido no CAVIAR, a "
        "similaridade de uma religacao correta varia de 0.92 a 0.99 conforme o "
        "tamanho da amostra, entao um piso alto rejeita religacao correta de "
        "perfil jovem. Quem separa e a margem",
    ),
    "reid.min_margin": Bounds(
        0.01,
        0.2,
        "esta e a guarda que de fato separa. O fundo da cena domina o histograma "
        "em recortes pequenos, entao toda similaridade fica alta e o que "
        "discrimina e a distancia para o segundo colocado. Zero religa quase "
        "qualquer par, e um ladrao herda o grupo da vitima -- o furto fica mudo",
    ),
    "reid.max_gap_s": Bounds(
        2.0,
        300.0,
        "acima de 300s a pessoa teve tempo de trocar de roupa ou sair do predio; "
        "abaixo de 2s a fragmentacao comum de tracker nao e alcancada",
    ),
    "reid.max_speed_ms": Bounds(
        0.5,
        10.0,
        "acima de 10 m/s qualquer reaparicao na cena e considerada plausivel, "
        "o que desliga a guarda fisica",
    ),
    "reid.max_settle_s": Bounds(
        0.2,
        1.0,
        "acima de 1s, numa fonte de poucos quadros por segundo quem carrega a "
        "bagagem fica fora de 'people' alem de carry_confirm_s (1s por padrao): "
        "a bagagem parece andar sozinha, o furto vira bag_ambiguous e some da "
        "fila; abaixo de 0.2s o prazo deixa de ser rede para fonte lenta e passa "
        "a decidir tambem a 25 fps, encurtando o perfil para perto das cinco "
        "amostras a partir das quais a margem foi medida estavel -- margem "
        "ruidosa religa o ladrao ao grupo da vitima, e o furto fica mudo",
    ),
    "flags.tau_s": Bounds(
        60.0,
        7200.0,
        "acima de 7200s o decaimento não decai, e tempo de permanência no "
        "aeroporto vira suspeição acumulada",
    ),
}

RETIRED_FIELDS: dict[str, str] = {
    "pipeline.missing_frames_before_occluded": (
        "contava quadros, e a mesma cena esperava durações reais diferentes "
        "conforme o fps da fonte. Use pipeline.missing_before_occluded_s, em "
        "segundos: N quadros valem N / fps da fonte em que o valor foi "
        "ajustado. Não há conversão automática porque a config não conhece o "
        "fps, e supor um reabriria o defeito em silêncio. O default novo, 0.18, "
        "decide no mesmo quinto quadro de antes a 25 fps"
    ),
    "pipeline.merge_every_frames": (
        "contava quadros, e a mesma cena formava grupos e vínculos em durações "
        "reais diferentes conforme o fps da fonte: 25 quadros eram 1 s a 25 fps "
        "e 8,3 s a 3 fps. Use pipeline.merge_every_s, em segundos: N quadros "
        "valem N / fps da fonte em que o valor foi ajustado. Não há conversão "
        "automática porque a config não conhece o fps, e supor um reabriria o "
        "defeito em silêncio. O default novo, 0.98, funde a cada 25 quadros a "
        "25 fps, como antes"
    ),
}
"""Campos removidos e o que pôr no lugar. Continuam recusados como qualquer
campo desconhecido, mas com a mensagem de migração em vez da lista seca."""

_SECTIONS = {field.name: field.type for field in fields(Config)}


def _resolve(key: str, raw: Any) -> float | int:
    """Extrai o valor e valida contra a faixa segura, se houver."""
    override: str | None = None
    if isinstance(raw, dict):
        if "value" not in raw:
            raise ValueError(f"{key}: objeto de config precisa da chave 'value'")
        desconhecidas = set(raw) - {"value", "unsafe_override"}
        if desconhecidas:
            raise ValueError(f"{key}: campo desconhecido {sorted(desconhecidas)}")
        override = raw.get("unsafe_override")
        value = raw["value"]
    else:
        value = raw

    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key}: valor deve ser numérico, recebido {value!r}")

    bounds = SAFE_BOUNDS.get(key)
    if bounds is None or bounds.minimum <= value <= bounds.maximum:
        return value

    if override is None:
        raise UnsafeValueError(
            f"{key} = {value} está fora da faixa segura "
            f"[{bounds.minimum}, {bounds.maximum}].\n"
            f"Motivo do limite: {bounds.reason}.\n"
            f"Para assumir o risco, use "
            f'{{"value": {value}, "unsafe_override": "<justificativa>"}}.'
        )

    if len(override.strip()) < MIN_JUSTIFICATION_CHARS:
        raise UnsafeValueError(
            f"{key} = {value}: unsafe_override exige justificativa de ao menos "
            f"{MIN_JUSTIFICATION_CHARS} caracteres. O campo existe para deixar "
            f"rastro de quem afrouxou o limite e por quê."
        )

    return value


def _unknown_field_error(
    path: Path, section_name: str, field_name: str, known: set[str]
) -> ValueError:
    """Campo retirado diz o que pôr no lugar; o resto lista os esperados."""
    key = f"{section_name}.{field_name}"
    retired = RETIRED_FIELDS.get(key)
    if retired is not None:
        return ValueError(f"{path}: campo retirado {key!r}: {retired}")
    return ValueError(
        f"{path}: campo desconhecido {section_name}.{field_name!r}; esperados {sorted(known)}"
    )


def load_config(path: Path | str) -> Config:
    """Carrega config de JSON, mesclando com os defaults.

    Seção ou campo desconhecido é erro, não aviso: um typo silencioso vira
    config ignorada, que vira comportamento surpresa em produção.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"{path}: raiz do JSON deve ser um objeto")

    config = Config()
    for section_name, section_data in data.items():
        if section_name not in _SECTIONS:
            raise ValueError(
                f"{path}: seção desconhecida {section_name!r}; esperadas {sorted(_SECTIONS)}"
            )
        if not isinstance(section_data, dict):
            raise ValueError(f"{path}: seção {section_name!r} deve ser um objeto")

        current = getattr(config, section_name)
        known = {field.name for field in fields(current)}
        updates: dict[str, float | int] = {}

        for field_name, raw in section_data.items():
            if field_name not in known:
                raise _unknown_field_error(path, section_name, field_name, known)
            updates[field_name] = _resolve(f"{section_name}.{field_name}", raw)

        config = replace(config, **{section_name: replace(current, **updates)})

    return config

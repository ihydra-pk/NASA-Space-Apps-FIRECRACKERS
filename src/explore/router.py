"""Route a plain-language question, extract its test conditions, plan searches.

Without a router the caller must pass ``conditions`` as a separate structured
object. The router reads the question once with the LLM, decides whether the
structured test data is relevant, returns any conditions the question states
so that the existing matcher can run on them, and writes a few focused search
queries so that document retrieval also finds evidence the question's own
wording misses (mechanisms, conditions, experiment names, acronyms).

The LLM's output is treated as untrusted: unknown keys, wrong types, unknown
units and numbers that do not appear in the question are dropped and reported;
search queries are limited in number and length. If the router cannot be used
the question falls back to a plain document search.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from src.experimental_data.matching import CATEGORICAL_FIELDS, NUMERIC_FIELDS
from src.llm.config import LLMConfig
from src.llm.providers import LLMProvider
from src.llm.service import LLMResponseError, _parse_json

ROUTES = ("documents", "test_data", "both")

# Units ``normalize_conditions`` can convert; anything else would be misread.
_ALLOWED_UNITS = {
    "oxygen_concentration": {"%", "mole_fraction"},
    "pressure": {"atm", "kPa", "Pa"},
    "flow_velocity": {"cm/s", "m/s"},
    "suppressant_concentration": {"mole_fraction", "%"},
}

SYSTEM_PROMPT = """You plan the searches of a NASA microgravity combustion assistant.

Two sources exist:
1. documents: NASA reports, science requirement documents and presentations for
   combustion experiments (FLEX, CFI, BASS, SAFFIRE, ACME, SPICE, SAME and
   others). Use them for explanations, objectives, methods and findings.
2. test_data: a table of NASA test conditions: FLEX droplet tests (a planned
   test matrix and observed runs), BASS-II solid-fuel tests (observed) and
   SAFFIRE runs. Each row has fuel, oxygen_concentration, pressure,
   flow_velocity, suppressant, suppressant_concentration, diluent and
   fiber_support. Observed rows hold measured outcomes; planned rows are only
   proposed conditions. It contains no temperatures or gravity levels.

Reply with one JSON object:
{"route": "documents" | "test_data" | "both",
 "conditions": {},
 "condition_units": {},
 "search_queries": [],
 "reason": "one short sentence"}

Routing:
- "test_data": the question asks which or how many tests have stated conditions, or what they measured.
- "both": the question states test conditions and also asks for an explanation.
- "documents": everything else. Use this whenever the question states no test condition.

Conditions:
- Include only values the question states explicitly. Never guess, infer or add defaults.
- Allowed keys: fuel, oxygen_concentration, pressure, flow_velocity, suppressant,
  suppressant_concentration, diluent, fiber_support.
- fuel is "heptane" or "methanol" for droplets (C7H16 is heptane, CH3OH is
  methanol); for a solid sample copy the material as written, e.g. "PMMA" or "SIBAL".
  suppressant is "CO2", "He" or "SF6". diluent is "N2". fiber_support is "SiC" or "none".
- Numbers: copy the number exactly as written in the question and put its unit in
  condition_units under the same key. Do not convert units.
  Allowed units: oxygen_concentration "%" or "mole_fraction"; pressure "atm", "kPa" or "Pa";
  flow_velocity "cm/s" or "m/s"; suppressant_concentration "mole_fraction" or "%".

Search queries:
- Always write 2 to 4 short search queries that would find the NASA document
  passages needed to answer the question, whatever the route.
- Each targets a different aspect: the physical mechanism, the variables or
  conditions involved, the experiments that studied it, or other terms and
  acronyms the documents may use (e.g. "cool flame" and "low-temperature
  chemistry", "NTC").
- Write them as phrases a report would contain, not as questions, and do not
  repeat the question.
"""
# Bounds on planner output, which is untrusted like everything the model writes.
_MAX_SEARCH_QUERIES = 4
_MAX_QUERY_CHARS = 200


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: str
    conditions: dict[str, object] = field(default_factory=dict)
    condition_units: dict[str, str] = field(default_factory=dict)
    reason: str = ""
    ignored: tuple[str, ...] = ()
    fallback: bool = False
    # Extra document searches; the question itself is always searched too.
    search_queries: tuple[str, ...] = ()

    @property
    def uses_test_data(self) -> bool:
        return self.route != "documents" and bool(self.conditions)

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "conditions": dict(self.conditions),
            "condition_units": dict(self.condition_units),
            "search_queries": list(self.search_queries),
            "reason": self.reason,
            "ignored": list(self.ignored),
            "fallback": self.fallback,
        }


def _numbers_in(text: str) -> set[float]:
    return {float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)}


def _search_queries(raw: object, question: str) -> tuple[str, ...]:
    """Up to four distinct, non-empty search queries that differ from the question."""

    if not isinstance(raw, list):
        return ()
    seen = {" ".join(question.split()).casefold()}
    kept: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())
        if not text or len(text) > _MAX_QUERY_CHARS or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        kept.append(text)
        if len(kept) == _MAX_SEARCH_QUERIES:
            break
    return tuple(kept)


def _fallback(reason: str) -> RouteDecision:
    return RouteDecision("documents", reason=reason, fallback=True)


def validate_route(parsed: dict[str, object], question: str) -> RouteDecision:
    """Keep only what the matcher supports and the question actually states."""

    route = parsed.get("route")
    if route not in ROUTES:
        return _fallback(f"Router returned an unknown route: {route!r}")
    raw_conditions = parsed.get("conditions") or {}
    raw_units = parsed.get("condition_units") or {}
    if not isinstance(raw_conditions, dict) or not isinstance(raw_units, dict):
        return _fallback("Router returned conditions or units that are not objects")

    stated_numbers = _numbers_in(question)
    conditions: dict[str, object] = {}
    units: dict[str, str] = {}
    ignored: list[str] = []
    for key, value in raw_conditions.items():
        if key in CATEGORICAL_FIELDS:
            if isinstance(value, str) and value.strip():
                conditions[key] = value.strip()
            else:
                ignored.append(f"{key}: not a text value")
        elif key in NUMERIC_FIELDS:
            if isinstance(value, str):  # models sometimes quote numbers
                try:
                    value = float(value)
                except ValueError:
                    pass
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                ignored.append(f"{key}: not a number")
            elif float(value) not in stated_numbers:
                ignored.append(f"{key}: {value} is not a number written in the question")
            elif key in raw_units and raw_units[key] not in _ALLOWED_UNITS[key]:
                ignored.append(f"{key}: unsupported unit {raw_units[key]!r}")
            else:
                conditions[key] = value
                if key in raw_units:
                    units[key] = raw_units[key]
        else:
            ignored.append(f"{key}: not a field in the test data")

    reason = parsed.get("reason")
    if not conditions:
        route = "documents"
    return RouteDecision(
        route=route,
        conditions=conditions,
        condition_units=units,
        reason=reason.strip() if isinstance(reason, str) else "",
        ignored=tuple(ignored),
        search_queries=_search_queries(parsed.get("search_queries"), question),
    )


class QuestionRouter:
    """One LLM call per question; never raises, falls back to documents."""

    def __init__(self, provider: LLMProvider, config: LLMConfig) -> None:
        self.provider = provider
        self.config = config

    def route(self, question: str) -> RouteDecision:
        try:
            raw = self.provider.generate(
                system_prompt=SYSTEM_PROMPT,
                # JSON mode is rejected unless the user message mentions JSON.
                user_prompt=f"Question: {question}\n\nReply with the JSON object described above.",
                config=self.config,
            )
            return validate_route(_parse_json(raw), question)
        except (LLMResponseError, RuntimeError) as error:
            return _fallback(f"Router unavailable: {error}")

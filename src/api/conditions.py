"""Deterministic extraction of structured conditions from a chat question.

Only values explicitly written in the question are extracted; nothing is guessed.
"""

from __future__ import annotations

import re

MMHG_PER_ATM = 760.0

_FUELS = {
    "methanol": "methanol",
    "heptane": "heptane",
    "silicone": "silicone",
    "pmma": "pmma",
    "sibal": "sibal",
    "nomex": "nomex",
}
_NUM = r"(\d+(?:\.\d+)?)"


def extract_conditions(question: str) -> dict[str, object]:
    text = question.lower()
    conditions: dict[str, object] = {}

    for word, fuel in _FUELS.items():
        if re.search(rf"\b{word}\b", text):
            conditions["fuel"] = fuel
            break

    oxygen = re.search(rf"{_NUM}\s*(?:%|percent)\s*(?:oxygen|o2|o₂)", text) or re.search(
        rf"(?:oxygen|o2)\s*(?:of|at|=)?\s*{_NUM}\s*(?:%|percent)", text
    )
    if oxygen:
        conditions["oxygen_concentration"] = float(oxygen.group(1))

    mmhg = re.search(rf"{_NUM}\s*mm\s*hg", text)
    atm = re.search(rf"{_NUM}\s*atm\b", text)
    if mmhg:
        conditions["pressure"] = round(float(mmhg.group(1)) / MMHG_PER_ATM, 4)
    elif atm:
        conditions["pressure"] = float(atm.group(1))

    flow = re.search(rf"{_NUM}\s*cm\s*/\s*s", text)
    if flow:
        conditions["flow_velocity"] = float(flow.group(1))
    return conditions

"""Conservative normalization for FLEX test-plan values."""

from __future__ import annotations

import re


def normalize_fuel(value: str | None) -> str | None:
    """Map formulas to unambiguous common names while retaining source text elsewhere."""

    if value is None:
        return None
    compact = re.sub(r"\s+", "", value).upper()
    return {"C7H16": "heptane", "CH3OH": "methanol"}.get(compact, value.strip())


def normalize_categorical(value: str | None) -> str | None:
    return " ".join(value.split()).casefold() if value and value.strip() else None


def normalize_test_number(value: str) -> str:
    return value.strip()


def units_for_test_plan(*, has_suppressant_concentration: bool) -> dict[str, str | None]:
    """Units explicitly shown or directly defined by the PDF's notation."""

    units: dict[str, str | None] = {
        "oxygen_concentration": "mole_fraction",
        "pressure": "atm",
        "flow_velocity": "cm/s",
        "suppressant_concentration": "mole_fraction" if has_suppressant_concentration else None,
    }
    return units

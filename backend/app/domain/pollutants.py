"""Pollutant identities, canonical units and documented unit conversions (PRD §9 normalize_units).

Canonical unit for every pollutant is µg/m³, the unit CPCB uses for the Indian NAAQS.
Gas conversions from ppb use reference conditions of 25 °C and 1 atm, where one mole of
ideal gas occupies 24.45 L:

    µg/m³ = ppb × molecular_weight / 24.45

Any (pollutant, unit) pair not listed here is rejected rather than guessed.
"""

from enum import StrEnum

CANONICAL_UNIT = "µg/m³"
MOLAR_VOLUME_L_25C = 24.45


class Pollutant(StrEnum):
    PM25 = "pm25"
    PM10 = "pm10"
    NO2 = "no2"
    O3 = "o3"
    SO2 = "so2"
    CO = "co"


MOLECULAR_WEIGHT_G_MOL: dict[Pollutant, float] = {
    Pollutant.NO2: 46.0055,
    Pollutant.O3: 47.9982,
    Pollutant.SO2: 64.066,
    Pollutant.CO: 28.010,
}

# Upper bound of physically plausible hourly means in µg/m³. Values above are kept but flagged
# `implausible` and excluded from training by default; they are never silently deleted.
PLAUSIBLE_MAX_UG_M3: dict[Pollutant, float] = {
    Pollutant.PM25: 1000.0,
    Pollutant.PM10: 2000.0,
    Pollutant.NO2: 1000.0,
    Pollutant.O3: 1000.0,
    Pollutant.SO2: 2000.0,
    Pollutant.CO: 50000.0,
}

# Unit spellings seen from sources, mapped to a normalized key.
_UNIT_ALIASES = {
    "µg/m³": "ug/m3",
    "μg/m³": "ug/m3",  # Greek mu (U+03BC) vs micro sign (U+00B5)
    "ug/m3": "ug/m3",
    "µg/m3": "ug/m3",
    "ppb": "ppb",
    "ppm": "ppm",
    "mg/m³": "mg/m3",
    "mg/m3": "mg/m3",
}

_POLLUTANT_ALIASES = {
    "pm25": Pollutant.PM25,
    "pm2.5": Pollutant.PM25,
    "pm10": Pollutant.PM10,
    "no2": Pollutant.NO2,
    "o3": Pollutant.O3,
    "ozone": Pollutant.O3,
    "so2": Pollutant.SO2,
    "co": Pollutant.CO,
}


class UnitConversionError(ValueError):
    pass


def parse_pollutant(name: str | None) -> Pollutant | None:
    """Map a source's pollutant name to the canonical enum; None if it's not one we model."""
    if not name:
        return None
    return _POLLUTANT_ALIASES.get(name.strip().lower().replace("_", "").replace(" ", ""))


def normalize_unit_token(unit: str | None) -> str | None:
    if unit is None:
        return None
    return _UNIT_ALIASES.get(unit.strip())


def to_canonical(pollutant: Pollutant, value: float, unit: str) -> float:
    """Convert `value` in `unit` to µg/m³. Raises UnitConversionError for unknown conversions."""
    unit_key = normalize_unit_token(unit)
    if unit_key == "ug/m3":
        return value
    if unit_key == "mg/m3":
        return value * 1000.0
    if unit_key in ("ppb", "ppm"):
        mw = MOLECULAR_WEIGHT_G_MOL.get(pollutant)
        if mw is None:
            raise UnitConversionError(f"{pollutant} is a particle mass; {unit} is not valid for it")
        ppb = value * 1000.0 if unit_key == "ppm" else value
        return ppb * mw / MOLAR_VOLUME_L_25C
    raise UnitConversionError(f"no documented conversion from {unit!r} for {pollutant}")

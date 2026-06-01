from __future__ import annotations

from typing import Any

from ..utils import dedupe_keep_order


VALID_DIRECTIONS = ("x", "y")
VALID_CARRIERS = ("electron", "hole")


def canonical_subchannel(direction: str, carrier: str) -> str:
    direction_token = str(direction or "").strip().lower()
    carrier_token = str(carrier or "").strip().lower()
    if direction_token not in VALID_DIRECTIONS:
        raise ValueError(f"invalid_direction:{direction}")
    if carrier_token not in VALID_CARRIERS:
        raise ValueError(f"invalid_carrier:{carrier}")
    return f"{carrier_token}_{direction_token}"


def default_subchannels() -> list[str]:
    return [canonical_subchannel(direction, carrier) for direction in VALID_DIRECTIONS for carrier in VALID_CARRIERS]


def direction_from_channel_token(token: Any) -> str | None:
    value = str(token or "").strip().lower()
    if value in VALID_DIRECTIONS:
        return value
    if "_" not in value:
        return None
    carrier, direction = value.split("_", 1)
    if carrier in VALID_CARRIERS and direction in VALID_DIRECTIONS:
        return direction
    return None


def subchannel_tokens_from_targets(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    tokens: list[str] = []
    for item in list(values or []):
        value = str(item or "").strip().lower()
        if not value:
            continue
        direction = direction_from_channel_token(value)
        if value in VALID_DIRECTIONS and direction is not None:
            tokens.extend(canonical_subchannel(direction, carrier) for carrier in VALID_CARRIERS)
        elif direction is not None:
            tokens.append(canonical_subchannel(direction, value.split("_", 1)[0]))
    return [str(item) for item in dedupe_keep_order(tokens)]


def directions_from_channel_tokens(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    directions: list[str] = []
    for item in list(values or []):
        direction = direction_from_channel_token(item)
        if direction is not None:
            directions.append(direction)
    return [str(item) for item in dedupe_keep_order(directions)]


def derive_direction_acceptance(retained_subchannels: Any, rejected_subchannels: Any | None = None) -> tuple[list[str], list[str]]:
    retained = set(subchannel_tokens_from_targets(retained_subchannels))
    rejected = set(subchannel_tokens_from_targets(rejected_subchannels))
    accepted_directions: list[str] = []
    rejected_directions: list[str] = []
    for direction in VALID_DIRECTIONS:
        members = {canonical_subchannel(direction, carrier) for carrier in VALID_CARRIERS}
        if members & retained:
            accepted_directions.append(direction)
        elif members and members.issubset(rejected):
            rejected_directions.append(direction)
    return (
        [str(item) for item in dedupe_keep_order(accepted_directions)],
        [str(item) for item in dedupe_keep_order(rejected_directions)],
    )

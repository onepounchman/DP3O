from types import UnionType
from typing import Any, Union, get_args, get_origin


def coerce_cli_value(value: str, annotation: Any) -> Any:
    """Coerce a string CLI override using a dataclass type annotation."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return coerce_cli_value(value, non_none[0])

    if annotation is bool:
        normalized = value.lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"Expected true or false, got {value!r}")
        return normalized == "true"
    if annotation in (int, float, str):
        return annotation(value)
    if origin is list:
        item_type = args[0] if args else str
        return [coerce_cli_value(item, item_type) for item in value.split(",")]
    return value

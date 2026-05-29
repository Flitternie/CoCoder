"""
A collection of update operations for TinyDB.

They are used for updates like this:

>>> db.update(delete('foo'), where('foo') == 2)

This would delete the ``foo`` field from all documents where ``foo`` equals 2.
"""

from typing import Any, Callable, MutableMapping, Union

Number = Union[int, float]


def delete(field: str) -> Callable[[MutableMapping[str, Any]], None]:
    """Return an update function that deletes `field` from a document.

    The returned function mutates the mapping in-place. If the field is not
    present, no exception is raised.
    """
    def transform(doc: MutableMapping[str, Any]) -> None:
        # Remove the field if present; do nothing otherwise.
        doc.pop(field, None)

    return transform


def add(field: str, n: Number) -> Callable[[MutableMapping[str, Any]], None]:
    """Return an update function that adds ``n`` to ``field``.

    If the field is missing it is treated as 0. Only ints and floats are
    supported for arithmetic; a TypeError is raised if an existing value is
    non-numeric.
    """
    def transform(doc: MutableMapping[str, Any]) -> None:
        current = doc.get(field, 0)
        if not isinstance(current, (int, float)):
            raise TypeError(f"Field '{field}' has non-numeric value: {type(current)!r}")
        doc[field] = current + n

    return transform


def subtract(field: str, n: Number) -> Callable[[MutableMapping[str, Any]], None]:
    """Return an update function that subtracts ``n`` from ``field``.

    Behaves like :func:`add` but subtracts the given value.
    """
    def transform(doc: MutableMapping[str, Any]) -> None:
        current = doc.get(field, 0)
        if not isinstance(current, (int, float)):
            raise TypeError(f"Field '{field}' has non-numeric value: {type(current)!r}")
        doc[field] = current - n

    return transform


def set(field: str, val: Any) -> Callable[[MutableMapping[str, Any]], None]:
    """Return an update function that sets ``field`` to ``val``.

    The returned function mutates the mapping in-place.
    """
    def transform(doc: MutableMapping[str, Any]) -> None:
        doc[field] = val

    return transform


def increment(field: str) -> Callable[[MutableMapping[str, Any]], None]:
    """Return an update function that increments ``field`` by 1.

    Implemented as a thin wrapper around :func:`add`.
    """
    return add(field, 1)


def decrement(field: str) -> Callable[[MutableMapping[str, Any]], None]:
    """Return an update function that decrements ``field`` by 1.

    Implemented as a thin wrapper around :func:`subtract`.
    """
    return subtract(field, 1)

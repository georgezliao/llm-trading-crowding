"""Strict observer-facing loading boundary for public episode data."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TypeAlias

import numpy as np

from herdlab.market.records import PublicRecord, public_field_names


PublicRecordSource: TypeAlias = str | PathLike[str] | PublicRecord


def load_public_record(source: PublicRecordSource) -> PublicRecord:
    """Return public data, rejecting private objects and mixed-schema files.

    Serialized ``.npz`` inputs must contain exactly the fields in
    :class:`PublicRecord`.  An archive containing even one private or unknown
    array is rejected rather than silently dropping it.
    """

    if isinstance(source, PublicRecord):
        return source
    if not isinstance(source, (str, PathLike)):
        raise TypeError("observer loader accepts only a PublicRecord or public .npz path")

    path = Path(source)
    if path.suffix != ".npz":
        raise ValueError("observer data must be stored in a .npz file")
    expected = set(public_field_names())
    with np.load(path, allow_pickle=False) as archive:
        actual = set(archive.files)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"not a public-record archive; missing={missing}, extra={extra}"
            )
        values = {name: np.asarray(archive[name]).copy() for name in expected}
    return PublicRecord(**values)


__all__ = ["PublicRecordSource", "load_public_record"]

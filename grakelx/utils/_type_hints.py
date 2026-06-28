"""Type hints for the ``grakelx.utils`` module.

This module centralises reusable type aliases used by the graph conversion
utilities. Keeping them in a dedicated module avoids circular imports and
keeps the public converters free of verbose inline type definitions.
"""

from __future__ import annotations

from typing import TypeAlias

#: Recursive alias representing a graph node/edge label.
#:
#: A label is any hashable Python value, restricted in practice to scalar
#: primitives (``int``, ``str``, ``float``, ``bool``) and to nested tuples of
#: such primitives (a common encoding for structured labels such as
#: ``("C", 1)``). ``np.ndarray`` is intentionally excluded because arrays
#: are not hashable and cannot be used as dictionary keys, which is how
#: labels are stored internally.
Label: TypeAlias = int | str | float | bool | tuple["Label", ...]

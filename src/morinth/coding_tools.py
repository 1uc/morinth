# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

def with_default(primary_value, default_value):
    """
    Return `primary_value` unless it's `None`.

    Examples:
        >>> def foo(bar=None):
        >>>     bar = with_default(bar, 42.0)

        >>> print(foo(None))
        >>> 42.0
        >>> print(foo("Convenient."))
        >>> Convenient.
    """

    if primary_value is not None:
        return primary_value
    else:
        return default_value

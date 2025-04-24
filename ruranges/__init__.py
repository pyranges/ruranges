
from typing import Any, Literal, TypeVar
import numpy as np
from numpy.typing import NDArray

from .ruranges import *


# Define a type variable for groups that allows only int8, int16, or int32.
GroupIdInt = TypeVar("GroupIdInt", np.int8, np.int16, np.int32)

# Define another type variable for the range arrays (starts/ends), which can be any integer.
RangeInt = TypeVar("RangeInt", bound=np.integer)


def overlaps(
    *,
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    starts2: NDArray[RangeInt],
    ends2: NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None = None,
    groups2: NDArray[GroupIdInt] | None = None,
    multiple: Literal["first", "all", "last", "contained"] = "all",
    contained: bool = False,
    slack: int = 0,
) -> tuple[GroupIdInt, GroupIdInt]:
    """
    Compute overlapping intervals between two sets of ranges.

    The four mandatory arrays (starts, ends, starts2, ends2) must all have the same length.
    If one of groups or groups2 is provided, then both must be provided and have the same length
    as the other arrays.

    The function returns a tuple (idx1, idx2) of numpy arrays, where each pair (idx1[i], idx2[i])
    indicates an overlapping interval between the first and second set.

    Examples
    --------
    Without groups:

    >>> import numpy as np
    >>> from numpy.typing import NDArray
    >>> RangeInt = np.int32
    >>> GroupIdInt = np.int32
    >>> starts = np.array([1, 10], dtype=RangeInt)
    >>> ends   = np.array([5, 15], dtype=RangeInt)
    >>> starts2 = np.array([3, 20], dtype=RangeInt)
    >>> ends2   = np.array([6, 25], dtype=RangeInt)
    >>> result = overlaps(starts=starts, ends=ends, starts2=starts2, ends2=ends2)
    >>> # In this hypothetical example only the first intervals overlap.
    >>> result
    (array([0], dtype=uint32), array([0], dtype=uint32))

    With groups:

    >>> starts = np.array([1, 1], dtype=RangeInt)
    >>> ends   = np.array([5, 5], dtype=RangeInt)
    >>> starts2 = np.array([3, 20], dtype=RangeInt)
    >>> ends2   = np.array([6, 25], dtype=RangeInt)
    >>> groups = np.array([1, 2], dtype=GroupIdInt)
    >>> groups2 = np.array([1, 2], dtype=GroupIdInt)
    >>> result = overlaps(starts=starts, ends=ends, starts2=starts2, ends2=ends2,
    ...                   groups=groups, groups2=groups2)
    >>> # Here the algorithm checks overlaps only within the same group.
    >>> result
    (array([0], dtype=uint32), array([0], dtype=uint32))

    Additional parameters such as `multiple`, `contained`, and `slack` control the overlap
    behavior; see the documentation for details.

    Raises
    ------
    ValueError
        If any of the length checks fail or if only one of groups/groups2 is provided.
    """

    length = check_array_lengths(starts, ends, groups)
    length2 = check_array_lengths(starts2, ends2, groups2)

    groups_validated = validate_groups(length, groups)
    groups2_validated = validate_groups(length2, groups2)

    _dtype_groupids = check_and_return_common_type_2(
        groups_validated, groups2_validated
    )
    _dtype_ranges = check_and_return_common_type_4(starts, ends, starts2, ends2)

    if slack:
        check_min_max_with_slack(starts, ends, slack, _dtype_ranges)

    idx1, idx2 = chromsweep_numpy(
        groups_validated,
        starts.astype(np.int64),
        ends.astype(np.int64),
        groups2_validated,
        starts2.astype(np.int64),
        ends2.astype(np.int64),
        slack,
        overlap_type=multiple,
        contained=contained,
    )

    return idx1.astype(_dtype_groupids), idx2.astype(_dtype_groupids)


def check_min_max_with_slack(
    starts: np.ndarray | list,
    ends: np.ndarray | list,
    slack: float,
    old_dtype: np.dtype | type,
) -> None:
    """Check whether the min of `starts` minus `slack` and the max of `ends` plus `slack` both fit into the range of `old_dtype`.
    Returns:
        True if both bounds fit, False otherwise.
    """
    # Convert old_dtype to a NumPy dtype object
    target_dtype = np.dtype(old_dtype)

    # Convert starts/ends to arrays (in case they're Python lists)
    arr_starts = np.asarray(starts)
    arr_ends = np.asarray(ends)

    # Compute "adjusted" bounds
    adjusted_min = arr_starts.min() - slack
    adjusted_max = arr_ends.max() + slack

    # Depending on whether it's an integer or floating dtype, get the min/max
    dtype_info: np.finfo[np.floating[Any]] | np.iinfo[Any]
    if target_dtype.kind == "i":
        dtype_info = np.iinfo(target_dtype)
    elif target_dtype.kind == "f":
        dtype_info = np.finfo(target_dtype)
    else:
        # Complex, object, etc. - no range check
        msg = f"Range check not implemented for dtype {target_dtype}."
        raise TypeError(msg)

    # Check if the adjusted min is too small
    if adjusted_min < dtype_info.min:
        msg = (
            f"Adjusted min ({adjusted_min}) is below the minimum "
            f"{dtype_info.min} for dtype {target_dtype}. "
            "Please use a smaller slack to avoid an out of bounds error."
        )
        raise ValueError(msg)

    # Check if the adjusted max is too large
    if adjusted_max > dtype_info.max:
        msg = (
            f"Adjusted max ({adjusted_max}) is above the maximum "
            f"{dtype_info.max} for dtype {target_dtype}. "
            "Please use a smaller slack to avoid an out of bounds error."
        )
        raise ValueError(msg)


def check_and_return_common_type_2(
    starts: np.ndarray, ends: np.ndarray
) -> np.dtype:
    """Check that `starts` and `ends` share the same dtype.
    If they do not, raises a TypeError.
    """
    if not isinstance(starts, np.ndarray):
        raise TypeError("`starts` must be a numpy.ndarray, not %r" % type(starts))
    if not isinstance(ends, np.ndarray):
        raise TypeError("`ends` must be a numpy.ndarray, not %r" % type(ends))

    dtype_starts = starts.dtype
    dtype_ends = ends.dtype

    if dtype_starts != dtype_ends:
        raise TypeError(
            f"`starts` and `ends` do not share the same dtype: {dtype_starts} != {dtype_ends}."
        )

    return dtype_starts


def check_and_return_common_type_4(
    start1: np.ndarray,
    end1: np.ndarray,
    start2: np.ndarray,
    end2: np.ndarray,
) -> np.dtype:
    """Check that start1, end1, start2, and end2 all share the same dtype.
    If they do not, raises a TypeError.
    """
    arrays = {
        "start1": start1,
        "end1": end1,
        "start2": start2,
        "end2": end2,
    }

    for name, arr in arrays.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"`{name}` must be a numpy.ndarray, not {type(arr)!r}")

    dtypes = {arr.dtype for arr in arrays.values()}
    if len(dtypes) != 1:
        raise TypeError(
            f"start1, end1, start2, end2 do not share the same dtype: {dtypes}."
        )

    return dtypes.pop()



def check_array_lengths(
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None = None,
) -> int:
    """
    Checks that the required input arrays have the same length.

    - `starts` and `ends` must have the same length.
    - If `groups` is provided, it must have the same length as `starts` and `ends`.

    Returns:
        The common length of the arrays.

    Raises:
        ValueError: If any of the length checks fail.
    """
    n = len(starts)
    if len(ends) != n:
        raise ValueError("`starts` and `ends` must have the same length.")
    if groups is not None and len(groups) != n:
        raise ValueError("`groups` must have the same length as `starts` and `ends`.")
    return n


def validate_groups(
    length: int,
    groups: NDArray[GroupIdInt] | None = None,
) -> NDArray[GroupIdInt]:
    """
    Ensures a single group array matches the expected length, or provides a default zero-filled array.

    Parameters:
    - length: Expected length of the group array.
    - groups: Optional NDArray of group IDs.

    Returns:
        An NDArray of group IDs of the given length.

    Raises:
        ValueError: If `groups` is provided but its length does not equal `length`.
    """
    if groups is None:
        return np.zeros(length, dtype=np.int32)

    if len(groups) != length:
        raise ValueError("`groups` must have the same length as specified by `length`.")

    return groups

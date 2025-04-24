import importlib
from typing import Any, Callable, Literal, TypeVar
import numpy as np
from numpy.typing import NDArray


# Define a type variable for groups that allows only int8, int16, or int32.
GroupIdInt = TypeVar("GroupIdInt", np.int8, np.int16, np.int32)

# Define another type variable for the range arrays (starts/ends), which can be any integer.
RangeInt = TypeVar("RangeInt", bound=np.integer)


# dtype-suffix map shared by every operation
# (group_dtype, range_dtype)  →  (suffix, group_target_dtype, range_target_dtype)
_SUFFIX_TABLE = {
    # ─── uint8 groups ────────────────────────────────────────────────
    (np.dtype(np.uint8),  np.dtype(np.int8)):  ("u8_i16",  np.uint8,  np.int16),
    (np.dtype(np.uint8),  np.dtype(np.int16)): ("u8_i16",  np.uint8,  np.int16),
    (np.dtype(np.uint8),  np.dtype(np.int32)): ("u8_i32",  np.uint8,  np.int32),
    (np.dtype(np.uint8),  np.dtype(np.int64)): ("u8_i64",  np.uint8,  np.int64),

    # ─── uint16 groups ───────────────────────────────────────────────
    (np.dtype(np.uint16), np.dtype(np.int8)):  ("u16_i16", np.uint16, np.int16),
    (np.dtype(np.uint16), np.dtype(np.int16)): ("u16_i16", np.uint16, np.int16),
    (np.dtype(np.uint16), np.dtype(np.int32)): ("u16_i32", np.uint16, np.int32),
    (np.dtype(np.uint16), np.dtype(np.int64)): ("u16_i64", np.uint16, np.int64),

    # ─── uint32 groups ───────────────────────────────────────────────
    (np.dtype(np.uint32), np.dtype(np.int8)):  ("u32_i16", np.uint32, np.int16),
    (np.dtype(np.uint32), np.dtype(np.int16)): ("u32_i16", np.uint32, np.int16),
    (np.dtype(np.uint32), np.dtype(np.int32)): ("u32_i32", np.uint32, np.int32),
    (np.dtype(np.uint32), np.dtype(np.int64)): ("u32_i64", np.uint32, np.int64),

    # ─── uint64 groups ───────────────────────────────────────────────
    (np.dtype(np.uint64), np.dtype(np.int8)):  ("u64_i64", np.uint64, np.int64),
    (np.dtype(np.uint64), np.dtype(np.int16)): ("u64_i64", np.uint64, np.int64),
    (np.dtype(np.uint64), np.dtype(np.int32)): ("u64_i64", np.uint64, np.int64),
    (np.dtype(np.uint64), np.dtype(np.int64)): ("u64_i64", np.uint64, np.int64),
}



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

    idx1, idx2 = _dispatch_binary(
        "chromsweep_numpy",
        groups,
        starts,
        ends,
        groups2,
        starts2,
        ends2,
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


def check_and_return_common_type_2(starts: np.ndarray, ends: np.ndarray) -> np.dtype:
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


# ---- zero-copy cast -------------------------------------------------
def _cast(
    a: NDArray,
    target: np.dtype,
) -> NDArray:
    """Return *a* unchanged if dtype already matches, else cast with copy=False."""
    return (
        a if a.dtype == target else a.astype(target, copy=False)
    )  # ndarray.astype will
    # reuse the buffer when
    # copy=False and the
    # conversion is safe :contentReference[oaicite:2]{index=2}


# ---- resolve the correct Rust wrapper ------------------------------
def _resolve_rust_fn(
    prefix: str,
    grp_dt: np.dtype,
    pos_dt: np.dtype,
) -> tuple[Callable, np.dtype, np.dtype]:
    """Look up (wrapper, target_grp_dt, target_pos_dt) or raise TypeError."""
    try:
        suffix, tgt_grp, tgt_pos = _SUFFIX_TABLE[(grp_dt, pos_dt)]
    except KeyError as exc:
        raise TypeError(f"Unsupported dtype pair: {grp_dt}, {pos_dt}") from exc

    rust_mod = importlib.import_module(".ruranges", package="ruranges")
    rust_fn = getattr(rust_mod, f"{prefix}_{suffix}")
    return rust_fn, tgt_grp, tgt_pos


def _dispatch_unary(
    prefix: str,
    chroms: NDArray,
    starts: NDArray,
    ends: NDArray,
    *extra_pos_args: Any,
    **extra_kw: Any,
):
    """Common body for functions that take one (chroms, starts, ends) trio."""
    rust_fn, grp_t, pos_t = _resolve_rust_fn(prefix, chroms.dtype, starts.dtype)

    chroms_c = _cast(chroms, grp_t)
    starts_c = _cast(starts, pos_t)
    ends_c = _cast(ends, pos_t)

    return rust_fn(chroms_c, starts_c, ends_c, *extra_pos_args, **extra_kw)


def _dispatch_binary(
    prefix: str,
    chroms1: NDArray,
    starts1: NDArray,
    ends1: NDArray,
    chroms2: NDArray,
    starts2: NDArray,
    ends2: NDArray,
    *extra_pos: Any,
    **extra_kw: Any,
):
    """Shared body for all two-interval-set operations."""
    rust_fn, grp_t, pos_t = _resolve_rust_fn(prefix, chroms1.dtype, starts1.dtype)

    # cast *every* array to the targets
    g1 = _cast(chroms1, grp_t)
    g2 = _cast(chroms2, grp_t)
    s1 = _cast(starts1, pos_t)
    e1 = _cast(ends1, pos_t)
    s2 = _cast(starts2, pos_t)
    e2 = _cast(ends2, pos_t)

    return rust_fn(g1, s1, e1, g2, s2, e2, *extra_pos, **extra_kw)

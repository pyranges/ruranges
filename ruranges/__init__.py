import importlib
from typing import Any, Callable, Literal, Sequence, TypeVar
import numpy as np
from numpy.typing import NDArray

from importlib.metadata import version
__version__: str = version(__name__)


# Define a type variable for groups that allows only int8, int16, or int32.
GroupIdInt = TypeVar("GroupIdInt", np.int8, np.int16, np.int32)

# Define another type variable for the range arrays (starts/ends), which can be any integer.
RangeInt = TypeVar("RangeInt", bound=np.integer)


# dtype-suffix map shared by every operation
# (group_dtype, range_dtype)  →  (suffix, group_target_dtype, range_target_dtype)
_SUFFIX_TABLE = {
    ("u8", None):  (None, "u8", np.uint8),
    ("u16", None): (None, "u16", np.uint16),
    ("u32", None): (None, "u32", np.uint32),

    (None, np.dtype(np.int8)):  ("i16", None, np.int16),
    (None, np.dtype(np.int16)): ("i16", None, np.int16),
    (None, np.dtype(np.int32)): ("i32", None, np.int32),
    (None, np.dtype(np.int64)): ("i64", None, np.int64),
    # ─── uint8 groups ────────────────────────────────────────────────
    (np.dtype(np.uint8), np.dtype(np.int8)): ("u8_i16", np.uint8, np.int16),
    (np.dtype(np.uint8), np.dtype(np.int16)): ("u8_i16", np.uint8, np.int16),
    (np.dtype(np.uint8), np.dtype(np.int32)): ("u8_i32", np.uint8, np.int32),
    (np.dtype(np.uint8), np.dtype(np.int64)): ("u8_i64", np.uint8, np.int64),
    # ─── uint16 groups ───────────────────────────────────────────────
    (np.dtype(np.uint16), np.dtype(np.int8)): ("u16_i16", np.uint16, np.int16),
    (np.dtype(np.uint16), np.dtype(np.int16)): ("u16_i16", np.uint16, np.int16),
    (np.dtype(np.uint16), np.dtype(np.int32)): ("u16_i32", np.uint16, np.int32),
    (np.dtype(np.uint16), np.dtype(np.int64)): ("u16_i64", np.uint16, np.int64),
    # ─── uint32 groups ───────────────────────────────────────────────
    (np.dtype(np.uint32), np.dtype(np.int8)): ("u32_i16", np.uint32, np.int16),
    (np.dtype(np.uint32), np.dtype(np.int16)): ("u32_i16", np.uint32, np.int16),
    (np.dtype(np.uint32), np.dtype(np.int32)): ("u32_i32", np.uint32, np.int32),
    (np.dtype(np.uint32), np.dtype(np.int64)): ("u32_i64", np.uint32, np.int64),
    # ─── uint64 groups ───────────────────────────────────────────────
    (np.dtype(np.uint64), np.dtype(np.int8)): ("u64_i64", np.uint64, np.int64),
    (np.dtype(np.uint64), np.dtype(np.int16)): ("u64_i64", np.uint64, np.int64),
    (np.dtype(np.uint64), np.dtype(np.int32)): ("u64_i64", np.uint64, np.int64),
    (np.dtype(np.uint64), np.dtype(np.int64)): ("u64_i64", np.uint64, np.int64),
}

RETURN_SIGNATURES: dict[str, tuple[str, ...]] = {
    "chromsweep_numpy": ("grp", "grp"),
    "nearest_numpy": ("grp", "grp", "pos"),
    "subtract_numpy": ("grp", "pos", "pos"),
    "complement_overlaps_numpy": ("grp",),
    "count_overlaps_numpy": ("count",),
    "sort_groups_numpy": ("idx",),
    "sort_intervals_numpy": ("idx",),
    "cluster_numpy": ("idx", "count"),
    "max_disjoint_numpy": ("idx",),
    "merge_numpy": ("grp", "pos", "pos", "count"),
    "window_numpy": ("grp", "pos", "pos"),
    "tile_numpy": ("grp", "pos", "pos", "fraction"),
    "complement_numpy": ("grp", "pos", "pos", "index"),
    "boundary_numpy": ("index", "pos", "pos", "count"),
    "spliced_subsequence_numpy": ("index", "pos", "pos", "_strand"),
    "spliced_subsequence_multi_numpy": ("index", "pos", "pos", "_strand"),
    "split_numpy": ("index", "pos", "pos"),
    "extend_numpy": ("pos", "pos"),
    "genome_bounds_numpy": ("index", "pos", "pos"),
    "group_cumsum_numpy": ("index", "pos", "pos"),
    "map_to_global_numpy": ("index", "pos", "pos", "strand"),
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
    >>> GroupIdInt = np.uint32
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

    return _dispatch_binary(
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


def map_to_global(
    *,
    # ─── query (local) table ─────────────────────────────────────────
    starts:    NDArray[RangeInt],
    ends:      NDArray[RangeInt],
    groups:    NDArray[GroupIdInt],
    strand:    NDArray[np.bool_],
    # ─── exon (annotation) table ─────────────────────────────────────
    starts2:   NDArray[RangeInt],
    ends2:     NDArray[RangeInt],
    groups2:   NDArray[GroupIdInt],
    chr_code2: NDArray[GroupIdInt],
    genome_start2: NDArray[RangeInt],
    genome_end2:   NDArray[RangeInt],
    strand2:   NDArray[np.bool_],
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Vectorised transcript-to-genome projection.

    All arrays must be 1-D and already sorted per transcript by
    (local_start).  The *strand* arrays encode '+' as True and '−' as False.

    Returns
    -------
    keep_idx : uint32
        Row numbers into the *query* table (duplicated if interval splits).
    g_start / g_end : same dtype as *starts*
        Genomic coordinates of the mapped pieces.
    g_strand_bool : bool
        True ⇒ '+', False ⇒ '−'.
    """

    return _dispatch_map_global_binary(
        "map_to_global_numpy",
        groups,  starts,  ends,  strand,
        groups2, starts2, ends2, strand2,
        chr_code2, genome_start2, genome_end2,
    )


def nearest(
    *,
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    starts2: NDArray[RangeInt],
    ends2: NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None = None,
    groups2: NDArray[GroupIdInt] | None = None,
    slack: int = 0,
    k: int = 1,
    include_overlaps: bool = True,
    direction: Literal["forward", "backward", "any"] = "any",
) -> tuple[NDArray[GroupIdInt], NDArray[GroupIdInt], NDArray[RangeInt]]:
    """
    Find the *k* nearest intervals from *(starts2, ends2)* for every interval
    in *(starts, ends)*, optionally restricting the search to matching
    `groups` / `groups2`.

    Parameters
    ----------
    starts, ends, starts2, ends2
        Coordinate arrays (all same length and dtype ``RangeInt``).
    groups, groups2
        Optional per-row group IDs (e.g. chromosome numbers).  If one is
        provided, the other **must** be provided too, and both must be the
        same length as their corresponding coordinate arrays.
    slack
        Maximum distance allowed between intervals (0 ⇒ no limit).
    k
        Number of nearest neighbours to report per query interval.
    include_overlaps
        If *False*, overlapping intervals are *excluded*; otherwise they count
        with distance 0.
    direction
        • ``"forward"`` – only neighbours that start **after** the query ends
        • ``"backward"`` – only neighbours that end **before** the query starts
        • ``"any"`` (default) – both directions.

    Returns
    -------
    idx1, idx2, dist
        *idx1* / *idx2* are ``uint32`` indices into the first / second
        interval sets; *dist* is the coordinate-typed distance between each
        pair.

    Raises
    ------
    ValueError
        If the input lengths don’t match or only one of ``groups`` /
        ``groups2`` is supplied.
    """
    return _dispatch_binary(
        "nearest_numpy",
        groups,
        starts,
        ends,
        groups2,
        starts2,
        ends2,
        slack,
        k=k,
        include_overlaps=include_overlaps,
        direction=direction,
    )


def subtract(
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    starts2: NDArray[RangeInt],
    ends2: NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None = None,
    groups2: NDArray[GroupIdInt] | None = None,
) -> tuple[NDArray[GroupIdInt], NDArray[RangeInt], NDArray[RangeInt]]:
    return _dispatch_binary(
        "subtract_numpy",
        groups,
        starts,
        ends,
        groups2,
        starts2,
        ends2,
    )


def complement_overlaps(
    *,
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    starts2: NDArray[RangeInt],
    ends2: NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None = None,
    groups2: NDArray[GroupIdInt] | None = None,
    slack: int = 0,
) -> NDArray[GroupIdInt]:
    """
    Return the indices of intervals in *(starts, ends)* that **do not** overlap
    *any* interval in *(starts2, ends2)*, subject to an optional `slack`.

    Parameters
    ----------
    starts, ends, starts2, ends2
        Coordinate arrays. All four must have the same dtype ``RangeInt``.
    groups, groups2
        Optional per-row group IDs (e.g. chromosome numbers).  If one is
        supplied, the other **must** be supplied too.  Overlap checks are then
        performed *within* matching groups only.
    slack
        Two intervals are considered overlapping if their distance is
        *strictly* less than or equal to `slack`.  A value of 0 (default)
        means they must actually touch or intersect.

    Returns
    -------
    idx : NDArray[GroupIdInt]
        A `uint32` array of indices into the *first* interval set indicating
        which rows have **no** overlaps in the second set.

    Examples
    --------
    >>> import numpy as np
    >>> starts  = np.array([ 1, 10, 30], dtype=np.int32)
    >>> ends    = np.array([ 5, 15, 35], dtype=np.int32)
    >>> starts2 = np.array([ 3, 20],     dtype=np.int32)
    >>> ends2   = np.array([ 6, 25],     dtype=np.int32)
    >>> complement_overlaps(starts=starts, ends=ends,
    ...                     starts2=starts2, ends2=ends2)
    array([1, 2], dtype=uint32)
    """
    return _dispatch_binary(
        "complement_overlaps_numpy",  # selects the correct Rust wrapper
        groups,
        starts,
        ends,
        groups2,
        starts2,
        ends2,
        slack,
    )


def count_overlaps(
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    starts2: NDArray[RangeInt],
    ends2: NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None = None,
    groups2: NDArray[GroupIdInt] | None = None,
    slack: int = 0,
) -> NDArray[GroupIdInt]:
    """
    For every interval in *(starts, ends)*, count how many intervals in
    *(starts2, ends2)* overlap it (distance ≤ ``slack``).

    Parameters
    ----------
    starts, ends, starts2, ends2
        Coordinate arrays (all same dtype ``RangeInt``).
    groups, groups2
        Optional group IDs (chromosomes, contigs, …).  If one is given, the
        other **must** be given too.  Counts are computed *within* matching
        groups only.
    slack
        Two intervals are considered overlapping if their gap is ≤ `slack`
        (0 ⇒ they must actually touch/intersect).

    Returns
    -------
    counts : NDArray[GroupIdInt]
        ``uint32`` array, length == ``len(starts)``, holding the per-row
        overlap counts.
    """
    return _dispatch_binary(
        "count_overlaps_numpy",
        groups,
        starts,
        ends,
        groups2,
        starts2,
        ends2,
        slack,
    )


def sort_intervals(
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None = None,
    sort_reverse_direction: NDArray[np.bool_] | None = None,
) -> NDArray[GroupIdInt]:
    """
    Return the permutation that sorts *(starts, ends)* (and their optional
    ``groups``) in ascending genomic order.

    Parameters
    ----------
    starts, ends
        Coordinate arrays (dtype ``RangeInt``) of equal length.
    groups
        Optional per-row group IDs (chromosomes, contigs, …).  If supplied, the
        sort is performed *within* each group; otherwise intervals are sorted
        globally.
    sort_reverse_direction
        Optional boolean array (same length as *starts*) marking rows that
        should be ordered **descendingly** within their group/position tier.
        A value of *None* (default) means no per-row reversal.

    Returns
    -------
    perm : NDArray[GroupIdInt]
        ``uint32`` array whose elements are the indices that sort the input.

    Notes
    -----
    *The heavy lifting happens in Rust; this wrapper only dispatches to the
    correct concrete wrapper based on dtypes.*
    """
    return _dispatch_unary(
        "sort_intervals_numpy",  # selects the Rust wrapper
        groups=groups,
        starts=starts,
        ends=ends,
        sort_reverse_direction=sort_reverse_direction,
    )

def sort_groups(
    groups: NDArray[GroupIdInt],
) -> NDArray[GroupIdInt]:
    """
    Return the permutation that sorts groups in order.

    Parameters
    ----------
    groups
        Optional per-row group IDs (chromosomes, contigs, …).  If supplied, the
        sort is performed *within* each group; otherwise intervals are sorted
        globally.

    Returns
    -------
    perm : NDArray[GroupIdInt]
        ``uint32`` array whose elements are the indices that sort the input.

    Notes
    -----
    *The heavy lifting happens in Rust; this wrapper only dispatches to the
    correct concrete wrapper based on dtypes.*
    """
    return _dispatch_unary(
        "sort_groups_numpy",  # selects the Rust wrapper
        groups=groups,
    )


def cluster(
    starts: NDArray[RangeInt],
    ends:   NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None = None,
    slack:  int = 0,
) -> tuple[NDArray[GroupIdInt], NDArray[GroupIdInt]]:
    """
    Group nearby/overlapping intervals into clusters.

    Parameters
    ----------
    starts, ends
        Coordinate arrays (same dtype ``RangeInt``).
    groups
        Optional group IDs (chromosome, contig …); clustering is performed
        *within* each group.  If omitted, all intervals are considered to be
        in the same group.
    slack
        Two intervals belong to the same cluster if their gap is ≤ `slack`
        (0 ⇒ they must touch/overlap).

    Returns
    -------
    cluster_ids , order_idx : tuple of ``uint32`` arrays
        *cluster_ids* gives the cluster label per input row; *order_idx* is
        the permutation that sorts the rows by cluster then position.
    """
    return _dispatch_unary(
        "cluster_numpy",      # dispatch key – matches the Rust wrapper base
        groups=groups,
        starts=starts,
        ends=ends,
        slack=slack,
    )

def merge(
    *,
    starts: NDArray[RangeInt],
    ends:   NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None = None,
    slack:  int = 0,
) -> tuple[
    NDArray[GroupIdInt],  # indices
    NDArray[RangeInt],    # merged starts
    NDArray[RangeInt],    # merged ends
    NDArray[GroupIdInt],  # counts
]:
    """
    Merge overlapping / *slack*-close intervals, optionally per group.

    Parameters
    ----------
    starts, ends
        Coordinate arrays (dtype ``RangeInt``).
    groups
        Optional group IDs (chromosome, contig …).  Merging is performed
        independently within each group.  Omit to merge globally.
    slack
        Two intervals are merged if their gap is ≤ `slack`
        (0 ⇒ they must touch/intersect).

    Returns
    -------
    indices, merged_starts, merged_ends, counts
        *indices* is the ``uint32`` row index of the first interval that
        contributed to each merged output.  *counts* reports how many original
        intervals were collapsed into each merge.
    """
    return _dispatch_unary(
        "merge_numpy",        # base name of the Rust wrapper
        groups=groups,
        starts=starts,
        ends=ends,
        slack=slack,
    )

def max_disjoint(
    *,
    starts: NDArray[RangeInt],
    ends:   NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None = None,
    slack:  int = 0,
) -> NDArray[GroupIdInt]:
    """
    Select a *maximum* subset of mutually non-overlapping intervals.

    Parameters
    ----------
    starts, ends
        Coordinate arrays (`dtype == RangeInt`, same length).
    groups
        Optional group IDs (chromosome, contig …).  The algorithm is applied
        independently within each group; omit to treat all intervals together.
    slack
        Two intervals are considered overlapping if their gap is ≤ `slack`
        (0 ⇒ they must touch/intersect).  Increase to allow a small gap.

    Returns
    -------
    indices : NDArray[GroupIdInt]
        ``uint32`` array of *row indices* in the **input** that comprise the
        largest disjoint subset.

    Notes
    -----
    The heavy lifting happens in Rust; this wrapper simply dispatches to the
    correct concrete wrapper based on the NumPy dtypes you provide.
    """
    return _dispatch_unary(
        "max_disjoint_numpy",   # base name of the Rust wrapper
        groups=groups,
        starts=starts,
        ends=ends,
        slack=slack,
    )


def complement(
    *,
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None,
    chrom_len_ids: NDArray[GroupIdInt],
    chrom_lens: NDArray[RangeInt],
    slack: int = 0,
    include_first_interval: bool = False,
) -> tuple[
    NDArray[GroupIdInt],  # out_chrs
    NDArray[RangeInt],    # out_starts
    NDArray[RangeInt],    # out_ends
    NDArray[GroupIdInt],  # out_idx
]:
    """
    Return the complement (gaps) of an interval set within chromosome bounds.

    Parameters
    ----------
    starts, ends
        Input coordinates (`dtype == RangeInt`).
    groups
        Optional per-row chromosome IDs.  If *None*, *chrom_len_ids* must
        describe a *single* chromosome that covers all intervals.
    chrom_len_ids, chrom_lens
        Parallel arrays mapping chromosome IDs to their total length.
    slack
        Two intervals are considered contiguous if the gap between them is
        ≤ `slack`.  Gaps smaller than or equal to `slack` are *not* reported.
    include_first_interval
        If *True*, emit a gap *before* the first input interval on each
        chromosome (from 0 to `start[0] − 1`).

    Returns
    -------
    out_chrs, out_starts, out_ends, out_idx
        `out_idx` holds the 0-based index of the input interval immediately
        **following** each gap (useful for attribution).

    Notes
    -----
    All heavy lifting happens in Rust; this wrapper only dispatches to the
    concrete wrapper that matches your NumPy dtypes.
    """
    return _dispatch_unary(
        "complement_numpy",
        starts,
        ends,
        groups,
        slack=slack,
        chrom_len_ids=chrom_len_ids,
        chrom_lens=chrom_lens,
        include_first_interval=include_first_interval,
    )

def boundary(
    *,
    starts: NDArray[RangeInt],
    ends:   NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None = None,
) -> tuple[
    NDArray[GroupIdInt],  # indices
    NDArray[RangeInt],    # boundary starts
    NDArray[RangeInt],    # boundary ends
    NDArray[GroupIdInt],  # counts
]:
    """
    Collapse adjacent/overlapping intervals into *boundary segments*.

    Each boundary represents a contiguous genomic stretch where at least one
    interval is present.  Returns the permutation (indices) of the *first*
    interval contributing to each boundary and how many intervals overlapped
    the segment (*counts*).

    Parameters
    ----------
    starts, ends
        Coordinate arrays (dtype ``RangeInt``).
    groups
        Optional per-row chromosome/contig IDs.  If supplied, boundaries are
        computed within each group independently.

    Returns
    -------
    indices, boundary_starts, boundary_ends, counts
        All arrays are `uint32` for IDs/counts and ``RangeInt`` for
        coordinates, matching the rest of the API.
    """
    return _dispatch_unary(
        "boundary_numpy",   # base name of the Rust wrapper
        starts,
        ends,
        groups,
    )


def window(
    *,
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    negative_strand: NDArray[np.bool_],
    window_size: int,
) -> tuple[
    NDArray[GroupIdInt],  # indices
    NDArray[RangeInt],    # windowed starts
    NDArray[RangeInt],    # windowed ends
]:
    """
    Expand each interval upstream/downstream by `window_size` bases.

    Expansion direction:

    * **positive strand** (`negative_strand == False`)
      → start − *window_size*, end + *window_size*
    * **negative strand** (`negative_strand == True`)
      → start + *window_size*, end − *window_size*

    Returns the permutation (`indices`) that sorts the window-shifted
    intervals, plus the shifted coordinates.

    Notes
    -----
    All heavy lifting happens inside Rust; this wrapper only dispatches to the
    correct concrete wrapper based on NumPy dtypes.
    """
    # `_dispatch_unary` accepts `groups=None` because this operation has no
    # grouping column.
    return _dispatch_unary(
        "window_numpy",          # base name of the Rust wrapper
        groups=None,                    # groups == None
        starts=starts,
        ends=ends,
        negative_strand=negative_strand,
        window_size=window_size,
    )

def tile(
    *,
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    negative_strand: NDArray[np.bool_],
    tile_size: int,
) -> tuple[
    NDArray[GroupIdInt],  # indices
    NDArray[RangeInt],    # tile starts
    NDArray[RangeInt],    # tile ends
    NDArray[np.float64],  # overlap fraction
]:
    """
    Split each interval into fixed-size tiles.

    * For positive-strand rows (`negative_strand == False`) tiling proceeds
      from 5'→3'.
    * For negative-strand rows (`negative_strand == True`) tiling proceeds
      from 3'→5'.

    Parameters
    ----------
    starts, ends
        Coordinate arrays (dtype ``RangeInt``).
    negative_strand
        Boolean array indicating strand per interval.
    tile_size
        Desired tile length in the same units as *starts/ends*.

    Returns
    -------
    indices, tile_starts, tile_ends, overlap_fraction
        *indices* (`uint32`) is the permutation that sorts the tiles in
        genomic order; *overlap_fraction* reports, for each tile, the fraction
        of its bases that overlap the original interval (useful when the last
        tile is truncated).
    """
    return _dispatch_unary(
        "tile_numpy",        # base name of the Rust wrapper
        groups=None,                    # groups == None
        starts=starts,
        ends=ends,
        negative_strand=negative_strand,
        tile_size=tile_size,
    )

def _as_vec(x, n: int, dtype) -> NDArray:
    """Return `x` as a 1-D ndarray of length *n*, repeating scalars if needed."""
    if np.isscalar(x):
        return np.full(n, x, dtype=dtype)
    arr = np.asarray(x, dtype=dtype)
    if arr.shape != (n,):
        raise ValueError("vector length mismatch")
    return arr


def spliced_subsequence(                     # same public signature
    *,
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None,
    strand_flags: NDArray[np.bool_],
    start: int | NDArray | list[int],
    end: int | NDArray | list[int] | None = None,
    force_plus_strand: bool = False,
) -> tuple[NDArray[GroupIdInt], NDArray[RangeInt], NDArray[RangeInt]]:

    n = len(starts)
    dtype = starts.dtype
    sentinel = np.iinfo(dtype).max                       # numeric “no-end” marker

    slice_starts = _as_vec(start, n, dtype=dtype)

    if end is None:
        slice_ends = np.full(n, sentinel, dtype=dtype)
    else:
        end_arr = np.asarray(end, dtype=object)
        if end_arr.shape == ():                         # scalar
            slice_ends = np.full(n, end_arr.item(), dtype=dtype)
        else:
            # replace None with sentinel, then cast once
            cleaned = [sentinel if v is None else v for v in end_arr]
            slice_ends = np.asarray(cleaned, dtype=dtype)

    return _dispatch_unary(                             # calls Rust wrapper
        "spliced_subsequence_multi_numpy",
        starts,
        ends,
        groups,
        strand_flags=strand_flags,
        slice_starts=slice_starts,
        slice_ends=slice_ends,
        force_plus_strand=force_plus_strand,
    )[:3]


def split(
    *,
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None = None,
    slack: int = 0,
    between: bool = False,
) -> tuple[
    NDArray[GroupIdInt],  # indices
    NDArray[RangeInt],    # split starts
    NDArray[RangeInt],    # split ends
]:
    """
    Split intervals wherever the gap **exceeds** `slack`.

    If *between* is *True*, emit only the *gaps* (regions between blocks);
    otherwise emit the original blocks split at the large gaps.

    Parameters
    ----------
    starts, ends
        Coordinate arrays (dtype ``RangeInt``).
    groups
        Optional per-row chromosome/contig IDs—splitting is performed within
        each group independently.
    slack
        Maximum internal gap tolerated **within** a block.  A gap >`slack`
        triggers a split.
    between
        If *True*, return the gaps; if *False* (default), return the block
        fragments.

    Returns
    -------
    indices, part_starts, part_ends
        *indices* (`uint32`) identifies the input interval that produced each
        output fragment or gap.
    """
    return _dispatch_unary(
        "split_numpy",   # base name of the Rust wrapper
        starts,
        ends,
        groups,
        slack=slack,
        between=between,
    )

def extend(
    *,
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    negative_strand: NDArray[np.bool_],
    groups: NDArray[GroupIdInt] | None = None,
    ext_3: int,
    ext_5: int,
) -> tuple[NDArray[RangeInt], NDArray[RangeInt]]:
    """Extend intervals upstream/downstream; see full docstring above."""
    if groups is None:
        groups = np.zeros(starts.shape[0], dtype=np.uint32)


    return _dispatch_unary(
        "extend_numpy",
        starts=starts,
        ends=ends,
        groups=groups,
        negative_strand=negative_strand,
        ext_3=ext_3,
        ext_5=ext_5,
    )

def group_cumsum(
    *,
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    negative_strand: NDArray[np.bool_],
    groups: NDArray[GroupIdInt] | None = None,
    sort: bool = True,
) -> tuple[NDArray[np.uint32], NDArray[RangeInt], NDArray[RangeInt]]:
    """
    Strand-aware cumulative lengths of every interval.

    For each chromosome the intervals are walked 5′→3′ **on their own strand**.
    The running total *before* and *after* each interval is returned, plus the
    original index of every interval so you can recover the traversal order.

    Parameters
    ----------
    starts, ends : 1-D integer arrays
        Zero-based, half-open coordinates (same shape, same dtype).
    negative_strand : 1-D bool array
        ``True`` for minus-strand intervals, ``False`` for plus-strand.
    groups : 1-D integer array, optional
        Chromosome / contig / group IDs.  If *None*, every interval is assumed
        to belong to a single group (filled with zeros).
    sort : bool, default True
        Whether to sort the results by the original row order.

    Returns
    -------
    tuple of ndarray
        ``(idx, cumsum_start, cumsum_end)``, where

        * ``idx`` is ``uint32`` – original indices,
        * ``cumsum_start`` / ``cumsum_end`` share the dtype of *starts*/*ends*.

    Notes
    -----
    This is a thin wrapper around the Rust function family
    ``cumsum_numpy_*_*`` generated by the macro in `src/lib.rs`.
    """

    if groups is None:
        groups = np.zeros(starts.shape[0], dtype=np.uint32)

    return _dispatch_unary(
        "group_cumsum_numpy",
        starts=starts,
        ends=ends,
        groups=groups,
        negative_strand=negative_strand,
        sort=sort,
    )


def genome_bounds(
    *,
    groups: NDArray[GroupIdInt],
    starts: NDArray[RangeInt],
    ends:   NDArray[RangeInt],
    chrom_length: NDArray[RangeInt],
    clip: bool = False,
    only_right: bool = False,
) -> tuple[
    NDArray[np.uintp],   # indices (usize → uintp)
    NDArray[RangeInt],   # new starts
    NDArray[RangeInt],   # new ends
]:
    """
    Clip or flag intervals that extend beyond chromosome bounds.

    Parameters
    ----------
    groups, starts, ends
        Interval set. *groups* must reference the chromosomes in *chrom_ids*.
    chrom_ids, chrom_length
        Parallel arrays mapping chromosome IDs to their total length.
    clip
        If *True*, coordinates are clipped to the bounds; if *False* only
        intervals lying wholly outside are returned.
    only_right
        When *True*, treat `ends > chrom_length` as out-of-bounds but ignore
        negative starts.

    Returns
    -------
    idx, new_starts, new_ends
        *idx* is the index of the input row affected.  If *clip=False* the
        two coordinate arrays echo the offending interval; if *clip=True*
        they hold the clipped coordinates.
    """
    return _dispatch_unary(
        "genome_bounds_numpy",    # base name of the Rust wrapper
        starts,
        ends,
        groups,
        chrom_lengths=chrom_length,
        clip=clip,
        only_right=only_right,
    )

def minimal_integer_dtype(arr: NDArray[np.integer]) -> np.dtype:
    """Return the narrowest integer dtype that can hold *arr*,
    preserving the signed/unsigned kind of the original dtype.
    """
    arr = np.asanyarray(arr)
    if arr.size == 0:
        return arr.dtype

    lo: int = int(arr.min())
    hi: int = int(arr.max())

    if arr.dtype.kind == "u":
        for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
            if hi <= np.iinfo(dt).max:
                return np.dtype(dt)
    elif arr.dtype.kind == "i":
        for dt in (np.int8, np.int16, np.int32, np.int64):
            info = np.iinfo(dt)
            if lo >= info.min and hi <= info.max:
                return np.dtype(dt)
    else:
        raise TypeError("Input must use an integer dtype")

    raise ValueError("Values exceed 64-bit integer range")


def _common_integer_dtype(*arrays: NDArray[np.integer]) -> np.dtype:
    """Private: return an integer dtype that can safely hold
    every value in *arrays*."""
    mins = [minimal_integer_dtype(a) for a in arrays]
    common = np.result_type(*mins)
    if common.kind not in {"i", "u"}:
        raise TypeError("No common *integer* dtype for these arrays")
    return common


def cast_two_arrays(
    a: NDArray[np.integer],
    b: NDArray[np.integer],
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    """Return *a* and *b* converted to the largest minimal-integer dtype
    required by either array."""
    tgt = _common_integer_dtype(a, b)
    return a.astype(tgt, copy=False), b.astype(tgt, copy=False)


def cast_four_arrays(
    a: NDArray[np.integer],
    b: NDArray[np.integer],
    c: NDArray[np.integer],
    d: NDArray[np.integer],
) -> tuple[
    NDArray[np.integer],
    NDArray[np.integer],
    NDArray[np.integer],
    NDArray[np.integer],
]:
    """Same idea for four arrays."""
    tgt = _common_integer_dtype(a, b, c, d)
    return (
        a.astype(tgt, copy=False),
        b.astype(tgt, copy=False),
        c.astype(tgt, copy=False),
        d.astype(tgt, copy=False),
    )


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
        return np.zeros(length, dtype=np.uint8)

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
    starts: NDArray,
    ends: NDArray,
    groups: NDArray | None = None,
    *extra_pos_args: Any,
    **extra_kw: Any,
) -> Any:
    """Common body for functions that take one (chroms, starts, ends) trio."""
    roles = RETURN_SIGNATURES[prefix]

    length = check_array_lengths(starts, ends, groups)

    groups_validated = validate_groups(length, groups)

    grp_orig: np.dtype = groups_validated.dtype
    pos_orig: np.dtype = starts.dtype

    rust_fn, grp_t, pos_t = _resolve_rust_fn(prefix, groups.dtype if groups is not None else None, starts.dtype)

    chroms_c = _cast(groups, grp_t) if groups is not None else None
    starts_c = _cast(starts, pos_t)
    ends_c = _cast(ends, pos_t)

    if groups is not None:
        raw = rust_fn(chroms_c, starts_c, ends_c, *extra_pos_args, **extra_kw)
    else:
        raw = rust_fn(starts_c, ends_c, *extra_pos_args, **extra_kw)

    return cast_kernel_outputs(prefix, raw, roles, grp_t, pos_t, grp_orig, pos_orig)


def _dispatch_binary(
    prefix: str,
    groups: NDArray | None,
    starts: NDArray,
    ends: NDArray,
    groups2: NDArray | None,
    starts2: NDArray,
    ends2: NDArray,
    *extra_pos: any,
    **extra_kw: any,
):
    """Shared body for all two-interval-set operations with automatic
    down-casting to minimal integer dtypes."""

    length = check_array_lengths(starts, ends, groups)
    length2 = check_array_lengths(starts2, ends2, groups2)

    groups_validated = validate_groups(length, groups)
    groups2_validated = validate_groups(length2, groups2)

    # ------------------------------------------------------------------
    # 1.  Remember caller-visible dtypes
    # ------------------------------------------------------------------
    grp_orig: np.dtype = groups_validated.dtype
    pos_orig: np.dtype = starts.dtype

    # ------------------------------------------------------------------
    # 2.  Choose tightest common dtypes for groups and positions
    # ------------------------------------------------------------------
    grp_tmp = _common_integer_dtype(
        groups_validated, groups2_validated
    )  # signed/unsigned kept
    pos_tmp = _common_integer_dtype(starts, ends, starts2, ends2)

    # Slack range check (only if the caller supplied slack > 0)
    slack = extra_kw.get("slack", 0)
    try:
        if slack:
            check_min_max_with_slack(starts, ends, slack, pos_tmp)
    except ValueError:
        # Too narrow → fall back to original pos dtype
        pos_tmp = pos_orig

    # ------------------------------------------------------------------
    # 3.  Now resolve the Rust kernel for the *temporary* dtypes
    # ------------------------------------------------------------------
    rust_fn, grp_t, pos_t = _resolve_rust_fn(prefix, grp_tmp, pos_tmp)

    roles = RETURN_SIGNATURES[prefix]

    # ------------------------------------------------------------------
    # 4.  Cast inputs for the FFI call
    # ------------------------------------------------------------------
    g1 = groups_validated.astype(grp_t, copy=False)
    g2 = groups2_validated.astype(grp_t, copy=False)
    s1 = starts.astype(pos_t, copy=False)
    e1 = ends.astype(pos_t, copy=False)
    s2 = starts2.astype(pos_t, copy=False)
    e2 = ends2.astype(pos_t, copy=False)

    # ------------------------------------------------------------------
    # 5.  Dispatch & post-process results
    # ------------------------------------------------------------------
    raw = rust_fn(g1, s1, e1, g2, s2, e2, *extra_pos, **extra_kw)

    return cast_kernel_outputs(prefix, raw, roles, grp_t, pos_t, grp_orig, pos_orig)


def cast_kernel_outputs(
    prefix: str,
    raw_out: Any,
    roles: tuple[str, ...],
    grp_t: np.dtype,
    pos_t: np.dtype,
    grp_orig: np.dtype,
    pos_orig: np.dtype,
) -> Any:
    """
    Cast every array returned by a Rust kernel back to the caller-visible dtype.

    Parameters
    ----------
    prefix
        Kernel name (used only for clearer error messages).
    raw_out
        Value returned by the Rust FFI stub – either a single ndarray *or*
        a tuple of ndarrays.
    roles
        Tuple that labels each output position, e.g. ('grp', 'grp'),
        ('grp', 'grp', 'pos'), or ('grp',) for single-output kernels.
    grp_t / pos_t
        Temporary dtypes used when **calling** the kernel
        (unsigned for groups,  signed for positions).
    grp_orig / pos_orig
        Dtypes the Python API guarantees to the caller.

    Returns
    -------
    Casted ndarray or tuple of ndarrays in the same shape as *raw_out*.
    """

    def _restore(role: str, arr: np.ndarray) -> np.ndarray:
        """Cast one array according to its semantic role."""
        if role == "grp" and arr.dtype == grp_t:  # unsigned → caller dtype
            return arr.astype(
                grp_orig, copy=False
            )  # zero-copy view :contentReference[oaicite:0]{index=0}
        if role == "pos" and arr.dtype == pos_t:  # signed   → caller dtype
            return arr.astype(
                pos_orig, copy=False
            )  # zero-copy view :contentReference[oaicite:1]{index=1}
        return arr  # strand / dist stay

    # ── SINGLE-OUTPUT KERNEL ────────────────────────────────────────────
    if len(roles) == 1:  # API says 1 array
        if not isinstance(
            raw_out, np.ndarray
        ):  # ndarray check :contentReference[oaicite:2]{index=2}
            raise TypeError(
                f"{prefix!r} should return one ndarray; got {type(raw_out).__name__}"
            )
        return _restore(roles[0], raw_out)

    # ── MULTI-OUTPUT KERNEL ─────────────────────────────────────────────
    if not isinstance(raw_out, tuple):  # stray array → wrap
        raw_out = (raw_out,)
    if len(raw_out) != len(roles):  # arity guard
        raise ValueError(
            f"{prefix!r} declared {len(roles)} outputs but kernel returned "
            f"{len(raw_out)}"
        )

    # zip stops at the shorter iterable; arity already checked above :contentReference[oaicite:3]{index=3}
    return tuple(_restore(role, arr) for role, arr in zip(roles, raw_out))


def _dispatch_map_global_binary(
    prefix: str,
    ex_tx, ex_local_start, ex_local_end, ex_fwd,                # left (exons)
    q_tx, q_start, q_end, q_fwd,                                # right (queries)
    ex_chr_code, ex_genome_start, ex_genome_end,                # extra
    **extra_kw
):
    """
    Special-purpose dispatch function for map_to_global, 
    so we don't rely on _dispatch_binary's argument ordering.

    Parameters
    ----------
    prefix : str
        Rust kernel prefix, e.g. "map_to_global_numpy".
    ex_tx, ex_local_start, ex_local_end : 1D arrays
        The group, start, end arrays for exon (left) side.
    ex_fwd : 1D bool array
        Exon side's strand flags.
    q_tx, q_start, q_end : 1D arrays
        The group, start, end arrays for query (right) side.
    q_fwd : 1D bool array
        Query side's strand flags.
    ex_chr_code, ex_genome_start, ex_genome_end : 1D arrays
        Additional arrays needed by the Rust function.
    **extra_kw
        Extra arguments if needed (usually none).

    Returns
    -------
    keep_idx, out_start, out_end, out_strand_bool : NDArray
        The 4 arrays returned by the "map_to_global_numpy" Rust kernel.
    """
    # ----------------------------------------------------------------------
    # 0. Helpers: same ones you used in _dispatch_binary
    # ----------------------------------------------------------------------
    def check_array_lengths(*arrays):
        # example length check used in your code
        lengths = [len(a) for a in arrays if a is not None]
        if not lengths:
            return 0
        if len(set(lengths)) > 1:
            raise ValueError(f"Array length mismatch: {lengths}")
        return lengths[0]

    def validate_groups(length, arr):
        # If groups is None, produce a zero array, etc. 
        if arr is None:
            return np.zeros(length, dtype=np.uint32)
        if len(arr) != length:
            raise ValueError("Group array length mismatch")
        return arr

    def _common_integer_dtype(*arrays):
        """Pick the smallest integer dtype that can hold all array min/max."""
        # same logic you had in _dispatch_binary
        # or do the short version for brevity
        return np.result_type(*[a.dtype for a in arrays if a is not None])

    # ----------------------------------------------------------------------
    # 1. Basic array-length checks
    # ----------------------------------------------------------------------
    left_len = check_array_lengths(ex_tx, ex_local_start, ex_local_end, ex_fwd,
                                   ex_chr_code, ex_genome_start, ex_genome_end)
    right_len = check_array_lengths(q_tx, q_start, q_end, q_fwd)

    # validate group arrays, etc.
    ex_tx_valid = validate_groups(left_len, ex_tx)
    q_tx_valid  = validate_groups(right_len, q_tx)

    # ----------------------------------------------------------------------
    # 2. Identify original dtypes (for final cast)
    # ----------------------------------------------------------------------
    grp_orig = ex_tx_valid.dtype  # a guess if ex_tx != None
    pos_orig = ex_local_start.dtype  # assume ex_local_start’s dtype is your reference

    # ----------------------------------------------------------------------
    # 3. Pick tightest dtype for groups + positions
    # ----------------------------------------------------------------------
    grp_tmp = _common_integer_dtype(ex_tx_valid, q_tx_valid)
    pos_tmp = _common_integer_dtype(
        ex_local_start, ex_local_end, q_start, q_end, 
        ex_chr_code, ex_genome_start, ex_genome_end
    )

    # optional slack check
    slack = extra_kw.get("slack", 0)
    # skip if you don't need it
    # (like check_min_max_with_slack(...))

    # ----------------------------------------------------------------------
    # 4. Resolve which Rust function to call, same as _dispatch_binary
    # ----------------------------------------------------------------------
    # e.g. rust_fn, grp_t, pos_t = _resolve_rust_fn(prefix, grp_tmp, pos_tmp)
    rust_fn, grp_t, pos_t = _resolve_rust_fn(prefix, grp_tmp, pos_tmp)

    # We'll define what your kernel returns
    roles = RETURN_SIGNATURES[prefix]  # e.g. ("grp", "pos", "pos", "strand")

    # ----------------------------------------------------------------------
    # 5. Convert arrays to the temporary dtype
    # ----------------------------------------------------------------------
    ex_tx_t         = ex_tx_valid.astype(grp_t, copy=False)
    ex_local_start_t= ex_local_start.astype(pos_t, copy=False)
    ex_local_end_t  = ex_local_end.astype(pos_t, copy=False)
    ex_fwd_t        = ex_fwd  # bool doesn't need casting for Rust

    q_tx_t          = q_tx_valid.astype(grp_t, copy=False)
    q_start_t       = q_start.astype(pos_t, copy=False)
    q_end_t         = q_end.astype(pos_t, copy=False)
    q_fwd_t         = q_fwd  # bool

    ex_chr_code_t   = ex_chr_code.astype(grp_t, copy=False)
    ex_genome_start_t = ex_genome_start.astype(pos_t, copy=False)
    ex_genome_end_t   = ex_genome_end.astype(pos_t, copy=False)

    # ----------------------------------------------------------------------
    # 6. Call the Rust function
    # ----------------------------------------------------------------------
    raw_out = rust_fn(
        # left triple
        ex_tx_t, ex_local_start_t, ex_local_end_t,
        # right triple
        q_tx_t, q_start_t, q_end_t,
        # extras
        ex_chr_code_t, ex_genome_start_t, ex_genome_end_t,
        ex_fwd_t, q_fwd_t,
    )

    # ----------------------------------------------------------------------
    # 7. Cast the outputs back
    # same logic your cast_kernel_outputs uses
    # roles might be ("grp", "pos", "pos", "strand")
    return cast_kernel_outputs(prefix, raw_out, roles, grp_t, pos_t, grp_orig, pos_orig)
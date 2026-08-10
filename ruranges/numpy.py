import importlib
from typing import Any, Callable, Literal, Sequence, TypeVar
import numpy as np
from numpy.typing import NDArray

from importlib.metadata import version
__version__: str = version("ruranges")


# Define a type variable for groups that allows only int8, int16, or int32.
GroupIdInt = TypeVar("GroupIdInt", np.int8, np.int16, np.int32)

# Define another type variable for the range arrays (starts/ends), which can be any integer.
RangeInt = TypeVar("RangeInt", bound=np.integer)


# dtype-suffix map shared by every operation
# (group_dtype, range_dtype)  →  (suffix, group_target_dtype, range_target_dtype)
_SUFFIX_TABLE = {
    (np.dtype(np.uint32), None): ("u32", np.uint32, None),
    (None, np.dtype(np.int32)): ("i32", None, np.int32),
    (None, np.dtype(np.int64)): ("i64", None, np.int64),
    (np.dtype(np.uint32), np.dtype(np.int32)): ("u32_i32", np.uint32, np.int32),
    (np.dtype(np.uint32), np.dtype(np.int64)): ("u32_i64", np.uint32, np.int64),
}

RETURN_SIGNATURES: dict[str, tuple[str, ...]] = {
    "chromsweep_numpy": ("index", "index"),
    "nearest_numpy": ("index", "index", "dist"),
    "subtract_numpy": ("index", "coord", "coord"),
    "complement_overlaps_numpy": ("index",),
    "count_overlaps_numpy": ("count",),
    "sort_groups_numpy": ("index",),
    "sort_intervals_numpy": ("index",),
    "cluster_numpy": ("label", "index"),
    "max_disjoint_numpy": ("index",),
    "merge_numpy": ("index", "coord", "coord", "count"),
    "window_numpy": ("index", "coord", "coord"),
    "tile_numpy": ("index", "coord", "coord", "fraction"),
    "complement_numpy": ("group", "coord", "coord", "index"),
    "boundary_numpy": ("index", "coord", "coord", "count"),
    "spliced_subsequence_numpy": ("index", "coord", "coord", "_strand"),
    "spliced_subsequence_multi_numpy": ("index", "coord", "coord", "_strand"),
    "split_numpy": ("index", "coord", "coord"),
    "extend_numpy": ("coord", "coord"),
    "genome_bounds_numpy": ("index", "coord", "coord"),
    "group_cumsum_numpy": ("index", "metric", "metric"),
    "map_to_global_numpy": ("index", "coord", "coord", "strand"),
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
    sort_output: bool = True,
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
        sort_output=sort_output,
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
    sort_output: bool = True,
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
        sort_output=sort_output,
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
    sort_output: bool = True,
    ties: Literal["all", "first"] = "all",
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
    ties
        How many neighbours to report when several sit at the same distance.
        • ``"all"`` (default) – all of them, at each of the *k* distances.
        • ``"first"`` – one of them per distance, so at most *k* rows per
          query. Which one is unspecified, but the same input gives the same
          answer every time. With ``include_overlaps`` this is the difference
          between reporting every interval a query overlaps and reporting one,
          which on dense inputs is orders of magnitude of output.

    Returns
    -------
    idx1, idx2, dist
        *idx1* / *idx2* are ``uint32`` indices into the first / second
        interval sets; *dist* is the coordinate-typed distance between each
        pair.

    Raises
    ------
    ValueError
        If the input lengths don’t match, only one of ``groups`` /
        ``groups2`` is supplied, or ``ties`` is neither ``"all"`` nor
        ``"first"``.
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
        sort_output=sort_output,
        ties=ties,
    )


def subtract(
    starts: NDArray[RangeInt],
    ends: NDArray[RangeInt],
    starts2: NDArray[RangeInt],
    ends2: NDArray[RangeInt],
    groups: NDArray[GroupIdInt] | None = None,
    groups2: NDArray[GroupIdInt] | None = None,
    sort_output: bool = True,
) -> tuple[NDArray[GroupIdInt], NDArray[RangeInt], NDArray[RangeInt]]:
    return _dispatch_binary(
        "subtract_numpy",
        groups,
        starts,
        ends,
        groups2,
        starts2,
        ends2,
        sort_output=sort_output,
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
    sort_output: bool = True,
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
        sort_output=sort_output,
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
    *,
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
        globally.  **Keyword-only**: all three arrays are same-length integer
        arrays, so passing them in the wrong order is otherwise undetectable —
        which is exactly how ``pyranges1`` came to sort by ``End`` before
        ``Start`` for two releases.
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


def natural_rank(values: Sequence[str]) -> NDArray[np.uint32]:
    """
    Return the position each value would occupy in natural order.

    Natural order reads a string as alternating text and number runs: text runs
    compare as text, number runs compare as numbers.  So ``t2`` sorts before
    ``t10``, and ``chr9 < chr10 < chrM``.

    This is the ordering ``sort_ranges`` applies to string keys when
    ``natsort=True``, and it is defined once — here — so that every library
    built on ruranges answers the same way.  It reproduces Python's
    ``natsort.natsorted`` under default settings except that it does not
    Unicode-normalise, so strings differing only in composed non-ASCII
    characters may order differently; see ``ruranges_core::ranks``.

    Parameters
    ----------
    values
        The **distinct** values of a key column, e.g. the ``uniques`` returned
        by :func:`pandas.factorize`.  Ranking distinct values rather than rows
        is what makes natural ordering affordable: the cost is per distinct
        value, not per row.

    Returns
    -------
    ranks : NDArray[np.uint32]
        ``ranks[i]`` is the sorted position of ``values[i]``.  Every element
        gets its own rank; values that compare equal (``t1`` and ``t01`` do,
        since leading zeros vanish in a number) keep their input order.

    Examples
    --------
    >>> natural_rank(["chr10", "chr2", "chrX", "chr1"])
    array([2, 1, 3, 0], dtype=uint32)
    """
    rust_mod = importlib.import_module(".ruranges", package="ruranges")
    return rust_mod.natural_rank_numpy(list(values))


def lexical_rank(values: Sequence[str]) -> NDArray[np.uint32]:
    """
    Return the position each value would occupy in byte-lexical order.

    The counterpart to :func:`natural_rank` for ``natsort=False`` and for
    generic (non-genomic) frames, which have no chromosome column and so no
    reason to order strings naturally.  Here ``t10`` sorts before ``t9``.

    Parameters
    ----------
    values
        The **distinct** values of a key column.

    Returns
    -------
    ranks : NDArray[np.uint32]
        ``ranks[i]`` is the sorted position of ``values[i]``.

    Examples
    --------
    >>> lexical_rank(["chr10", "chr2", "chr1"])
    array([1, 2, 0], dtype=uint32)
    """
    rust_mod = importlib.import_module(".ruranges", package="ruranges")
    return rust_mod.lexical_rank_numpy(list(values))


def fold_ranks(
    group: NDArray[np.uint32],
    group_cardinality: int,
    codes: NDArray[np.uint32],
    codes_cardinality: int,
) -> tuple[NDArray[np.uint32], int]:
    """
    Fold one key column's codes into a running group id.

    A multi-key sort codes each key column separately and then collapses all but
    the last two into the single ``groups`` argument :func:`sort_intervals`
    takes.  This is that collapse, one column at a time.

    Parameters
    ----------
    group
        Codes of every key folded so far, one per row.
    group_cardinality
        Number of distinct values in *group*.
    codes
        Codes of the next key, one per row.
    codes_cardinality
        Number of distinct values in *codes*.

    Returns
    -------
    folded : NDArray[np.uint32]
        Codes ordering rows by the old key first and the new key second.
    cardinality : int
        Number of distinct values in *folded*, to pass back in as
        *group_cardinality* on the next fold.

    Notes
    -----
    ``group * codes_cardinality + codes`` overflows ``uint32`` once the keys are
    wide enough; when it would, the combined key is re-densified instead, which
    preserves the order and caps cardinality at the row count.

    Examples
    --------
    >>> folded, cardinality = fold_ranks(
    ...     np.array([0, 0, 1, 1], dtype=np.uint32), 2,
    ...     np.array([1, 0, 1, 0], dtype=np.uint32), 2,
    ... )
    >>> folded
    array([1, 0, 3, 2], dtype=uint32)
    >>> cardinality
    4
    """
    rust_mod = importlib.import_module(".ruranges", package="ruranges")
    return rust_mod.fold_ranks_numpy(
        _normalize_groups(group, name="group"),
        int(group_cardinality),
        _normalize_groups(codes, name="codes"),
        int(codes_cardinality),
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
    return _dispatch_groups_only("sort_groups_numpy", groups)


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
    sort_output: bool = True,
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
        sort_output=sort_output,
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
        `out_chrs` holds the group each gap belongs to and is the reliable key
        for attribution.  `out_idx` holds the 0-based index of *one* input
        interval belonging to that group; it identifies the group, not a
        particular member, so callers must not depend on which member is
        reported.  Prefer `out_chrs` when mapping gaps back to per-group values.

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
    Collapse each group's intervals into its outer boundary.

    Returns one row per group, spanning that group's minimum start to its
    maximum end.  Gaps inside a group do not split the result: a group always
    yields exactly one row.

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
        ``indices`` holds the position of *one* input interval belonging to each
        group.  It identifies the group, not a particular member, so callers
        must not depend on which member is reported; use it only to read values
        that are constant across the group, such as grouping columns.
        ``counts`` holds the number of input intervals in each group.
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
    groups: NDArray[GroupIdInt] | None = None,
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
        groups = _groups_or_arange(groups, len(starts)),
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
        groups = None,
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
    sort_output: bool = True,
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
        sort_output=sort_output,
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


# ---- dtype normalization and dispatch ---------------------------------
_CORE_GROUP_DTYPE = np.dtype(np.uint32)
_GROUPLESS_PREFIXES = {"tile_numpy"}
_BINARY_SCALAR_PREFIXES = {
    "chromsweep_numpy",
    "nearest_numpy",
    "complement_overlaps_numpy",
    "count_overlaps_numpy",
}
_UNARY_POS_SCALAR_KEYS = {"slack", "window_size", "tile_size", "ext_3", "ext_5"}


def _resolve_rust_fn(
    prefix: str,
    grp_dt: np.dtype | None,
    pos_dt: np.dtype | None,
) -> tuple[Callable, np.dtype | None, np.dtype | None]:
    try:
        suffix, tgt_grp, tgt_pos = _SUFFIX_TABLE[(grp_dt, pos_dt)]
    except KeyError as exc:
        raise TypeError(f"Unsupported dtype pair: {grp_dt}, {pos_dt}") from exc
    rust_mod = importlib.import_module(".ruranges", package="ruranges")
    return getattr(rust_mod, f"{prefix}_{suffix}"), tgt_grp, tgt_pos


def _as_1d_array(a: NDArray | Sequence[Any], *, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be a 1-D array.")
    return arr


def _int_bounds(arr: np.ndarray, *, name: str) -> tuple[int, int]:
    kind = arr.dtype.kind
    if arr.size == 0:
        return 0, 0
    if kind in {"i", "u", "b"}:
        return int(arr.min()), int(arr.max())
    raise TypeError(f"`{name}` must be integer-like (int/uint/bool), got {arr.dtype}.")


def _normalize_groups(groups: NDArray | Sequence[Any], *, name: str) -> np.ndarray:
    arr = _as_1d_array(groups, name=name)
    lo, hi = _int_bounds(arr, name=name)
    if lo < 0:
        raise TypeError(f"`{name}` must be non-negative.")
    if hi > np.iinfo(np.uint32).max:
        raise OverflowError(f"`{name}` values exceed uint32 range.")
    return arr.astype(np.uint32, copy=False)


def _to_signed(arr: np.ndarray, *, name: str, target: np.dtype) -> np.ndarray:
    lo, hi = _int_bounds(arr, name=name)
    info = np.iinfo(target)
    if lo < int(info.min) or hi > int(info.max):
        raise OverflowError(f"`{name}` values exceed {target} range.")
    return arr.astype(target, copy=False)


def _coerce_int(v: Any, *, name: str) -> int:
    x = np.asarray(v)
    if x.ndim != 0:
        raise TypeError(f"`{name}` must be scalar.")
    val = x.item()
    if isinstance(val, (np.integer, int, np.bool_)):
        val = int(val)
    else:
        raise TypeError(f"`{name}` must be an integer.")
    return val


def _choose_pos_dtype(min_value: int, max_value: int) -> np.dtype:
    if min_value >= np.iinfo(np.int32).min and max_value <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    if min_value >= np.iinfo(np.int64).min and max_value <= np.iinfo(np.int64).max:
        return np.dtype(np.int64)
    raise OverflowError("Coordinate values exceed int64 range.")


def _lossless_cast(arr: np.ndarray, target: np.dtype) -> np.ndarray | None:
    target = np.dtype(target)
    try:
        cast = arr.astype(target, copy=False)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if np.array_equal(arr.astype(object), cast.astype(object)):
            return cast
    except Exception:
        return None
    return None


def _restore_outputs(
    prefix: str,
    raw_out: Any,
    roles: tuple[str, ...],
    *,
    grp_orig: np.dtype,
    pos_orig: np.dtype,
) -> Any:
    def _restore(role: str, arr: np.ndarray) -> np.ndarray:
        if role == "group":
            cast = _lossless_cast(arr, grp_orig)
            return cast if cast is not None else arr
        if role == "coord":
            cast = _lossless_cast(arr, pos_orig)
            if cast is not None:
                return cast
            return arr
        if role in {"dist", "metric"}:
            cast = _lossless_cast(arr, pos_orig)
            return cast if cast is not None else arr
        return arr

    if len(roles) == 1:
        if not isinstance(raw_out, np.ndarray):
            raise TypeError(f"{prefix!r} should return one ndarray; got {type(raw_out).__name__}")
        return _restore(roles[0], raw_out)

    if not isinstance(raw_out, tuple):
        raw_out = (raw_out,)
    if len(raw_out) != len(roles):
        raise ValueError(
            f"{prefix!r} declared {len(roles)} outputs but kernel returned {len(raw_out)}"
        )
    return tuple(_restore(role, arr) for role, arr in zip(roles, raw_out))


def _dispatch_unary(
    prefix: str,
    starts: NDArray,
    ends: NDArray,
    groups: NDArray | None = None,
    *extra_pos_args: Any,
    **extra_kw: Any,
) -> Any:
    roles = RETURN_SIGNATURES[prefix]
    starts_arr = _as_1d_array(starts, name="starts")
    ends_arr = _as_1d_array(ends, name="ends")
    if len(starts_arr) != len(ends_arr):
        raise ValueError("`starts` and `ends` must have the same length.")

    scalar_vals: list[int] = []
    pos_extra_arrays: list[tuple[str, np.ndarray]] = []
    for k in _UNARY_POS_SCALAR_KEYS.intersection(extra_kw.keys()):
        v = _coerce_int(extra_kw[k], name=k)
        extra_kw[k] = v
        scalar_vals.append(v)

    if prefix == "complement_numpy":
        chrom_ids = _normalize_groups(extra_kw["chrom_len_ids"], name="chrom_len_ids")
        extra_kw["chrom_len_ids"] = chrom_ids
        chrom_lens = _as_1d_array(extra_kw["chrom_lens"], name="chrom_lens")
        pos_extra_arrays.append(("chrom_lens", chrom_lens))
    elif prefix == "genome_bounds_numpy":
        chrom_lengths = _as_1d_array(extra_kw["chrom_lengths"], name="chrom_lengths")
        pos_extra_arrays.append(("chrom_lengths", chrom_lengths))
    elif prefix == "spliced_subsequence_multi_numpy":
        slice_starts = _as_1d_array(extra_kw["slice_starts"], name="slice_starts")
        slice_ends = _as_1d_array(extra_kw["slice_ends"], name="slice_ends")
        pos_extra_arrays.extend([("slice_starts", slice_starts), ("slice_ends", slice_ends)])

    mins_maxes = [_int_bounds(starts_arr, name="starts"), _int_bounds(ends_arr, name="ends")]
    mins_maxes.extend(_int_bounds(arr, name=name) for name, arr in pos_extra_arrays)
    min_v = min(m for m, _ in mins_maxes)
    max_v = max(m for _, m in mins_maxes)
    pos_t = _choose_pos_dtype(min(min_v, min(scalar_vals, default=0)), max(max_v, max(scalar_vals, default=0)))

    starts_u = _to_signed(starts_arr, name="starts", target=pos_t)
    ends_u = _to_signed(ends_arr, name="ends", target=pos_t)
    for name, arr in pos_extra_arrays:
        extra_kw[name] = _to_signed(arr, name=name, target=pos_t)

    pos_orig = starts_arr.dtype
    grp_orig = np.asarray(groups).dtype if groups is not None else np.dtype(np.uint32)

    if prefix in _GROUPLESS_PREFIXES:
        rust_fn, _, _ = _resolve_rust_fn(prefix, None, pos_t)
        raw = rust_fn(starts_u, ends_u, *extra_pos_args, **extra_kw)
    else:
        g = np.zeros(len(starts_arr), dtype=np.uint32) if groups is None else _normalize_groups(groups, name="groups")
        if len(g) != len(starts_arr):
            raise ValueError("`groups` must have same length as coordinates.")
        rust_fn, _, _ = _resolve_rust_fn(prefix, _CORE_GROUP_DTYPE, pos_t)
        raw = rust_fn(g, starts_u, ends_u, *extra_pos_args, **extra_kw)

    return _restore_outputs(prefix, raw, roles, grp_orig=grp_orig, pos_orig=pos_orig)


def _dispatch_binary(
    prefix: str,
    groups: NDArray | None,
    starts: NDArray,
    ends: NDArray,
    groups2: NDArray | None,
    starts2: NDArray,
    ends2: NDArray,
    *extra_pos: Any,
    **extra_kw: Any,
) -> Any:
    roles = RETURN_SIGNATURES[prefix]
    s1 = _as_1d_array(starts, name="starts")
    e1 = _as_1d_array(ends, name="ends")
    s2 = _as_1d_array(starts2, name="starts2")
    e2 = _as_1d_array(ends2, name="ends2")
    if len(s1) != len(e1):
        raise ValueError("`starts` and `ends` must have the same length.")
    if len(s2) != len(e2):
        raise ValueError("`starts2` and `ends2` must have the same length.")

    extra_pos = list(extra_pos)
    scalar_vals: list[int] = []
    if prefix in _BINARY_SCALAR_PREFIXES and extra_pos:
        v = _coerce_int(extra_pos[0], name="slack")
        extra_pos[0] = v
        scalar_vals.append(v)

    bounds = [
        _int_bounds(s1, name="starts"),
        _int_bounds(e1, name="ends"),
        _int_bounds(s2, name="starts2"),
        _int_bounds(e2, name="ends2"),
    ]
    min_v = min(m for m, _ in bounds)
    max_v = max(m for _, m in bounds)
    pos_t = _choose_pos_dtype(min(min_v, min(scalar_vals, default=0)), max(max_v, max(scalar_vals, default=0)))

    s1u = _to_signed(s1, name="starts", target=pos_t)
    e1u = _to_signed(e1, name="ends", target=pos_t)
    s2u = _to_signed(s2, name="starts2", target=pos_t)
    e2u = _to_signed(e2, name="ends2", target=pos_t)

    g1 = np.zeros(len(s1), dtype=np.uint32) if groups is None else _normalize_groups(groups, name="groups")
    g2 = np.zeros(len(s2), dtype=np.uint32) if groups2 is None else _normalize_groups(groups2, name="groups2")
    if len(g1) != len(s1):
        raise ValueError("`groups` must have same length as first coordinate arrays.")
    if len(g2) != len(s2):
        raise ValueError("`groups2` must have same length as second coordinate arrays.")

    grp_orig = np.asarray(groups).dtype if groups is not None else np.dtype(np.uint32)
    pos_orig = s1.dtype
    rust_fn, _, _ = _resolve_rust_fn(prefix, _CORE_GROUP_DTYPE, pos_t)
    raw = rust_fn(g1, s1u, e1u, g2, s2u, e2u, *extra_pos, **extra_kw)
    return _restore_outputs(prefix, raw, roles, grp_orig=grp_orig, pos_orig=pos_orig)


def _dispatch_groups_only(prefix: str, groups: NDArray | Sequence[Any]) -> NDArray:
    g = _normalize_groups(groups, name="groups")
    grp_orig = np.asarray(groups).dtype
    rust_fn, _, _ = _resolve_rust_fn(prefix, _CORE_GROUP_DTYPE, None)
    raw = rust_fn(g)
    if not isinstance(raw, np.ndarray):
        raise TypeError(f"{prefix!r} should return one ndarray; got {type(raw).__name__}")
    return raw


def _dispatch_map_global_binary(
    prefix: str,
    ex_tx, ex_local_start, ex_local_end, ex_fwd,
    q_tx, q_start, q_end, q_fwd,
    ex_chr_code, ex_genome_start, ex_genome_end,
    **extra_kw
):
    roles = RETURN_SIGNATURES[prefix]
    ex_start = _as_1d_array(ex_local_start, name="starts")
    ex_end = _as_1d_array(ex_local_end, name="ends")
    q_start_arr = _as_1d_array(q_start, name="starts2")
    q_end_arr = _as_1d_array(q_end, name="ends2")
    ex_genome_start_arr = _as_1d_array(ex_genome_start, name="genome_start2")
    ex_genome_end_arr = _as_1d_array(ex_genome_end, name="genome_end2")
    ex_fwd_arr = _as_1d_array(ex_fwd, name="strand")
    q_fwd_arr = _as_1d_array(q_fwd, name="strand2")

    left_len = len(ex_start)
    right_len = len(q_start_arr)
    if len(ex_end) != left_len:
        raise ValueError("`starts` and `ends` must have same length.")
    if len(q_end_arr) != right_len:
        raise ValueError("`starts2` and `ends2` must have same length.")
    if len(ex_genome_start_arr) != left_len or len(ex_genome_end_arr) != left_len:
        raise ValueError("Genome start/end arrays must match exon length.")
    if len(ex_fwd_arr) != left_len or len(q_fwd_arr) != right_len:
        raise ValueError("Strand arrays must match corresponding table lengths.")

    bounds = [
        _int_bounds(ex_start, name="starts"),
        _int_bounds(ex_end, name="ends"),
        _int_bounds(q_start_arr, name="starts2"),
        _int_bounds(q_end_arr, name="ends2"),
        _int_bounds(ex_genome_start_arr, name="genome_start2"),
        _int_bounds(ex_genome_end_arr, name="genome_end2"),
    ]
    min_v = min(m for m, _ in bounds)
    max_v = max(m for _, m in bounds)
    pos_t = _choose_pos_dtype(min_v, max_v)

    ex_tx_u = np.zeros(left_len, dtype=np.uint32) if ex_tx is None else _normalize_groups(ex_tx, name="groups")
    q_tx_u = np.zeros(right_len, dtype=np.uint32) if q_tx is None else _normalize_groups(q_tx, name="groups2")
    ex_chr_u = _normalize_groups(ex_chr_code, name="chr_code2")
    if len(ex_tx_u) != left_len or len(ex_chr_u) != left_len:
        raise ValueError("Exon-side group arrays must match exon length.")
    if len(q_tx_u) != right_len:
        raise ValueError("Query-side group arrays must match query length.")

    ex_start_u = _to_signed(ex_start, name="starts", target=pos_t)
    ex_end_u = _to_signed(ex_end, name="ends", target=pos_t)
    q_start_u = _to_signed(q_start_arr, name="starts2", target=pos_t)
    q_end_u = _to_signed(q_end_arr, name="ends2", target=pos_t)
    ex_genome_start_u = _to_signed(ex_genome_start_arr, name="genome_start2", target=pos_t)
    ex_genome_end_u = _to_signed(ex_genome_end_arr, name="genome_end2", target=pos_t)

    grp_orig = np.asarray(ex_tx).dtype if ex_tx is not None else np.dtype(np.uint32)
    pos_orig = ex_start.dtype
    rust_fn, _, _ = _resolve_rust_fn(prefix, _CORE_GROUP_DTYPE, pos_t)
    raw = rust_fn(
        ex_tx_u,
        ex_start_u,
        ex_end_u,
        q_tx_u,
        q_start_u,
        q_end_u,
        ex_chr_u,
        ex_genome_start_u,
        ex_genome_end_u,
        ex_fwd_arr.astype(np.bool_, copy=False),
        q_fwd_arr.astype(np.bool_, copy=False),
        **extra_kw,
    )
    return _restore_outputs(prefix, raw, roles, grp_orig=grp_orig, pos_orig=pos_orig)

def _groups_or_arange(groups, n, dtype=np.uint8):
    if groups is None:
        return np.arange(n, dtype=dtype)
    return groups

# ruranges-py - blazing-fast interval algebra for NumPy

ruranges-py is the Python bindings package for `ruranges-core`, a separate Rust crate/repo that implements common genomic / interval algorithms at native speed. All public functions accept and return plain NumPy arrays so you can drop the results straight into your existing Python data-science stack.

---

## Why ruranges-py?

* Speed: heavy kernels in Rust compiled with --release.
* Zero copy: results are numpy views whenever possible.
* Flexible dtypes: integer-like inputs are normalized to a compact kernel core (`uint32` groups, `int32`/`int64` coordinates) and converted back when possible.
* Stateless: plain functions, no classes.

---

## Installation

```bash
pip install ruranges-py                # PyPI
# or
pip install git+https://github.com/your-org/ruranges-py.git
```

### Development environment (from local checkout)

```bash
cd ~/code
git clone <your-remote>/ruranges-py

cd ~/code/ruranges-py
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install maturin
maturin develop --release
```

Quick check:
```bash
python -c "import ruranges; print(ruranges.__version__)"
```

---

## Cheat sheet

| Category              | Function                                   | What it does                                    |
| --------------------- | ------------------------------------------ | ----------------------------------------------- |
| Overlap and proximity | overlaps                                   | all overlapping pairs between two sets          |
|                       | nearest                                    | k nearest intervals with optional strand filter |
|                       | count\_overlaps                            | how many rows in B overlap each row in A        |
| Set algebra           | subtract                                   | A minus B                                       |
|                       | complement                                 | gaps within chromosome bounds                   |
|                       | merge, cluster, max\_disjoint              | collapse or filter overlaps                     |
| Sorting               | sort\_intervals                            | row order for a (group, start, end) sort        |
|                       | natural\_rank, lexical\_rank               | order a key column's distinct string values     |
|                       | fold\_ranks                                | collapse several coded keys into one group id   |
| Utility               | window, tile, extend, ...                  | assorted helpers                                |

Below are the three most common calls: overlaps, nearest, subtract.

---

## 1. overlaps

Simple example:

```python
import pandas as pd
import numpy as np
import ruranges

df1 = pd.DataFrame({
    "chr": ["chr1", "chr1", "chr2"],
    "strand": ["+", "+", "-"],
    "start": [1, 10, 30],
    "end":   [5, 15, 35],
})

df2 = pd.DataFrame({
    "chr": ["chr1", "chr2", "chr2"],
    "strand": ["+", "-", "-"],
    "start": [3, -50, 0],
    "end":   [6, 50, 2],
})

print("Inputs:")

print(df1)
print(df2)


# Vectorised: concatenate, then ngroup
combo = pd.concat([df1[["chr", "strand"]], df2[["chr", "strand"]]], ignore_index=True)
labels = combo.groupby(["chr", "strand"], sort=False).ngroup().astype(np.uint32).to_numpy()

groups  = labels[:len(df1)]
groups2 = labels[len(df1):]

idx1, idx2 = ruranges.numpy.overlaps(
    starts=df1["start"].to_numpy(np.int32),
    ends=df1["end"].to_numpy(np.int32),
    starts2=df2["start"].to_numpy(np.int32),
    ends2=df2["end"].to_numpy(np.int32),
    groups=groups,
    groups2=groups2,
)


print("Output:")
print(idx1, idx2)

print("Extracts rows:")
print(df1.iloc[idx1])
print(df2.iloc[idx2])

# Inputs:
#     chr strand  start  end
# 0  chr1      +      1    5
# 1  chr1      +     10   15
# 2  chr2      -     30   35
#     chr strand  start  end
# 0  chr1      +      3    6
# 1  chr2      -    -50   50
# 2  chr2      -      0    2
# Output:
# [0 2] [0 1]
# Extracts rows:
#     chr strand  start  end
# 0  chr1      +      1    5
# 2  chr2      -     30   35
#     chr strand  start  end
# 0  chr1      +      3    6
# 1  chr2      -    -50   50
```

## 2. nearest

```python
import numpy as np
import ruranges

starts  = np.array([1, 10, 30], dtype=np.int32)
ends    = np.array([5, 15, 35], dtype=np.int32)
starts2 = np.array([3, 20, 28], dtype=np.int32)
ends2   = np.array([6, 25, 32], dtype=np.int32)

idx1, idx2, dist = ruranges.numpy.nearest(
    starts=starts, ends=ends,
    starts2=starts2, ends2=ends2,
    k=2,
    include_overlaps=False,
    direction="any",
)

for a, b, d in zip(idx1, idx2, dist):
    print(f"query[{a}] <-> ref[{b}] : {d} bp")

# query[0] <-> ref[1] : 16 bp
# query[0] <-> ref[2] : 24 bp
# query[1] <-> ref[0] : 5 bp
# query[1] <-> ref[1] : 6 bp
# query[2] <-> ref[1] : 6 bp
# query[2] <-> ref[0] : 25 bp
```

Set direction to "forward" or "backward" to restrict to one side.

---

## 3. subtract

```python
import numpy as np
import ruranges

starts  = np.array([0, 10], dtype=np.int32)
ends    = np.array([10, 20], dtype=np.int32)
starts2 = np.array([5, 12], dtype=np.int32)
ends2   = np.array([15, 18], dtype=np.int32)

idx_keep, sub_starts, sub_ends = ruranges.numpy.subtract(
    starts, ends,
    starts2, ends2,
)

print(idx_keep) 
print(sub_starts)
print(sub_ends)
# [0 1]
# [ 0 18]
# [ 5 20]
```

Because interval 1 is broken into two pieces it appears twice in idx\_keep.

---

## 4. Sorting by string keys

`sort_intervals` sorts by `(group, start, end)`, where `group` is an integer. To
sort by *string* keys — chromosome names, transcript ids — code each key column
first, then fold the codes into one group id:

```python
import numpy as np
import pandas as pd
import ruranges

chromosome = pd.array(["chr10", "chr2", "chr2", "chr10"], dtype="str")
start = np.array([100, 300, 100, 300], dtype=np.int32)
end = np.array([150, 350, 150, 350], dtype=np.int32)

# 1. Code the key column: factorize the rows, rank the *distinct* values.
codes, uniques = pd.factorize(chromosome, sort=False)
rank = ruranges.numpy.natural_rank(uniques)
group = rank[codes].astype(np.uint32)

# 2. One kernel call, one gather.
order = ruranges.numpy.sort_intervals(start, end, groups=group)
print(chromosome[order])
# ['chr2', 'chr2', 'chr10', 'chr10']
```

Natural order puts `chr2` before `chr10`; swap in `lexical_rank` for plain byte
order. Ranking the distinct values rather than the rows is what makes this
affordable — the cost is per distinct value, so 25 chromosome names cost
nothing regardless of row count.

With more than one key column, fold each column's codes into the running group
id before the kernel call:

```python
group, cardinality = ruranges.numpy.fold_ranks(
    group, len(uniques), strand_codes, 2
)
```

`fold_ranks` handles the `uint32` overflow that a plain
`group * cardinality + codes` would hit once the keys are wide enough.

---

## FAQ

### Supported dtypes

* Groups: integer-like NumPy dtypes (`int*`, `uint*`, `bool`) are accepted if values are non-negative and fit in `uint32`.
* Coordinates: integer-like NumPy dtypes (`int*`, `uint*`, `bool`) are accepted. Inputs are normalized (offset-shifted when needed) to internal signed kernels.
* Internal kernel core: `group = uint32`, `position = int32 | int64`.

### Do I need sorted intervals?

No. Functions sort internally where needed and return index permutations so you can restore the original order.

### How to encode strand?

Any function that needs strand expects a boolean array: True for the minus strand, False for the plus strand.

---

## License

Apache 2.0. See LICENSE for details.

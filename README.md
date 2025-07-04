# ruranges - blazing-fast interval algebra for NumPy

ruranges is a thin Python wrapper around a set of Rust kernels that implement common genomic / interval algorithms at native speed. All public functions accept and return plain NumPy arrays so you can drop the results straight into your existing Python data-science stack.

---

## Why ruranges?

* Speed: heavy kernels in Rust compiled with --release.
* Zero copy: results are numpy views whenever possible.
* Flexible dtypes: unsigned int8/16/32/64 for group ids, signed ints for coordinates. The wrapper chooses the smallest safe dtype automatically.
* Stateless: plain functions, no classes.

---

## Installation

```bash
pip install ruranges                # PyPI
# or
pip install git+https://github.com/your-org/ruranges.git
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
| Utility               | sort\_intervals, window, tile, extend, ... | assorted helpers                                |

Below are the three most common calls: overlaps, nearest, subtract.

---

## 1. overlaps

Simple example:

```python
import pandas as pd
import numpy as np
from ruranges import overlaps

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

idx1, idx2 = overlaps(
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
from ruranges import nearest

starts  = np.array([1, 10, 30], dtype=np.int32)
ends    = np.array([5, 15, 35], dtype=np.int32)
starts2 = np.array([3, 20, 28], dtype=np.int32)
ends2   = np.array([6, 25, 32], dtype=np.int32)

idx1, idx2, dist = nearest(
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
from ruranges import subtract

starts  = np.array([0, 10], dtype=np.int32)
ends    = np.array([10, 20], dtype=np.int32)
starts2 = np.array([5, 12], dtype=np.int32)
ends2   = np.array([15, 18], dtype=np.int32)

idx_keep, sub_starts, sub_ends = subtract(
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

## FAQ

### Supported dtypes

* Groups: uint8, uint16, uint32, uint64
* Coordinates: int8, int16, int32, int64

### Do I need sorted intervals?

No. Functions sort internally where needed and return index permutations so you can restore the original order.

### How to encode strand?

Any function that needs strand expects a boolean array: True for the minus strand, False for the plus strand.

---

## License

Apache 2.0. See LICENSE for details.



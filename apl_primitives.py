#!/usr/bin/env python3
"""
APL Primitives Runtime Library

NumPy-based implementations of APL primitive functions and operators.
Used by transpiled APL code.

APL Index Origin: This library defaults to 1-based indexing (⎕IO←1) to match
traditional APL behavior. Set APL_INDEX_ORIGIN=0 environment variable for 0-based.

Usage:
    from apl_primitives import *

    result = iota(5)           # ⍳5 → [1, 2, 3, 4, 5]
    result = rho([2, 3], x)    # 2 3⍴X
    result = reduce_plus(x)    # +/X
"""

import os
import numpy as np
from typing import Any, List, Union, Optional, Callable
from functools import reduce as functools_reduce

# Index origin (⎕IO): 1 for traditional APL, 0 for zero-based
INDEX_ORIGIN = int(os.environ.get('APL_INDEX_ORIGIN', '1'))

# Type aliases
APLArray = Union[np.ndarray, List, int, float, str]
APLScalar = Union[int, float, str, bool]


# =============================================================================
# Helper Functions
# =============================================================================

def to_array(x: APLArray) -> np.ndarray:
    """Convert input to numpy array, handling various APL types."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, str):
        return np.array(list(x))
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return np.array([x])


def to_scalar(x: APLArray) -> APLScalar:
    """Extract scalar from single-element array."""
    arr = to_array(x)
    if arr.size == 1:
        return arr.flat[0]
    return arr


def is_scalar(x: APLArray) -> bool:
    """Check if value is a scalar."""
    if isinstance(x, (int, float, bool)):
        return True
    if isinstance(x, str) and len(x) <= 1:
        return True
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return True
    return False


def ensure_compatible(left: APLArray, right: APLArray) -> tuple:
    """Ensure arrays are compatible for dyadic operations."""
    left = to_array(left)
    right = to_array(right)
    # Scalar extension
    if left.size == 1:
        left = np.broadcast_to(left.flat[0], right.shape)
    if right.size == 1:
        right = np.broadcast_to(right.flat[0], left.shape)
    return left, right


# =============================================================================
# Monadic Scalar Functions
# =============================================================================

def conjugate(x: APLArray) -> np.ndarray:
    """+ monadic: Conjugate (identity for reals)"""
    return to_array(x).copy()


def negate(x: APLArray) -> np.ndarray:
    """- monadic: Negate"""
    return -to_array(x)


def signum(x: APLArray) -> np.ndarray:
    """× monadic: Signum (sign of)"""
    return np.sign(to_array(x))


def reciprocal(x: APLArray) -> np.ndarray:
    """÷ monadic: Reciprocal"""
    return 1 / to_array(x)


def magnitude(x: APLArray) -> np.ndarray:
    """| monadic: Absolute value / Magnitude"""
    return np.abs(to_array(x))


def floor(x: APLArray) -> np.ndarray:
    """⌊ monadic: Floor"""
    return np.floor(to_array(x))


def ceiling(x: APLArray) -> np.ndarray:
    """⌈ monadic: Ceiling"""
    return np.ceil(to_array(x))


def exponential(x: APLArray) -> np.ndarray:
    """* monadic: Exponential (e^x)"""
    return np.exp(to_array(x))


def natural_log(x: APLArray) -> np.ndarray:
    """⍟ monadic: Natural logarithm"""
    return np.log(to_array(x))


def pi_times(x: APLArray) -> np.ndarray:
    """○ monadic: Pi times"""
    return np.pi * to_array(x)


def factorial(x: APLArray) -> np.ndarray:
    """! monadic: Factorial / Gamma"""
    from scipy.special import gamma
    arr = to_array(x)
    return gamma(arr + 1)


def roll(x: APLArray) -> np.ndarray:
    """? monadic: Roll (random integers 1 to x)"""
    arr = to_array(x).astype(int)
    return np.random.randint(INDEX_ORIGIN, arr + INDEX_ORIGIN)


def logical_not(x: APLArray) -> np.ndarray:
    """~ monadic: Logical NOT"""
    return (~to_array(x).astype(bool)).astype(int)


# =============================================================================
# Dyadic Scalar Functions
# =============================================================================

def plus(left: APLArray, right: APLArray) -> np.ndarray:
    """+ dyadic: Addition"""
    left, right = ensure_compatible(left, right)
    return left + right


def minus(left: APLArray, right: APLArray) -> np.ndarray:
    """- dyadic: Subtraction"""
    left, right = ensure_compatible(left, right)
    return left - right


def times(left: APLArray, right: APLArray) -> np.ndarray:
    """× dyadic: Multiplication"""
    left, right = ensure_compatible(left, right)
    return left * right


def divide(left: APLArray, right: APLArray) -> np.ndarray:
    """÷ dyadic: Division"""
    left, right = ensure_compatible(left, right)
    return left / right


def residue(left: APLArray, right: APLArray) -> np.ndarray:
    """| dyadic: Residue (modulo)"""
    left, right = ensure_compatible(left, right)
    return np.mod(right, left)


def minimum(left: APLArray, right: APLArray) -> np.ndarray:
    """⌊ dyadic: Minimum"""
    left, right = ensure_compatible(left, right)
    return np.minimum(left, right)


def maximum(left: APLArray, right: APLArray) -> np.ndarray:
    """⌈ dyadic: Maximum"""
    left, right = ensure_compatible(left, right)
    return np.maximum(left, right)


def power(left: APLArray, right: APLArray) -> np.ndarray:
    """* dyadic: Power"""
    left, right = ensure_compatible(left, right)
    return np.power(left, right)


def logarithm(left: APLArray, right: APLArray) -> np.ndarray:
    """⍟ dyadic: Logarithm (base left of right)"""
    left, right = ensure_compatible(left, right)
    return np.log(right) / np.log(left)


def circular(left: APLArray, right: APLArray) -> np.ndarray:
    """○ dyadic: Circular/Trigonometric functions"""
    left = to_array(left)
    right = to_array(right)
    result = np.zeros_like(right, dtype=float)

    # APL circular function codes
    funcs = {
        0: lambda x: np.sqrt(1 - x**2),
        1: np.sin,
        2: np.cos,
        3: np.tan,
        4: lambda x: np.sqrt(1 + x**2),
        5: np.sinh,
        6: np.cosh,
        7: np.tanh,
        -1: np.arcsin,
        -2: np.arccos,
        -3: np.arctan,
        -4: lambda x: np.sqrt(x**2 - 1),
        -5: np.arcsinh,
        -6: np.arccosh,
        -7: np.arctanh,
    }

    for code in np.unique(left):
        mask = left == code
        if int(code) in funcs:
            result[mask] = funcs[int(code)](right[mask])

    return result


def binomial(left: APLArray, right: APLArray) -> np.ndarray:
    """! dyadic: Binomial coefficient"""
    from scipy.special import comb
    left, right = ensure_compatible(left, right)
    return comb(right, left, exact=False)


def deal(left: APLArray, right: APLArray) -> np.ndarray:
    """? dyadic: Deal (select left random items from right)"""
    left_val = int(to_scalar(left))
    right_val = int(to_scalar(right))
    return np.random.choice(
        np.arange(INDEX_ORIGIN, right_val + INDEX_ORIGIN),
        size=left_val,
        replace=False
    )


# =============================================================================
# Comparison Functions
# =============================================================================

def equal(left: APLArray, right: APLArray) -> np.ndarray:
    """= dyadic: Equal"""
    left, right = ensure_compatible(left, right)
    return (left == right).astype(int)


def not_equal(left: APLArray, right: APLArray) -> np.ndarray:
    """≠ dyadic: Not equal"""
    left, right = ensure_compatible(left, right)
    return (left != right).astype(int)


def less_than(left: APLArray, right: APLArray) -> np.ndarray:
    """< dyadic: Less than"""
    left, right = ensure_compatible(left, right)
    return (left < right).astype(int)


def less_equal(left: APLArray, right: APLArray) -> np.ndarray:
    """≤ dyadic: Less than or equal"""
    left, right = ensure_compatible(left, right)
    return (left <= right).astype(int)


def greater_than(left: APLArray, right: APLArray) -> np.ndarray:
    """> dyadic: Greater than"""
    left, right = ensure_compatible(left, right)
    return (left > right).astype(int)


def greater_equal(left: APLArray, right: APLArray) -> np.ndarray:
    """≥ dyadic: Greater than or equal"""
    left, right = ensure_compatible(left, right)
    return (left >= right).astype(int)


# =============================================================================
# Logical Functions
# =============================================================================

def logical_and(left: APLArray, right: APLArray) -> np.ndarray:
    """∧ dyadic: Logical AND (also LCM for non-boolean)"""
    left, right = ensure_compatible(left, right)
    if np.issubdtype(left.dtype, np.integer) and np.all((left == 0) | (left == 1)):
        return (left & right).astype(int)
    return np.lcm(left.astype(int), right.astype(int))


def logical_or(left: APLArray, right: APLArray) -> np.ndarray:
    """∨ dyadic: Logical OR (also GCD for non-boolean)"""
    left, right = ensure_compatible(left, right)
    if np.issubdtype(left.dtype, np.integer) and np.all((left == 0) | (left == 1)):
        return (left | right).astype(int)
    return np.gcd(left.astype(int), right.astype(int))


def logical_nand(left: APLArray, right: APLArray) -> np.ndarray:
    """⍲ dyadic: Logical NAND"""
    left, right = ensure_compatible(left, right)
    return (~(left.astype(bool) & right.astype(bool))).astype(int)


def logical_nor(left: APLArray, right: APLArray) -> np.ndarray:
    """⍱ dyadic: Logical NOR"""
    left, right = ensure_compatible(left, right)
    return (~(left.astype(bool) | right.astype(bool))).astype(int)


# =============================================================================
# Structural Functions - Monadic
# =============================================================================

def iota(n: APLArray) -> np.ndarray:
    """⍳N monadic: Index generator"""
    n = to_scalar(n)
    if isinstance(n, (int, float)):
        return np.arange(INDEX_ORIGIN, int(n) + INDEX_ORIGIN)
    # Multi-dimensional iota
    shape = to_array(n).astype(int)
    total = np.prod(shape)
    indices = np.arange(INDEX_ORIGIN, total + INDEX_ORIGIN)
    return indices.reshape(shape)


def shape(x: APLArray) -> np.ndarray:
    """⍴X monadic: Shape"""
    arr = to_array(x)
    if arr.ndim == 0:
        return np.array([], dtype=int)
    return np.array(arr.shape)


def ravel(x: APLArray) -> np.ndarray:
    """,X monadic: Ravel (flatten)"""
    return to_array(x).ravel()


def reverse(x: APLArray) -> np.ndarray:
    """⌽X monadic: Reverse along last axis"""
    arr = to_array(x)
    if arr.ndim == 0:
        return arr
    return np.flip(arr, axis=-1)


def reverse_first(x: APLArray) -> np.ndarray:
    """⊖X monadic: Reverse along first axis"""
    arr = to_array(x)
    if arr.ndim == 0:
        return arr
    return np.flip(arr, axis=0)


def transpose(x: APLArray) -> np.ndarray:
    """⍉X monadic: Transpose"""
    arr = to_array(x)
    if arr.ndim <= 1:
        return arr
    return np.transpose(arr)


def grade_up(x: APLArray) -> np.ndarray:
    """⍋X monadic: Grade up (indices that would sort ascending)"""
    arr = to_array(x)
    return np.argsort(arr) + INDEX_ORIGIN


def grade_down(x: APLArray) -> np.ndarray:
    """⍒X monadic: Grade down (indices that would sort descending)"""
    arr = to_array(x)
    return np.argsort(arr)[::-1] + INDEX_ORIGIN


def unique(x: APLArray) -> np.ndarray:
    """∪X monadic: Unique"""
    arr = to_array(x)
    return np.unique(arr)


def enclose(x: APLArray) -> list:
    """⊂X monadic: Enclose (box)"""
    return [to_array(x)]


def disclose(x: APLArray) -> np.ndarray:
    """⊃X monadic: Disclose / First"""
    if isinstance(x, list) and len(x) > 0:
        return to_array(x[0])
    arr = to_array(x)
    if arr.size > 0:
        return arr.flat[0]
    return arr


def first(x: APLArray) -> APLScalar:
    """↑X monadic: First element"""
    arr = to_array(x)
    if arr.size > 0:
        return arr.flat[0]
    return 0  # or appropriate default


def depth(x: APLArray) -> int:
    """≡X monadic: Depth of nesting"""
    if isinstance(x, (int, float, str, bool)):
        return 0
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            return 1 + max((depth(item) for item in x.flat), default=0)
        return 1
    if isinstance(x, list):
        if len(x) == 0:
            return 1
        return 1 + max(depth(item) for item in x)
    return 0


def tally(x: APLArray) -> int:
    """≢X monadic: Tally (count of first dimension)"""
    arr = to_array(x)
    if arr.ndim == 0:
        return 1
    return arr.shape[0]


# =============================================================================
# Structural Functions - Dyadic
# =============================================================================

def rho(shape_spec: APLArray, data: APLArray) -> np.ndarray:
    """N⍴X dyadic: Reshape"""
    shape_arr = to_array(shape_spec).astype(int).ravel()
    data_arr = to_array(data).ravel()

    total_size = np.prod(shape_arr)

    # APL recycles data if needed
    if data_arr.size < total_size:
        repeats = int(np.ceil(total_size / data_arr.size))
        data_arr = np.tile(data_arr, repeats)

    return data_arr[:total_size].reshape(tuple(shape_arr))


def catenate(left: APLArray, right: APLArray) -> np.ndarray:
    """X,Y dyadic: Catenate along last axis"""
    left = to_array(left)
    right = to_array(right)

    # Handle scalar extension
    if left.ndim == 0:
        left = left.reshape(1)
    if right.ndim == 0:
        right = right.reshape(1)

    # Match ranks
    while left.ndim < right.ndim:
        left = left[np.newaxis, ...]
    while right.ndim < left.ndim:
        right = right[np.newaxis, ...]

    return np.concatenate([left, right], axis=-1)


def catenate_first(left: APLArray, right: APLArray) -> np.ndarray:
    """X⍪Y dyadic: Catenate along first axis"""
    left = to_array(left)
    right = to_array(right)

    if left.ndim == 0:
        left = left.reshape(1)
    if right.ndim == 0:
        right = right.reshape(1)

    while left.ndim < right.ndim:
        left = left[np.newaxis, ...]
    while right.ndim < left.ndim:
        right = right[np.newaxis, ...]

    return np.concatenate([left, right], axis=0)


def take(n: APLArray, x: APLArray) -> np.ndarray:
    """N↑X dyadic: Take"""
    n_arr = to_array(n).astype(int).ravel()
    x_arr = to_array(x)

    if len(n_arr) == 1:
        n_val = n_arr[0]
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1)

        if n_val >= 0:
            if n_val <= x_arr.shape[0]:
                return x_arr[:n_val]
            else:
                # Pad with zeros/spaces
                pad_shape = list(x_arr.shape)
                pad_shape[0] = n_val - x_arr.shape[0]
                if x_arr.dtype.kind in ('U', 'S'):
                    pad = np.full(pad_shape, ' ', dtype=x_arr.dtype)
                else:
                    pad = np.zeros(pad_shape, dtype=x_arr.dtype)
                return np.concatenate([x_arr, pad], axis=0)
        else:
            n_val = abs(n_val)
            if n_val <= x_arr.shape[0]:
                return x_arr[-n_val:]
            else:
                pad_shape = list(x_arr.shape)
                pad_shape[0] = n_val - x_arr.shape[0]
                if x_arr.dtype.kind in ('U', 'S'):
                    pad = np.full(pad_shape, ' ', dtype=x_arr.dtype)
                else:
                    pad = np.zeros(pad_shape, dtype=x_arr.dtype)
                return np.concatenate([pad, x_arr], axis=0)

    # Multi-dimensional take
    result = x_arr
    for axis, n_val in enumerate(n_arr):
        if axis >= result.ndim:
            break
        result = np.take(result, range(abs(n_val)), axis=axis)
    return result


def drop(n: APLArray, x: APLArray) -> np.ndarray:
    """N↓X dyadic: Drop"""
    n_arr = to_array(n).astype(int).ravel()
    x_arr = to_array(x)

    if len(n_arr) == 1:
        n_val = n_arr[0]
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1)

        if n_val >= 0:
            return x_arr[n_val:]
        else:
            return x_arr[:n_val]

    # Multi-dimensional drop
    slices = []
    for axis, n_val in enumerate(n_arr):
        if axis >= x_arr.ndim:
            break
        if n_val >= 0:
            slices.append(slice(n_val, None))
        else:
            slices.append(slice(None, n_val))

    return x_arr[tuple(slices)]


def rotate(n: APLArray, x: APLArray) -> np.ndarray:
    """N⌽X dyadic: Rotate along last axis"""
    n_val = int(to_scalar(n))
    x_arr = to_array(x)
    return np.roll(x_arr, -n_val, axis=-1)


def rotate_first(n: APLArray, x: APLArray) -> np.ndarray:
    """N⊖X dyadic: Rotate along first axis"""
    n_val = int(to_scalar(n))
    x_arr = to_array(x)
    return np.roll(x_arr, -n_val, axis=0)


def dyadic_transpose(axes: APLArray, x: APLArray) -> np.ndarray:
    """A⍉X dyadic: Transpose with axis specification"""
    axes_arr = to_array(axes).astype(int) - INDEX_ORIGIN
    x_arr = to_array(x)
    return np.transpose(x_arr, axes_arr)


def index_of(x: APLArray, y: APLArray) -> np.ndarray:
    """X⍳Y dyadic: Index of"""
    x_arr = to_array(x)
    y_arr = to_array(y)

    result = np.full(y_arr.shape, x_arr.size + INDEX_ORIGIN, dtype=int)

    for i, val in enumerate(y_arr.flat):
        matches = np.where(x_arr == val)[0]
        if len(matches) > 0:
            result.flat[i] = matches[0] + INDEX_ORIGIN

    return result


def membership(x: APLArray, y: APLArray) -> np.ndarray:
    """X∊Y dyadic: Membership"""
    x_arr = to_array(x)
    y_arr = to_array(y)
    return np.isin(x_arr, y_arr).astype(int)


def intersection(left: APLArray, right: APLArray) -> np.ndarray:
    """X∩Y dyadic: Intersection"""
    left_arr = to_array(left)
    right_arr = to_array(right)
    return np.intersect1d(left_arr, right_arr)


def union(left: APLArray, right: APLArray) -> np.ndarray:
    """X∪Y dyadic: Union"""
    left_arr = to_array(left)
    right_arr = to_array(right)
    return np.union1d(left_arr, right_arr)


def without(left: APLArray, right: APLArray) -> np.ndarray:
    """X~Y dyadic: Without (set difference)"""
    left_arr = to_array(left)
    right_arr = to_array(right)
    mask = ~np.isin(left_arr, right_arr)
    return left_arr[mask]


def replicate(counts: APLArray, data: APLArray) -> np.ndarray:
    """N/X dyadic: Replicate"""
    counts_arr = to_array(counts).astype(int)
    data_arr = to_array(data)
    return np.repeat(data_arr, counts_arr)


def expand(mask: APLArray, data: APLArray) -> np.ndarray:
    """M\\X dyadic: Expand"""
    mask_arr = to_array(mask).astype(int)
    data_arr = to_array(data)

    # Count non-zeros in mask
    result = []
    data_idx = 0

    for m in mask_arr:
        if m == 0:
            # Insert fill element
            if data_arr.dtype.kind in ('U', 'S'):
                result.append(' ')
            else:
                result.append(0)
        else:
            # Copy from data
            for _ in range(m):
                if data_idx < len(data_arr):
                    result.append(data_arr[data_idx])
                    data_idx += 1

    return np.array(result)


def encode(radix: APLArray, n: APLArray) -> np.ndarray:
    """R⊤N dyadic: Encode (representation in given radix)"""
    radix_arr = to_array(radix).astype(int)
    n_val = int(to_scalar(n))

    result = []
    for r in reversed(radix_arr):
        result.insert(0, n_val % r)
        n_val //= r

    return np.array(result)


def decode(radix: APLArray, digits: APLArray) -> np.ndarray:
    """R⊥D dyadic: Decode (polynomial evaluation)"""
    radix_arr = to_array(radix).astype(int)
    digits_arr = to_array(digits).astype(int)

    result = 0
    for r, d in zip(radix_arr, digits_arr):
        result = result * r + d

    return np.array(result)


def matrix_divide(left: APLArray, right: APLArray) -> np.ndarray:
    """A⌹B dyadic: Matrix divide"""
    left_arr = to_array(left).astype(float)
    right_arr = to_array(right).astype(float)
    return np.linalg.lstsq(right_arr, left_arr, rcond=None)[0]


def matrix_inverse(x: APLArray) -> np.ndarray:
    """⌹X monadic: Matrix inverse"""
    x_arr = to_array(x).astype(float)
    return np.linalg.inv(x_arr)


# =============================================================================
# Reduction Operators
# =============================================================================

def reduce_plus(x: APLArray, axis: int = -1) -> np.ndarray:
    """+/X: Sum reduction"""
    return np.sum(to_array(x), axis=axis)


def reduce_minus(x: APLArray, axis: int = -1) -> np.ndarray:
    """-/X: Alternating sum reduction"""
    arr = to_array(x)
    if arr.ndim == 0:
        return arr
    # Alternating signs: + - + - ...
    signs = np.array([(-1)**i for i in range(arr.shape[axis])])
    return np.sum(arr * signs, axis=axis)


def reduce_times(x: APLArray, axis: int = -1) -> np.ndarray:
    """×/X: Product reduction"""
    return np.prod(to_array(x), axis=axis)


def reduce_divide(x: APLArray, axis: int = -1) -> np.ndarray:
    """÷/X: Alternating divide reduction"""
    arr = to_array(x)
    if arr.size == 0:
        return np.array(1)
    result = arr.flat[-1]
    for i in range(arr.size - 2, -1, -1):
        result = arr.flat[i] / result
    return result


def reduce_max(x: APLArray, axis: int = -1) -> np.ndarray:
    """⌈/X: Maximum reduction"""
    return np.max(to_array(x), axis=axis)


def reduce_min(x: APLArray, axis: int = -1) -> np.ndarray:
    """⌊/X: Minimum reduction"""
    return np.min(to_array(x), axis=axis)


def reduce_and(x: APLArray, axis: int = -1) -> np.ndarray:
    """∧/X: AND reduction"""
    return np.all(to_array(x).astype(bool), axis=axis).astype(int)


def reduce_or(x: APLArray, axis: int = -1) -> np.ndarray:
    """∨/X: OR reduction"""
    return np.any(to_array(x).astype(bool), axis=axis).astype(int)


# =============================================================================
# Scan Operators
# =============================================================================

def scan_plus(x: APLArray, axis: int = -1) -> np.ndarray:
    """+\\X: Cumulative sum"""
    return np.cumsum(to_array(x), axis=axis)


def scan_times(x: APLArray, axis: int = -1) -> np.ndarray:
    """×\\X: Cumulative product"""
    return np.cumprod(to_array(x), axis=axis)


def scan_max(x: APLArray, axis: int = -1) -> np.ndarray:
    """⌈\\X: Cumulative maximum"""
    return np.maximum.accumulate(to_array(x), axis=axis)


def scan_min(x: APLArray, axis: int = -1) -> np.ndarray:
    """⌊\\X: Cumulative minimum"""
    return np.minimum.accumulate(to_array(x), axis=axis)


# =============================================================================
# Outer Product
# =============================================================================

def outer_product(func: Callable, left: APLArray, right: APLArray) -> np.ndarray:
    """∘.f: Outer product"""
    left_arr = to_array(left)
    right_arr = to_array(right)

    # Create outer product
    result_shape = left_arr.shape + right_arr.shape
    result = np.zeros(result_shape)

    for idx_left in np.ndindex(left_arr.shape):
        for idx_right in np.ndindex(right_arr.shape):
            idx = idx_left + idx_right
            result[idx] = func(left_arr[idx_left], right_arr[idx_right])

    return result


# =============================================================================
# Inner Product
# =============================================================================

def inner_product(left_func: Callable, right_func: Callable,
                  left: APLArray, right: APLArray) -> np.ndarray:
    """f.g: Inner product"""
    left_arr = to_array(left)
    right_arr = to_array(right)

    # Apply right_func element-wise, then reduce with left_func
    if left_arr.ndim == 1 and right_arr.ndim == 1:
        products = right_func(left_arr, right_arr)
        return functools_reduce(left_func, products)

    # Matrix case
    return np.tensordot(left_arr, right_arr, axes=1)


# =============================================================================
# I/O Functions
# =============================================================================

def quad_output(x: APLArray) -> APLArray:
    """⎕← output"""
    print(to_array(x))
    return x


def quad_input(prompt: str = "") -> str:
    """⎕: Evaluated input"""
    return input(prompt)


def quote_quad_input(prompt: str = "") -> str:
    """⍞: Character input"""
    return input(prompt)


# =============================================================================
# Format Functions
# =============================================================================

def format_array(x: APLArray) -> str:
    """⍕X monadic: Format to string"""
    return str(to_array(x))


def format_with_spec(spec: APLArray, x: APLArray) -> str:
    """S⍕X dyadic: Format with specification"""
    arr = to_array(x)
    spec_arr = to_array(spec)

    if len(spec_arr) >= 2:
        width = int(spec_arr[0])
        decimals = int(spec_arr[1])
        return f"{arr:{width}.{decimals}f}"

    return str(arr)


def execute(s: str) -> Any:
    """⍎S: Execute APL expression (limited support)"""
    # This would need the full parser to work properly
    # For now, evaluate as Python
    return eval(s)


# =============================================================================
# Miscellaneous
# =============================================================================

def zilde() -> np.ndarray:
    """⍬: Empty numeric vector"""
    return np.array([], dtype=float)


def left_tack(left: APLArray, right: APLArray) -> APLArray:
    """⊣ dyadic: Left (returns left argument)"""
    return left


def right_tack(left: APLArray, right: APLArray) -> APLArray:
    """⊢ dyadic: Right (returns right argument)"""
    return right


def same(x: APLArray) -> APLArray:
    """⊢ monadic: Same (identity)"""
    return x


# =============================================================================
# Convenience Aliases
# =============================================================================

# Map APL symbols to functions
MONADIC_FUNCTIONS = {
    '+': conjugate,
    '-': negate,
    '×': signum,
    '÷': reciprocal,
    '|': magnitude,
    '⌊': floor,
    '⌈': ceiling,
    '*': exponential,
    '⍟': natural_log,
    '○': pi_times,
    '!': factorial,
    '?': roll,
    '~': logical_not,
    '⍳': iota,
    '⍴': shape,
    ',': ravel,
    '⌽': reverse,
    '⊖': reverse_first,
    '⍉': transpose,
    '⍋': grade_up,
    '⍒': grade_down,
    '∪': unique,
    '⊂': enclose,
    '⊃': disclose,
    '↑': first,
    '≡': depth,
    '≢': tally,
    '⌹': matrix_inverse,
    '⍕': format_array,
    '⊢': same,
}

DYADIC_FUNCTIONS = {
    '+': plus,
    '-': minus,
    '×': times,
    '÷': divide,
    '|': residue,
    '⌊': minimum,
    '⌈': maximum,
    '*': power,
    '⍟': logarithm,
    '○': circular,
    '!': binomial,
    '?': deal,
    '=': equal,
    '≠': not_equal,
    '<': less_than,
    '≤': less_equal,
    '>': greater_than,
    '≥': greater_equal,
    '∧': logical_and,
    '∨': logical_or,
    '⍲': logical_nand,
    '⍱': logical_nor,
    '⍴': rho,
    ',': catenate,
    '⍪': catenate_first,
    '↑': take,
    '↓': drop,
    '⌽': rotate,
    '⊖': rotate_first,
    '⍉': dyadic_transpose,
    '⍳': index_of,
    '∊': membership,
    '∩': intersection,
    '∪': union,
    '~': without,
    '/': replicate,
    '\\': expand,
    '⊤': encode,
    '⊥': decode,
    '⌹': matrix_divide,
    '⊣': left_tack,
    '⊢': right_tack,
}

REDUCE_FUNCTIONS = {
    '+': reduce_plus,
    '-': reduce_minus,
    '×': reduce_times,
    '÷': reduce_divide,
    '⌈': reduce_max,
    '⌊': reduce_min,
    '∧': reduce_and,
    '∨': reduce_or,
}

SCAN_FUNCTIONS = {
    '+': scan_plus,
    '×': scan_times,
    '⌈': scan_max,
    '⌊': scan_min,
}


# =============================================================================
# Export all public functions
# =============================================================================

__all__ = [
    # Helpers
    'to_array', 'to_scalar', 'is_scalar', 'INDEX_ORIGIN',

    # Monadic scalar
    'conjugate', 'negate', 'signum', 'reciprocal', 'magnitude',
    'floor', 'ceiling', 'exponential', 'natural_log', 'pi_times',
    'factorial', 'roll', 'logical_not',

    # Dyadic scalar
    'plus', 'minus', 'times', 'divide', 'residue', 'minimum', 'maximum',
    'power', 'logarithm', 'circular', 'binomial', 'deal',

    # Comparison
    'equal', 'not_equal', 'less_than', 'less_equal', 'greater_than', 'greater_equal',

    # Logical
    'logical_and', 'logical_or', 'logical_nand', 'logical_nor',

    # Structural monadic
    'iota', 'shape', 'ravel', 'reverse', 'reverse_first', 'transpose',
    'grade_up', 'grade_down', 'unique', 'enclose', 'disclose', 'first',
    'depth', 'tally', 'matrix_inverse',

    # Structural dyadic
    'rho', 'catenate', 'catenate_first', 'take', 'drop', 'rotate', 'rotate_first',
    'dyadic_transpose', 'index_of', 'membership', 'intersection', 'union',
    'without', 'replicate', 'expand', 'encode', 'decode', 'matrix_divide',

    # Reduction
    'reduce_plus', 'reduce_minus', 'reduce_times', 'reduce_divide',
    'reduce_max', 'reduce_min', 'reduce_and', 'reduce_or',

    # Scan
    'scan_plus', 'scan_times', 'scan_max', 'scan_min',

    # Higher order
    'outer_product', 'inner_product',

    # I/O
    'quad_output', 'quad_input', 'quote_quad_input',

    # Format
    'format_array', 'format_with_spec', 'execute',

    # Misc
    'zilde', 'left_tack', 'right_tack', 'same',

    # Lookup tables
    'MONADIC_FUNCTIONS', 'DYADIC_FUNCTIONS', 'REDUCE_FUNCTIONS', 'SCAN_FUNCTIONS',
]

"""Run the kmaziarz/GLN fork without its compiled torch extensions.

GLN's compiled torch op is the jagged log-softmax (`extlib`, `extlib_cuda`). It also imports
`torch_scatter` and `torch_sparse`. The shims below provide the same functions in plain torch, so the fork needs no build against a given torch version.
`prepare` also puts the GLN checkout on sys.path. GLN's C++ molecule library (libmolgnn.so) is built with `make`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

GLN_COMMIT = "5437fa7a7cfe98354aa5b0e743b964a97971016c"
GLN_URL = "https://github.com/kmaziarz/GLN"


def _segments(prefix_sum):
    ends = prefix_sum.to(torch.long)
    lengths = torch.diff(ends, prepend=torch.zeros(1, dtype=ends.dtype, device=ends.device))
    return torch.repeat_interleave(torch.arange(len(ends), device=ends.device), lengths), len(ends)


def jagged_log_softmax_forward(logits, prefix_sum):
    """Per-segment log-softmax. `prefix_sum` holds the end offset of each segment."""
    seg, n = _segments(prefix_sum)
    x = logits.double()
    peak = torch.full((n,), -torch.inf, dtype=x.dtype, device=x.device).scatter_reduce(
        0, seg, x, "amax"
    )
    shifted = x - peak[seg]
    total = torch.zeros(n, dtype=x.dtype, device=x.device).index_add_(0, seg, shifted.exp())
    return (shifted - total.log()[seg]).to(logits.dtype)


def jagged_log_softmax_backward(output, grad_output, prefix_sum):
    seg, n = _segments(prefix_sum)
    grad_sum = torch.zeros(n, dtype=grad_output.dtype, device=grad_output.device).index_add_(
        0, seg, grad_output
    )
    return grad_output - output.exp() * grad_sum[seg]


def _expand(src, index, dim):
    """Broadcast a 1-D `index` along `dim`, as torch_scatter does."""
    if index.dim() == 1:
        index = index.view([-1 if d == dim % src.dim() else 1 for d in range(src.dim())])
    return index.expand_as(src)


def _zeros(src, index, dim, dim_size):
    size = list(src.shape)
    size[dim] = int(index.max()) + 1 if dim_size is None else dim_size
    return torch.zeros(size, dtype=src.dtype, device=src.device)


def scatter_add(src, index, dim=0, dim_size=None):
    return _zeros(src, index, dim, dim_size).scatter_add_(dim, _expand(src, index, dim), src)


def scatter_mean(src, index, dim=0, dim_size=None):
    index = _expand(src, index, dim)
    count = _zeros(src, index, dim, dim_size).scatter_add_(dim, index, torch.ones_like(src))
    return scatter_add(src, index, dim, dim_size) / count.clamp(min=1)


def _scatter_extreme(reduce, src, index, dim=-1, out=None, dim_size=None):
    result = _zeros(src, index, dim, dim_size).scatter_reduce_(
        dim, _expand(src, index, dim), src, reduce, include_self=False
    )
    return result, None


def scatter_max(*args, **kwargs):
    return _scatter_extreme("amax", *args, **kwargs)


def scatter_min(*args, **kwargs):
    return _scatter_extreme("amin", *args, **kwargs)


def spmm(index, value, m, matrix):
    """Product of the sparse matrix (`index`, `value`) with `m` rows and the dense `matrix`."""
    row, col = index
    return torch.zeros(m, matrix.shape[1], dtype=matrix.dtype, device=matrix.device).index_add_(
        0, row, value[:, None] * matrix[col]
    )


def prepare(gln_root: Path) -> None:
    """Make `import gln` work: shim the extensions and add the checkout to sys.path."""
    for name, suffix in (("extlib", ""), ("extlib_cuda", "_cuda")):
        module = types.ModuleType(name)
        module.__dict__[f"jagged_log_softmax_forward{suffix}"] = jagged_log_softmax_forward
        module.__dict__[f"jagged_log_softmax_backward{suffix}"] = jagged_log_softmax_backward
        sys.modules.setdefault(name, module)
    scatter = types.ModuleType("torch_scatter")
    scatter.scatter_add, scatter.scatter_mean = scatter_add, scatter_mean
    scatter.scatter_max, scatter.scatter_min = scatter_max, scatter_min
    sys.modules.setdefault("torch_scatter", scatter)
    sparse = types.ModuleType("torch_sparse")
    sparse.spmm = spmm
    sys.modules.setdefault("torch_sparse", sparse)
    library = gln_root / "gln/mods/mol_gnn/mg_clib/build/dll/libmolgnn.so"
    if not library.exists():
        raise FileNotFoundError(
            f"{library} is missing. Run: git clone {GLN_URL} {gln_root} && git -C {gln_root} checkout {GLN_COMMIT} "
            f"&& make -C {gln_root}/gln/mods/mol_gnn/mg_clib"
        )
    sys.path.insert(0, str(gln_root))

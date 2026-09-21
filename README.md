# torch-inductor-inplace-slice-shift-aliasing-guard

Detects and guards against a real, currently-open PyTorch correctness
defect: `torch.compile(backend="inductor")` silently corrupts an
in-place row-shift slice assignment such as `x[1:] = x[:-1].clone()`
(and equivalent spellings like `x[1:].copy_(x[:-1].clone())`).

Tracking issue: [pytorch/pytorch#197829](https://github.com/pytorch/pytorch/issues/197829)
(open, no fix merged as of this guard's creation).

## The bug, in one example

```python
import torch

def shift(x):
    x[1:] = x[:-1].clone()
    return x

x = torch.arange(1, 13, dtype=torch.float64).reshape(6, 2).clone()
shift(torch.compile(lambda t: shift(t))(x))
# On CPU: row 0 silently clobbers every other row, every run, at any
# tensor size. On CUDA: correct at small shapes, wrong
# non-deterministically at larger ones.
```

Independently reproduced on this project's own CI and on the
maintainer's host (torch 2.14.0, CPU-only): a real, deterministic
divergence between eager and compiled output -- 256/768 elements
wrong at shape (3, 16, 16), 24/32 wrong at shape (8, 4), and
~2.09M/2.10M wrong at shape (1024, 2048), in every case with row 0's
values propagated into every subsequent row.

## Root cause

Per the issue author's own trace (upstream comment, 2026-09-20):
`remove_noop_ops` drops the `.clone()` on the right-hand side as a
no-op while the graph is still in its functional form (the clone
looks redundant in isolation -- the graph doesn't yet know the target
will be reinplaced). The reinplacing pass then turns the resulting
`_generalized_scatter(x, src, ...)` into `view(x).copy_(src)` -- but
by then `src` is a VIEW OF `x`'s OWN STORAGE (the clone that would
have made it independent was already eliminated), and
`should_reinplace_scatter`'s `can_inplace` check only looks at LATER
uses of that storage, not whether `src` itself already aliases `x`.
The resulting in-place copy reads and writes the same overlapping
buffer.

This is a **distinct** root cause and code path from this fleet's
sibling repos:
- `torch-inductor-scatter-copyback-alias-guard` (pytorch#195451) guards
  a RETURN-VALUE aliasing contract change where the underlying VALUES
  are still correct.
- `torch-inductor-duplicate-index-writeorder-guard` (pytorch#197582)
  guards computed-vs-literal duplicate INDEX tensors in index
  assignment, a different Inductor pass entirely.

Here the assignment TARGET itself is corrupted with genuinely wrong
values -- there is no separate "output" to patch after the call,
because `x` is silently wrong.

## The guard

`safe_slice_shift(x, shift=1, dim=0)` performs the equivalent row-shift
by forcing the assignment to execute OUTSIDE any enclosing
`torch.compile` region via `torch.compiler.disable`, so Inductor's
reinplacing pass never gets a chance to mis-optimize it -- restoring
eager's correct semantics even when called from inside a compiled
function. When the upstream bug is eventually fixed, this guard
remains correct (eager execution is always correct) at the cost of one
extra graph-break boundary, which can be removed once #197829 is fixed
in a stable release this fleet targets.

```python
from torch_inductor_inplace_slice_shift_aliasing_guard import safe_slice_shift

def shift(x):
    return safe_slice_shift(x, shift=1, dim=0)

compiled_shift = torch.compile(shift)
x = torch.arange(1, 13, dtype=torch.float64).reshape(6, 2).clone()
compiled_shift(x)  # correct, matches eager
```

## Limitations

- Only covers `dim=0` row-shift assignments matching the issue's exact
  pattern shape; other axes or non-contiguous slice patterns are out
  of scope for this guard (raises `NotImplementedError`).
- Verified on CPU (this fleet's macOS/Linux CI). The upstream issue
  also reports non-deterministic corruption on CUDA at larger shapes;
  this guard's `torch.compiler.disable` approach should be equally
  effective there (it prevents Inductor from seeing the pattern at
  all, regardless of device), but this repo's CI does not have GPU
  runners, so CUDA behavior is not independently verified here --
  only the CPU-deterministic case is CI-tested.
- A userspace workaround for the interim only; once upstream fixes
  `should_reinplace_scatter`'s aliasing check (issue #197829), this
  guard becomes unnecessary and should be retired.

## CLI

```
torch-inductor-inplace-slice-shift-aliasing-guard         # human-readable report
torch-inductor-inplace-slice-shift-aliasing-guard --json  # machine-readable
torch-inductor-inplace-slice-shift-aliasing-guard --no-color
```

Exit code 0 if the guard is fully correct on this host (or torch is
unavailable is exit 2); exit 1 if the guard fails on any case.

## Install

```
pip install torch-inductor-inplace-slice-shift-aliasing-guard[torch]
```

## License

MIT

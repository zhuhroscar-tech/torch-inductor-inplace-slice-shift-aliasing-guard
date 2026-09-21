"""Tests for torch-inductor-inplace-slice-shift-aliasing-guard.
Requires the 'torch' extra (skipped otherwise).

Design mirrors this fleet's established discipline: every guard claim
is backed by a real reproduction, not an assumption, and at least one
test proves the test suite itself would have failed before the fix
(bug-injection verification), not just that the fixed code path
returns success. Includes an independent oracle (row-by-row
index-copy into a fresh tensor) that shares no code with either the
buggy pattern or the guard.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_inductor_inplace_slice_shift_aliasing_guard.core import (  # noqa: E402
    TorchUnavailableError,
    _unsafe_slice_shift,
    diagnose,
    safe_slice_shift,
)


def _oracle_shift(x, shift=1):
    """Independent reference: allocate fresh, index-copy row by row."""
    n = x.shape[0]
    out = x.clone()
    for i in range(n - 1, shift - 1, -1):
        out[i] = x[i - shift]
    return out


def test_diagnose_runs_and_reports_torch_version():
    report = diagnose()
    assert report["torch_version"] == torch.__version__
    assert len(report["cases"]) == 3
    assert report["issue_url"] == "https://github.com/pytorch/pytorch/issues/197829"


def test_native_bug_is_actually_reproduced_on_this_host():
    """Core evidentiary claim: prove the Inductor in-place slice-shift
    corruption is real on the CURRENTLY installed torch build, not
    merely cited from the issue tracker (pytorch/pytorch#197829). If
    torch fixes this upstream, this assertion should start failing --
    news the tool should surface, not silently pass."""
    report = diagnose()
    assert report["any_native_bug"] is True, (
        "Expected the known upstream Inductor in-place slice-shift "
        f"aliasing bug (pytorch/pytorch#197829) to reproduce on torch "
        f"{torch.__version__}; if this now fails, the bug may have "
        "been fixed upstream -- verify against the issue tracker "
        "before assuming a test regression."
    )
    for c in report["cases"]:
        assert c["eager_correct"] is True, c


def test_guard_fully_correct_across_all_cases():
    report = diagnose()
    assert report["guard_fully_correct"] is True
    for c in report["cases"]:
        assert c["guarded_matches_eager"], c


def test_safe_slice_shift_matches_oracle_direct():
    """Direct, minimal reproduction of the guard's core claim without
    going through diagnose(): calling safe_slice_shift from inside a
    compiled function must match the independent row-by-row oracle."""
    x = torch.arange(1, 13, dtype=torch.float64).reshape(6, 2).clone()
    expected = _oracle_shift(x, shift=1)

    def fn(t):
        return safe_slice_shift(t, shift=1, dim=0)

    compiled = torch.compile(fn, fullgraph=False)
    torch._dynamo.reset()
    out = compiled(x)
    assert torch.equal(out, expected)
    assert torch.equal(x, expected)  # mutated in place


def test_native_unsafe_pattern_diverges_bug_injection_check():
    """Bug-injection check proving the regression tests above are
    real: deliberately compile the RAW (unguarded) buggy pattern and
    confirm it DOES diverge from the oracle under torch.compile --
    i.e. if safe_slice_shift() were a no-op passthrough (the bug this
    tool guards against), the guard tests above would correctly fail.
    This proves those tests are not tautological."""
    x = torch.arange(1, 13, dtype=torch.float64).reshape(6, 2).clone()
    expected = _oracle_shift(x, shift=1)

    def fn(t):
        return _unsafe_slice_shift(torch, t, shift=1, dim=0)

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(x)
    assert not torch.equal(out, expected), (
        "Expected the RAW (unguarded) compiled slice-shift to diverge "
        "from the oracle (that is the whole bug this tool detects, "
        "pytorch/pytorch#197829); if this assertion fails, the "
        "underlying bug may have disappeared upstream, which would "
        "make the guard tautologically pass for the wrong reason."
    )


def test_eager_unsafe_pattern_is_correct():
    """The buggy pattern is correct in EAGER mode -- only torch.compile
    corrupts it. This isolates the defect to compilation, matching the
    upstream issue's own characterization."""
    x = torch.arange(1, 13, dtype=torch.float64).reshape(6, 2).clone()
    expected = _oracle_shift(x, shift=1)
    _unsafe_slice_shift(torch, x, shift=1, dim=0)
    assert torch.equal(x, expected)


def test_invalid_shift_raises():
    x = torch.arange(1, 7, dtype=torch.float64)
    with pytest.raises(ValueError):
        _unsafe_slice_shift(torch, x, shift=0, dim=0)
    with pytest.raises(ValueError):
        _unsafe_slice_shift(torch, x, shift=6, dim=0)


def test_dim_other_than_zero_not_implemented():
    x = torch.arange(1, 7, dtype=torch.float64).reshape(2, 3)
    with pytest.raises(NotImplementedError):
        _unsafe_slice_shift(torch, x, shift=1, dim=1)


def test_torch_unavailable_error_is_distinct_type():
    """Sanity check the error type exists and is a RuntimeError
    subclass, independent of whether torch is actually installed in
    this env."""
    assert issubclass(TorchUnavailableError, RuntimeError)

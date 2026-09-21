"""Command-line interface: run the from-scratch diagnosis of the
torch.compile (Inductor) in-place slice-shift aliasing bug
(pytorch/pytorch#197829) against the currently installed torch build,
using the shared semantic-color design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-inductor-inplace-slice-shift-aliasing-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "Inductor backend silently corrupts an in-place row-shift "
            "slice assignment (x[1:] = x[:-1].clone(), pytorch/pytorch"
            "#197829) -- and verify safe_slice_shift() restores "
            "eager's correct semantics. Never trusts a cached or "
            "previously-reported result, always re-runs the repro on "
            "THIS host's actual installed torch version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-inductor-inplace-slice-shift-aliasing-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_native_bug"]:
        print(status_headline(style, "fail", "Inductor in-place slice-shift aliasing bug reproduced on this host (pytorch#197829)"))
    else:
        print(status_headline(style, "info", "no in-place slice-shift aliasing divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_slice_shift() restores eager's correct semantics on every case"))
    else:
        print(status_headline(style, "fail", "guard did NOT restore correct semantics on at least one case"))

    section("cases (shape, shift -> compiled mismatches / guard result)")
    for c in report["cases"]:
        native_flag = (
            f"WRONG ({c['compiled_mismatches']}/{c['compiled_total']})"
            if not c["compiled_matches_eager"]
            else "ok"
        )
        guard_flag = "guard-ok" if c["guarded_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"shape={c['shape']} shift={c['shift']}",
                    f"eager_correct={c['eager_correct']}  native={native_flag:20s}  {guard_flag}",
                )
            ]
        )

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())

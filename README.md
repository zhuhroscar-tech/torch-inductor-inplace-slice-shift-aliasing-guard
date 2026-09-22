# torch-inductor-inplace-slice-shift-aliasing-guard

This repository has been consolidated into **torch-correctness-guards**.

Use the maintained umbrella package instead:

```bash
pip install torch-correctness-guards[torch]
torch-guard run inplace-slice-shift-aliasing
```

Python API:

```python
from torch_correctness_guards.guards.inplace_slice_shift_aliasing import safe_slice_shift
```

The original guard detected and worked around PyTorch Inductor issue
[pytorch/pytorch#197829](https://github.com/pytorch/pytorch/issues/197829),
where compiled in-place slice-shift assignments such as
`x[1:] = x[:-1].clone()` can corrupt overlapping source/target storage.

The consolidated package keeps the same diagnostic and guard behavior while
sharing one CLI, test suite, packaging story, and release surface with the rest
of the PyTorch correctness guards.

Canonical repository:
https://github.com/zhuhroscar-tech/torch-correctness-guards

This repository is archived for discoverability and history only.

## License

MIT

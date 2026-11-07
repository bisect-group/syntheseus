"""LocalRetro's preprocessing, run with a fix for numpy 2.

`Run_preprocessing` stores its label lists by writing `str(labels)` to CSV and reads them back with
`eval`. Under numpy 2 an integer taken from a dataframe prints as `np.int64(3)` rather than `3`, so
the stored text only evaluates in a namespace where `np` is bound; the upstream module never
imports numpy and the read fails with `NameError: name 'np' is not defined`. Binding numpy into
that module's globals makes the stored text evaluate to the same integers it came from.

`combine_preprocessed_data` also calls `DataFrame.append`, which pandas 2 removed after renaming it
to `_append`; the name is restored for the call.

The same `np.int64(...)` text reaches the written CSVs, where `Train.py`'s dataset reads it back
with `eval` as well. Writing the plain integers into those files fixes every reader at once and
leaves the values unchanged.

Everything else follows the upstream `__main__` block, with the same defaults and order, so the
files produced are the ones `Train.py` expects.

    $PY -m emulator_bench.localretro_preprocess -d <dataset>    # run from the workdir's scripts/
"""

from __future__ import annotations

import re
from argparse import ArgumentParser
from pathlib import Path


def main() -> None:
    import numpy as np
    import pandas as pd
    from local_retro.preprocessing import Run_preprocessing as rp
    from local_retro.preprocessing.Extract_from_train_data import build_template_extractor

    rp.np = np  # the label strings written under numpy 2 evaluate only with `np` in scope
    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = pd.DataFrame._append  # pandas 2 renamed it

    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", default="USPTO_50K")
    parser.add_argument("-f", "--force", default=False)
    parser.add_argument("-r", "--retro", default=True)
    parser.add_argument("-v", "--verbose", default=False)
    parser.add_argument("-m", "--max-edit-n", default=8)
    parser.add_argument("-stereo", "--use-stereo", default=True)
    parser.add_argument("-min", "--min-template-n", type=int, default=1)
    args = parser.parse_args().__dict__
    args["output_dir"] = "../data/%s" % args["dataset"]

    template_dicts, template_infos = rp.load_templates(args)
    extractor = build_template_extractor(args)
    test_pre = rp.labeling_dataset(args, "test", template_dicts, template_infos, extractor)
    rp.make_simulate_output(args)
    val_pre = rp.labeling_dataset(args, "val", template_dicts, template_infos, extractor)
    train_pre = rp.labeling_dataset(args, "train", template_dicts, template_infos, extractor)
    rp.combine_preprocessed_data(train_pre, val_pre, test_pre, args)
    unwrap_numpy_reprs(Path(args["output_dir"]))


NUMPY_REPR = re.compile(r"np\.(?:int|uint|float)\d*\(([^()]*)\)")


def unwrap_numpy_reprs(output_dir: Path) -> list[Path]:
    """Rewrite `np.int64(3)` as `3` in the generated CSVs. Returns the files changed."""
    changed = []
    for path in sorted(output_dir.glob("*.csv")):
        text = path.read_text()
        fixed = NUMPY_REPR.sub(r"\1", text)
        if fixed != text:
            tmp = path.with_name(f"{path.name}.tmp")
            tmp.write_text(fixed)
            tmp.replace(path)
            changed.append(path)
    print(f"unwrapped numpy reprs in {[p.name for p in changed]}", flush=True)
    return changed


if __name__ == "__main__":
    main()

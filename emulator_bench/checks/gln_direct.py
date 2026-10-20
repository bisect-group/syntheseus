"""Run GLN RetroGLN directly (same calls as the syntheseus wrapper). Usage: gln_direct.py <model_dir> <in.smiles> <out.json> <num_results> [shim_root]"""

import json
import sys
from pathlib import Path

EMUL = str(Path(__file__).resolve().parents[2])
model_dir, inp, out, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
if len(sys.argv) > 5:
    sys.path.insert(0, EMUL)
    from emulator_bench.gln_compat import prepare

    prepare(Path(sys.argv[5]))
import torch  # noqa: E402

name = "schneider50k"
ckpt = Path(model_dir) / f"{name}.ckpt"
args = {
    "dropbox": model_dir,
    "data_name": name,
    "model_for_test": ckpt,
    "tpl_name": "default",
    "f_atoms": Path(model_dir) / f"cooked_{name}" / "atom_list.txt",
    "gpu": 0 if torch.cuda.is_available() else -1,
}
from gln.common.cmd_args import cmd_args  # noqa: E402

sys.argv = []
for a, v in args.items():
    setattr(cmd_args, a, v)
    sys.argv += [f"-{a}", str(v)]
from gln.test.model_inference import RetroGLN  # noqa: E402

model = RetroGLN(model_dir, ckpt)
res = []
for smi in Path(inp).read_text().split():
    r = model.run(smi, k, k)
    res.append(
        {
            "input": smi,
            "reactants": None if r is None else r["reactants"],
            "scores": None if r is None else [float(x) for x in r["scores"]],
        }
    )
Path(out).write_text(json.dumps(res))
print("done", len(res))

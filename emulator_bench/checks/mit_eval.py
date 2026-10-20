import os
import sys

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from pathlib import Path

from emulator_bench.common import OUT_ROOT
from emulator_bench.run_model import build_model
from syntheseus.cli.eval_single_step import compute_metrics
from syntheseus.reaction_prediction.data.dataset import DataFold, DiskReactionDataset
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample
from syntheseus.reaction_prediction.utils.misc import set_random_seed

model_name, out = sys.argv[1], sys.argv[2]
set_random_seed(0)
model = build_model(model_name, "forward", OUT_ROOT, "cuda:0")
r = compute_metrics(
    model,
    dataset=DiskReactionDataset(Path(sys.argv[3]), sample_cls=ReactionSample),
    num_dataset_truncation=None,
    num_top_results=int(sys.argv[4]) if len(sys.argv) > 4 else 10,
    fold=DataFold.TEST,
    batch_size=16,
    include_predictions=True,
    resumable=True,
    output_dir=Path(out),
)
print("RESULT", model_name, r.num_samples, [round(x, 3) for x in r.top_k[:10]], r.mrr)

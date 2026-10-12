"""Plot retained Q/K role scores and final routing; no GPU model work."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from safetensors.torch import load_file


def main():
    torch.set_num_threads(4)
    root = Path(sys.argv[1]).resolve()
    record = {"scope": "Q/K role-softmax and saved prototype scores; probabilities are heuristic, not calibrated attention or mask accuracy",
              "routing_source": "multi_proto_bg uses only step 2 / block 18; ensemble_bg uses all four taps",
              "head_aggregation": "mean of all 48 heads for each role; background role averages its phrase tokens",
              "cases": []}
    for case in ("park", "woman_front_strong_overlap"):
        fig, axes = plt.subplots(2, 7, figsize=(19, 6), constrained_layout=True)
        for row, branch in enumerate(("off", "on")):
            folder = root / case / "suites" / branch
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            tensors = load_file(str(folder / "maps.safetensors"))
            grid = tuple(manifest["report"]["grid"])
            config = manifest["report"]["normalized_config"]
            for column, (step, block) in enumerate(((1, 16), (1, 18), (2, 16), (2, 18))):
                prefix = f"evidence.step_{step}.block_{block}."
                logits = torch.stack([tensors[prefix + role + ".token_scores"].mean(dim=0)
                                      for role in ("target", "protected", "background")], dim=-1)
                assert all(torch.all(tensors[prefix + role + ".token_allowed_fraction"] == 1)
                           for role in ("target", "protected", "background"))
                context = (logits / config["context_temperature"]).softmax(dim=-1)
                axes[row, column].imshow(context[:, 0].reshape(grid).numpy(), vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
                axes[row, column].set_title(f"{branch.upper()} Q/K role score\nstep {step} / block {block}", fontsize=10)
            prefix = "multi_proto_bg."
            score = tensors[prefix + "raw_maps.target"].reshape(grid)
            axes[row, 4].imshow(score.numpy(), vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
            axes[row, 4].set_title(f"{branch.upper()} prototype T score\nstep 2 / block 18", fontsize=10)
            labels = torch.zeros(grid, dtype=torch.int64)
            labels[tensors[prefix + "masks.target"][0].bool()] = 1
            labels[tensors[prefix + "masks.protected"][0].bool()] = 2
            labels[tensors[prefix + "uncertain_mask"][0].bool()] = 3
            colors = ListedColormap(["#242b34", "#f4c14c", "#4598dc", "#b374c7"])
            axes[row, 5].imshow(labels.numpy(), vmin=0, vmax=3, cmap=colors, interpolation="nearest")
            axes[row, 5].set_title(f"{branch.upper()} routing\nyellow T / blue P / purple unknown", fontsize=9)
            ensemble = tensors.get("ensemble_bg.masks.target")
            if ensemble is not None:
                axes[row, 6].imshow(ensemble[0].numpy(), vmin=0, vmax=1, cmap="gray", interpolation="nearest")
                axes[row, 6].set_title(f"{branch.upper()} existing ensemble\nT mask (no generation)", fontsize=10)
            else:
                axes[row, 6].text(.1, .5, "ensemble failed", transform=axes[row, 6].transAxes)
        for axis in axes.ravel():
            axis.set_xlim(-.5, 63.5)
            axis.set_ylim(63.5, -.5)
            axis.set_xticks([0, 16, 32, 48, 63])
            axis.set_yticks([0, 16, 32, 48, 63])
            axis.tick_params(labelsize=8)
        title = "Park" if case == "park" else "Strong overlap"
        fig.suptitle(title + " | OFF (top) / ON (bottom) | Fixed 0..1 score scale\nRouting also includes prototype features, seeds, confidence and margin", fontsize=12)
        path = root / case / "attention-score-comparison-v2.png"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        record["cases"].append({"case": case, "figure": str(path)})
        print("PLOTTED", case, flush=True)
    with (root / "attention-score-analysis-v2.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)


if __name__ == "__main__":
    main()

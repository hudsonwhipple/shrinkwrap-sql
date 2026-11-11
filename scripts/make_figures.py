"""Draw the README's charts from the result files, in a light and a dark version.

    .venv/bin/python scripts/make_figures.py      # or: make figures

Writes docs/figures/<chart>-light.svg and <chart>-dark.svg. The README shows whichever matches the
reader's GitHub theme. The three series colors are a categorical palette checked for colorblind
separation against both page backgrounds; every plotted value is also in the README's tables.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "figures"

# Surfaces are GitHub's page backgrounds, so the charts sit on the page rather than in a box.
THEMES = {
    "light": {
        "surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#7a7873",
        "grid": "#e8e7e1", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#0d1117", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#8d8b85",
        "grid": "#21262d", "axis": "#3d444d", "series": ["#3987e5", "#d95926", "#199e70"],
    },
}
LINE = 1.5        # points; about 2 px
MARKER = 7        # points; about 9 px across
RING = 1.5        # surface-colored ring around each marker, about 2 px
BITS = ["bf16", "8-bit", "6-bit", "4-bit", "3-bit"]


def load() -> dict:
    def csv_rows(path, key):
        return {row[key]: row for row in csv.DictReader(open(RESULTS / path))}

    def acc(path):
        return 100 * json.load(open(RESULTS / path))["execution_accuracy"]

    sft_sweep = csv_rows("sweep_sft/summary.csv", "run")
    grpo_sweep = csv_rows("sweep_grpo/summary.csv", "run")
    speed = csv_rows("speed.csv", "format")
    quant = ["q8g64", "q6g64", "q4g64", "q3g64"]

    def sweep(rows, pipeline):
        return [100 * float(rows[f"{q}-{pipeline}"]["execution_accuracy"]) for q in quant]

    def tps(fmt):
        return float(speed[fmt]["generation_tps"])

    return {
        # MLX's quantizer; adapters fused before quantizing. Every bf16 point is the fused model, the one
        # its quantized versions are made from.
        "base": [acc("baseline_dev1034.json")] + sweep(sft_sweep, "base"),
        "sft": [acc("sft_fused.json")] + sweep(sft_sweep, "merged"),
        "grpo": [acc("grpo_fused.json")] + sweep(grpo_sweep, "merged"),
        "gptq": [acc("gptq_q3g64.json"), acc("sft_gptq_q3g64.json"), acc("grpo_gptq_q3g64.json")],
        # The SFT model: speed and accuracy, fused then quantized vs the adapter on a quantized base.
        "fused_tps": [tps("bf16-fused")] + [tps(f"mlx-q{b}-fused") for b in (8, 6, 4, 3)],
        "fused_acc": [acc("sft_fused.json")] + sweep(sft_sweep, "merged"),
        "adapter_tps": [tps("bf16-adapter")] + [tps(f"mlx-q{b}-adapter") for b in (8, 6, 4, 3)],
        "adapter_acc": [acc("mine_r16.json")] + sweep(sft_sweep, "adapter"),
    }


def setup(t: dict, title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.075, right=0.975, top=0.72, bottom=0.17)
    fig.text(0.012, 0.955, title, color=t["ink"], fontsize=15, fontweight="bold", va="top")
    fig.text(0.012, 0.885, subtitle, color=t["ink2"], fontsize=10.5, va="top")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=10, length=0, pad=6)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.75)
    ax.set_axisbelow(True)
    return fig, ax


def label(ax, t, x, y, text, dx, dy, ha, va):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                color=t["ink2"], fontsize=9.5)


def legend(fig, t, handles, labels):
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.835), ncol=len(handles),
               frameon=False, fontsize=10, labelcolor=t["ink2"], handlelength=1.8, columnspacing=1.6)


def accuracy_by_bits(d: dict, t: dict):
    fig, ax = setup(t, "How much accuracy survives quantization",
                    "Spider dev execution accuracy, Qwen3-1.7B. Adapters are fused into the weights before quantizing.")
    xs, gx = range(5), 5.15
    names = ["Base model", "LoRA SFT", "SFT + GRPO"]
    for i, (key, color) in enumerate(zip(["base", "sft", "grpo"], t["series"])):
        ys = d[key]
        ax.plot(xs, ys, color=color, lw=LINE, marker="o", ms=MARKER, mec=t["surface"], mew=RING,
                solid_joinstyle="round", solid_capstyle="round", zorder=3)
        ax.plot([gx], [d["gptq"][i]], color=color, marker="D", ms=MARKER - 0.5, mec=t["surface"], mew=RING,
                ls="none", zorder=3)
        # Direct labels at both ends of each line and on the GPTQ points; SFT and GRPO start 1.5 points
        # apart, so their first labels go above and below.
        dy = {0: 0, 1: -6, 2: 6}[i]
        label(ax, t, 0, ys[0], f"{ys[0]:.1f}%", -9, dy, "right", "center")
        label(ax, t, 4, ys[-1], f"{ys[-1]:.1f}%", 9, 0, "left", "center")
        label(ax, t, gx, d["gptq"][i], f"{d['gptq'][i]:.1f}%", 9, {0: 0, 1: -3, 2: 3}[i], "left", "center")

    ax.axvline(4.58, color=t["grid"], lw=0.75, zorder=1)
    ax.set_xticks([*xs, gx], [*BITS, "3-bit\nGPTQ"])
    ax.set_xlim(-0.75, 5.85)
    ax.set_ylim(0, 82)
    ax.set_yticks(range(0, 81, 20), [f"{v}%" for v in range(0, 81, 20)])
    fig.text(0.012, 0.03, "bf16 to 3-bit: MLX's quantizer.  GPTQ: the from-scratch implementation "
             "(3-bit accuracy; its weights are not packed yet).", color=t["muted"], fontsize=9.5)
    handles = [Line2D([], [], color=c, lw=LINE, marker="o", ms=MARKER - 1, mec=t["surface"]) for c in t["series"]]
    handles.append(Line2D([], [], color=t["muted"], marker="D", ms=MARKER - 1.5, ls="none"))
    legend(fig, t, handles, [*names, "from-scratch GPTQ"])
    return fig


def fuse_vs_adapter(d: dict, t: dict):
    fig, ax = setup(t, "Fuse the adapter before quantizing",
                    "LoRA SFT model: Spider dev accuracy against generation speed. Points are labeled with bits per weight.")
    blue, orange = t["series"][0], t["series"][1]
    # Fused on top: the adapter's 8-bit point lands almost exactly on the fused bf16 point.
    for key, color, z in (("adapter", orange, 3), ("fused", blue, 4)):
        ax.plot(d[f"{key}_tps"], d[f"{key}_acc"], color=color, lw=LINE, marker="o", ms=MARKER,
                mec=t["surface"], mew=RING, solid_joinstyle="round", solid_capstyle="round", zorder=z)

    # Bit-width labels: fused above or right of its points, the adapter's below or left, so the two
    # overlapping points near 75 tok/s each keep a readable label.
    fused_pos = [(0, 9, "center", "bottom")] * 3 + [(9, 4, "left", "bottom"), (9, 0, "left", "center")]
    adapter_pos = [(-9, 0, "right", "center"), (0, -9, "center", "top"), (0, -9, "center", "top"),
                   (-9, 0, "right", "center"), (-9, 0, "right", "center")]
    for key, positions in (("fused", fused_pos), ("adapter", adapter_pos)):
        for bits, x, y, pos in zip(["bf16", "8", "6", "4", "3"], d[f"{key}_tps"], d[f"{key}_acc"], positions):
            label(ax, t, x, y, bits, *pos)

    # The 4-bit comparison: same accuracy, about 2.3x the speed.
    (ax_, ay), (fx, fy) = (d["adapter_tps"][3], d["adapter_acc"][3]), (d["fused_tps"][3], d["fused_acc"][3])
    y_arrow = min(ay, fy) - 4.5
    ax.annotate("", xy=(fx - 4, y_arrow), xytext=(ax_ + 4, y_arrow),
                arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1, mutation_scale=10))
    ax.text((ax_ + fx) / 2, y_arrow - 1.2, f"{fx / ax_:.1f}× faster at 4 bits, same accuracy",
            ha="center", va="top", color=t["ink2"], fontsize=10)

    ax.set_xlim(0, 290)
    ax.set_xticks(range(0, 251, 50))
    ax.set_xlabel("Generation speed (tokens/s, one request at a time, M5 Pro)", color=t["muted"], fontsize=10,
                  labelpad=8)
    ax.set_ylim(27, 77)
    ax.set_yticks(range(30, 71, 10), [f"{v}%" for v in range(30, 71, 10)])
    handles = [Line2D([], [], color=c, lw=LINE, marker="o", ms=MARKER - 1, mec=t["surface"]) for c in (blue, orange)]
    legend(fig, t, handles, ["Fused, then quantized", "Adapter kept separate"])
    return fig


def main() -> None:
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "svg.hashsalt": "shrinkwrap",  # stable element ids, so unchanged charts don't show up in diffs
    })
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw in (("accuracy_by_bits", accuracy_by_bits), ("fuse_vs_adapter", fuse_vs_adapter)):
        for theme, t in THEMES.items():
            fig = draw(data, t)
            path = OUT / f"{name}-{theme}.svg"
            fig.savefig(path, facecolor=t["surface"], metadata={"Date": None})
            plt.close(fig)
            print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

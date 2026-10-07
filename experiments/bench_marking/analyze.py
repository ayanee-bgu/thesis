"""
Analyze and visualize results from ruler_results.json and mint_results.json.

Produces:
  - Terminal summary table
  - Learning curve plots (accuracy vs agent iteration)
  - Knowledge accumulation vs accuracy scatter
  - MINT: turn efficiency over iterations
  - Saved to bench_marking/plots/
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # headless — saves to file without needing a display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

HERE = Path(__file__).parent
PLOTS = HERE / 'plots'
PLOTS.mkdir(exist_ok=True)

RULER_PATH = HERE / 'ruler_results.json'
MINT_PATH  = HERE / 'mint_results.json'


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [missing] {path.name}")
        return None
    return json.loads(path.read_text())


def rolling_accuracy(corrects: list[bool], window: int = 3) -> list[float]:
    out = []
    for i in range(len(corrects)):
        start = max(0, i - window + 1)
        chunk = corrects[start:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _style():
    plt.rcParams.update({
        'figure.dpi': 140,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
    })


# ── Text Summary ──────────────────────────────────────────────────────────────

def print_summary(label: str, data: dict):
    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"{'─'*62}")
    print(f"  {'Task':<22}  {'Acc':>6}  {'Conv':>6}  {'N':>4}")
    print(f"  {'────':<22}  {'───':>6}  {'────':>6}  {'─':>4}")
    for task, r in data.items():
        conv = str(r['convergence_agent']) if r['convergence_agent'] else '—'
        n = len(r['agents'])
        print(f"  {task:<22}  {r['accuracy']:>5.1%}  {conv:>6}  {n:>4}")

    # Per-agent accuracy table
    print()
    all_tasks = list(data.keys())
    header = f"  {'Agent':>5}  " + "  ".join(f"{t[:14]:>14}" for t in all_tasks)
    print(header)
    max_agents = max(len(r['agents']) for r in data.values())
    for i in range(max_agents):
        row = f"  {i+1:>5}  "
        for task in all_tasks:
            agents = data[task]['agents']
            if i < len(agents):
                a = agents[i]
                mark = '✓' if a['correct'] else '✗'
                ki = a.get('knowledge_items', a.get('knowledge_size', 0))
                row += f"  {mark} (K={ki:>2})       "
            else:
                row += f"  {'—':>14}  "
        print(row)


# ── Learning Curve Plot ────────────────────────────────────────────────────────

def plot_learning_curves(label: str, data: dict, fname: str):
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.tab10.colors

    for idx, (task, r) in enumerate(data.items()):
        agents = r['agents']
        xs = [a['agent'] for a in agents]
        corrects = [a['correct'] for a in agents]
        rolling = rolling_accuracy(corrects, window=3)

        ax.plot(xs, rolling, marker='o', markersize=5,
                color=colors[idx % 10], label=task, linewidth=1.8)

        # Mark convergence
        conv = r['convergence_agent']
        if conv:
            ax.axvline(conv, color=colors[idx % 10], linestyle='--', alpha=0.4, linewidth=1)

    ax.set_xlabel('Agent iteration')
    ax.set_ylabel('Rolling accuracy (window=3)')
    ax.set_title(f'{label} — learning curve')
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    out = PLOTS / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Knowledge vs Accuracy ──────────────────────────────────────────────────────

def plot_knowledge_vs_accuracy(label: str, data: dict, fname: str):
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10.colors

    for idx, (task, r) in enumerate(data.items()):
        agents = r['agents']
        ki_field = 'knowledge_items' if 'knowledge_items' in agents[0] else 'knowledge_size'
        xs = [a[ki_field] for a in agents]
        ys = [int(a['correct']) for a in agents]

        # Jitter y slightly for readability
        jitter = np.random.default_rng(idx).uniform(-0.02, 0.02, len(ys))
        ax.scatter(xs, np.array(ys, float) + jitter, alpha=0.7,
                   color=colors[idx % 10], label=task, s=40)

        # Trend line (requires ≥2 distinct x values)
        if len(set(xs)) >= 2:
            z = np.polyfit(xs, ys, 1)
            xr = np.linspace(min(xs), max(xs), 50)
            ax.plot(xr, np.poly1d(z)(xr), color=colors[idx % 10], alpha=0.4, linewidth=1.2)

    ax.set_xlabel('Accumulated knowledge items')
    ax.set_ylabel('Correct (1) / Wrong (0)')
    ax.set_title(f'{label} — knowledge accumulation vs accuracy')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Wrong', 'Correct'])
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOTS / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── MINT: Turn Efficiency ──────────────────────────────────────────────────────

def plot_turn_efficiency(data: dict, fname: str):
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.tab10.colors

    for idx, (task, r) in enumerate(data.items()):
        agents = r['agents']
        xs = [a['agent'] for a in agents]
        turns = [a.get('turns', a.get('turns_used', 0)) for a in agents]
        ax.plot(xs, turns, marker='s', markersize=5,
                color=colors[idx % 10], label=task, linewidth=1.8)

    ax.set_xlabel('Agent iteration')
    ax.set_ylabel('Turns used')
    ax.set_title('MINT — turn efficiency over iterations')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOTS / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Combined Convergence Bar ───────────────────────────────────────────────────

def plot_convergence_bars(ruler_data: dict | None, mint_data: dict | None, fname: str):
    _style()
    labels, values, colors_list = [], [], []
    palette = plt.cm.tab10.colors

    datasets = []
    if ruler_data:
        datasets.append(('RULER', ruler_data, 0))
    if mint_data:
        datasets.append(('MINT', mint_data, 5))

    for bench, data, color_offset in datasets:
        for i, (task, r) in enumerate(data.items()):
            labels.append(f"{bench}\n{task}")
            values.append(r['convergence_agent'] if r['convergence_agent'] else float('nan'))
            colors_list.append(palette[(color_offset + i) % 10])

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.2), 4))
    xs = range(len(labels))
    bars = ax.bar(xs, [v if not (isinstance(v, float) and np.isnan(v)) else 0 for v in values],
                  color=colors_list, width=0.5, edgecolor='white')

    for bar, v in zip(bars, values):
        if isinstance(v, float) and np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.3, 'N/R',
                    ha='center', va='bottom', fontsize=9, color='#888')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, str(int(v)),
                    ha='center', va='bottom', fontsize=9)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Convergence agent (lower = faster learning)')
    ax.set_title('Convergence across benchmarks and task types')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    out = PLOTS / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Experiment Analysis ===")

    ruler = load(RULER_PATH)
    mint  = load(MINT_PATH)

    if ruler:
        print_summary("RULER", ruler)
        plot_learning_curves("RULER", ruler, "ruler_learning_curves.png")
        plot_knowledge_vs_accuracy("RULER", ruler, "ruler_knowledge_vs_accuracy.png")

    if mint:
        print_summary("MINT", mint)
        plot_learning_curves("MINT", mint, "mint_learning_curves.png")
        plot_knowledge_vs_accuracy("MINT", mint, "mint_knowledge_vs_accuracy.png")
        plot_turn_efficiency(mint, "mint_turn_efficiency.png")

    if ruler or mint:
        plot_convergence_bars(ruler, mint, "convergence_comparison.png")

    if not ruler and not mint:
        print("\n  No result files found. Run ruler.py and/or mint.py first.")
    else:
        print(f"\nAll plots saved to {PLOTS}/")


if __name__ == '__main__':
    main()

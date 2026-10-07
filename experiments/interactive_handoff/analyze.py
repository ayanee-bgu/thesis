"""
Analyze and visualize results from handoff_results.json.

Produces:
  - Terminal summary table
  - Detail preservation by condition (bar chart, grouped by task type)
  - Task success rate by condition (bar chart, grouped by task type)
  - Per-agent detail preservation over iterations (line chart by condition)
  - Interactive condition: questions asked per agent (line chart)
  - Hierarchical: supervisor clarification requests by condition (bar chart)
  - Saved to plots/
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

HERE         = Path(__file__).parent
PLOTS        = HERE / 'plots'
RESULTS_PATH = HERE / 'handoff_results.json'
PLOTS.mkdir(exist_ok=True)

CONDITION_LABELS = {
    'none':         'No Handoff',
    'static':       'Static Doc',
    'message_pass': 'Msg Passing',
    'interactive':  'Interactive',
}

CONDITION_COLORS = {
    'none':         '#d9534f',
    'static':       '#f0ad4e',
    'message_pass': '#5bc0de',
    'interactive':  '#5cb85c',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load() -> dict | None:
    if not RESULTS_PATH.exists():
        print(f"[missing] {RESULTS_PATH.name}")
        return None
    return json.loads(RESULTS_PATH.read_text())


def _style():
    plt.rcParams.update({
        'figure.dpi': 140,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
    })


# ── Terminal Summary ──────────────────────────────────────────────────────────

def _get_avg(r: dict, key: str) -> float | None:
    """Return stored average or compute from agents if condition is still running."""
    if key in r:
        return r[key]
    agents = r.get('agents', [])
    if not agents:
        return None
    metric_map = {
        'avg_detail_preservation': 'detail_preservation',
        'avg_task_success': 'task_success',
        'avg_questions_asked': 'questions_asked',
        'avg_clarification_requests': 'clarification_requests',
    }
    field = metric_map.get(key)
    if not field:
        return None
    vals = [a[field] for a in agents if field in a]
    return sum(vals) / len(vals) if vals else None


def print_summary(data: dict):
    print(f"\n{'─'*72}")
    print("  Interactive Handoff Experiment — Results Summary")
    print(f"{'─'*72}")
    for task_type, cond_results in data.items():
        print(f"\n  {task_type.replace('_', ' ').upper()}")
        print(f"  {'Condition':<16} {'n':>3} {'Detail Pres':>13} {'Success Rate':>13} {'Avg Q':>7} {'Avg CR':>7}")
        print(f"  {'─'*16} {'─'*3} {'─'*13} {'─'*13} {'─'*7} {'─'*7}")
        for cond, r in cond_results.items():
            label     = CONDITION_LABELS.get(cond, cond)
            completed = r.get('completed', False)
            n         = len(r.get('agents', []))
            suffix    = '' if completed else ' *'
            dp_val    = _get_avg(r, 'avg_detail_preservation')
            ts_val    = _get_avg(r, 'avg_task_success')
            q_val     = _get_avg(r, 'avg_questions_asked')
            cr_val    = _get_avg(r, 'avg_clarification_requests')
            dp = f"{dp_val:.2f}{suffix}" if dp_val is not None else "—"
            ts = f"{ts_val:.1%}{suffix}" if ts_val is not None else "—"
            q  = f"{q_val:.1f}" if q_val is not None else "—"
            cr = f"{cr_val:.1f}" if cr_val is not None else "—"
            print(f"  {label:<16} {n:>3} {dp:>13} {ts:>13} {q:>7} {cr:>7}")
    print("  (* = in progress, partial data)")

    # Per-agent breakdown
    print(f"\n{'─'*72}")
    print("  Per-agent detail preservation")
    print(f"{'─'*72}")
    for task_type, cond_results in data.items():
        print(f"\n  {task_type.replace('_', ' ').upper()}")
        conditions = list(cond_results.keys())
        max_agents = max(len(r['agents']) for r in cond_results.values()) if cond_results else 0
        header = f"  {'Agent':>5}  " + "  ".join(f"{CONDITION_LABELS.get(c,c):>13}" for c in conditions)
        print(header)
        for i in range(max_agents):
            row = f"  {i+1:>5}  "
            for cond in conditions:
                agents = cond_results[cond]['agents']
                if i < len(agents):
                    a   = agents[i]
                    dp  = a['detail_preservation']
                    ts  = '✓' if a['task_success'] else '✗'
                    row += f"  {dp:.2f} {ts}        "
                else:
                    row += f"  {'—':>13}  "
            print(row)


# ── Grouped Bar Chart ─────────────────────────────────────────────────────────

def plot_grouped_bars(data: dict, metric: str, ylabel: str, title: str, fname: str):
    _style()
    task_types = list(data.keys())
    conditions = list(next(iter(data.values())).keys())

    x     = np.arange(len(task_types))
    width = 0.8 / len(conditions)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, cond in enumerate(conditions):
        values = [_get_avg(data[tt].get(cond, {}), metric) or 0 for tt in task_types]
        offset = (i - len(conditions) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width * 0.9,
                      label=CONDITION_LABELS.get(cond, cond),
                      color=CONDITION_COLORS.get(cond, '#aaaaaa'),
                      edgecolor='white')
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels([tt.replace('_', '\n') for tt in task_types])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    out = PLOTS / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Per-Agent Line Chart ──────────────────────────────────────────────────────

def plot_per_agent_detail(data: dict, fname: str):
    _style()
    task_types = list(data.keys())
    n = len(task_types)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    conditions = list(next(iter(data.values())).keys())

    for ax, task_type in zip(axes, task_types):
        for cond in conditions:
            r = data[task_type].get(cond, {})
            agents = r.get('agents', [])
            if not agents:
                continue
            xs = [a['agent'] for a in agents]
            ys = [a['detail_preservation'] for a in agents]
            ax.plot(xs, ys, marker='o', markersize=4, linewidth=1.6,
                    label=CONDITION_LABELS.get(cond, cond),
                    color=CONDITION_COLORS.get(cond, '#999999'))

        ax.set_title(task_type.replace('_', '\n'), fontsize=9)
        ax.set_xlabel('Agent iteration')
        ax.set_ylim(-0.05, 1.1)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[0].set_ylabel('Detail preservation score')
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, fontsize=8, loc='lower right')
    fig.suptitle('Detail preservation per agent — by handoff condition', fontsize=11)
    fig.tight_layout()
    out = PLOTS / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Interactive Questions Chart ───────────────────────────────────────────────

def plot_interactive_questions(data: dict, fname: str):
    _style()
    task_types = [tt for tt in data if data[tt].get('interactive', {}).get('agents')]
    if not task_types:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, task_type in enumerate(task_types):
        agents = data[task_type]['interactive']['agents']
        xs = [a['agent'] for a in agents]
        qs = [a['questions_asked'] for a in agents]
        ax.plot(xs, qs, marker='s', markersize=5, linewidth=1.5,
                label=task_type.replace('_', ' '),
                color=plt.cm.tab10.colors[i % 10])

    ax.set_xlabel('Agent iteration')
    ax.set_ylabel('Questions asked by Worker B')
    ax.set_title('Interactive condition — questions per agent pair')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOTS / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Clarification Requests Bar ────────────────────────────────────────────────

def plot_clarification_requests(data: dict, fname: str):
    _style()
    if 'hierarchical_orchestration' not in data:
        return
    h = data['hierarchical_orchestration']
    conditions = list(h.keys())
    values     = [h[c].get('avg_clarification_requests', 0) or 0 for c in conditions]
    colors     = [CONDITION_COLORS.get(c, '#aaaaaa') for c in conditions]
    labels     = [CONDITION_LABELS.get(c, c) for c in conditions]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Avg supervisor escalations per agent pair')
    ax.set_title('Hierarchical orchestration — supervisor clarification requests\n(lower is better)')
    fig.tight_layout()
    out = PLOTS / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Condition Lift Summary ────────────────────────────────────────────────────

def print_lift_summary(data: dict):
    """Print the gain of interactive vs no-handoff baseline."""
    print(f"\n{'─'*72}")
    print("  Lift: interactive vs no-handoff baseline")
    print(f"{'─'*72}")
    for task_type, cond_results in data.items():
        none_dp = _get_avg(cond_results.get('none', {}), 'avg_detail_preservation') or 0
        intr_dp = _get_avg(cond_results.get('interactive', {}), 'avg_detail_preservation') or 0
        none_ts = _get_avg(cond_results.get('none', {}), 'avg_task_success') or 0
        intr_ts = _get_avg(cond_results.get('interactive', {}), 'avg_task_success') or 0
        intr_complete = cond_results.get('interactive', {}).get('completed', False)
        suffix = '' if intr_complete else ' (partial)'
        print(f"\n  {task_type.replace('_', ' ').upper()}")
        print(f"    detail preservation: none={none_dp:.2f}  interactive={intr_dp:.2f}{suffix}  Δ={intr_dp - none_dp:+.2f}")
        print(f"    task success:        none={none_ts:.1%}  interactive={intr_ts:.1%}{suffix}  Δ={intr_ts - none_ts:+.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Interactive Handoff Experiment Analysis ===")
    data = load()
    if not data:
        print("No results found. Run main.py first.")
        return

    print_summary(data)
    print_lift_summary(data)

    plot_grouped_bars(
        data, 'avg_detail_preservation',
        'Avg detail preservation (0–1)',
        'Detail preservation by handoff condition',
        'detail_preservation_by_condition.png',
    )
    plot_grouped_bars(
        data, 'avg_task_success',
        'Task success rate (0–1)',
        'Task success rate by handoff condition',
        'task_success_by_condition.png',
    )
    plot_per_agent_detail(data, 'per_agent_detail_preservation.png')
    plot_interactive_questions(data, 'interactive_questions_per_agent.png')
    plot_clarification_requests(data, 'clarification_requests_by_condition.png')

    print(f"\nAll plots saved to {PLOTS}/")


if __name__ == '__main__':
    main()

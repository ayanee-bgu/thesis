"""
Merge n=2 base results with n=1 offset results to produce combined n=3 results.

Usage:
    python experiments/memory_controller/merge_results.py
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "memory_controller"
CONDITIONS = ["no_memory", "memp_flat", "graph_pertraj", "graph_session"]


def merge(cond: str) -> dict | None:
    base_path = RESULTS_DIR / f"{cond}.json"
    ext_path = RESULTS_DIR / f"{cond}_offset4.json"

    if not base_path.exists():
        print(f"  Missing base: {base_path}")
        return None
    if not ext_path.exists():
        print(f"  Missing offset file: {ext_path} — offset run not yet complete")
        return None

    base = json.loads(base_path.read_text())
    ext = json.loads(ext_path.read_text())

    # Merge task lists: base has S1=[0,1], S2=[2,3]; ext has S1=[4], S2=[5]
    # Result: S1=[0,1,4], S2=[2,3,5]
    all_tasks = base["tasks"] + ext["tasks"]

    # Recompute per-session summaries
    sessions: dict[int, list] = {}
    for t in all_tasks:
        s = t["session"]
        sessions.setdefault(s, []).append(t)

    session_summaries = []
    for snum in sorted(sessions):
        tasks = sessions[snum]
        sr = sum(t["success"] for t in tasks) / len(tasks)
        session_summaries.append({
            "session": snum,
            "num_tasks": len(tasks),
            "success_rate": sr,
            "mean_tokens": sum(t["tokens"] for t in tasks) / len(tasks),
            "retrieval_calls": sum(t["retrieval_calls"] for t in tasks),
            "graph_nodes_after": max(
                (t.get("graph_nodes_after") or 0 for t in tasks), default=None
            ) or None,
        })

    delta = None
    if len(session_summaries) >= 2:
        delta = session_summaries[1]["success_rate"] - session_summaries[0]["success_rate"]

    merged = {
        "condition": cond,
        "memory_type": base["memory_type"],
        "update_policy": base["update_policy"],
        "n_per_session": len(sessions.get(1, [])),
        "sessions": session_summaries,
        "tasks": all_tasks,
        "summary": {
            "session_1_success_rate": session_summaries[0]["success_rate"] if session_summaries else None,
            "session_2_success_rate": session_summaries[1]["success_rate"] if len(session_summaries) > 1 else None,
            "delta_s1_to_s2": delta,
            "mean_tokens_s2": session_summaries[1]["mean_tokens"] if len(session_summaries) > 1 else None,
            "total_retrieval_calls_s2": session_summaries[1]["retrieval_calls"] if len(session_summaries) > 1 else None,
            "final_graph_nodes": ext.get("summary", {}).get("final_graph_nodes"),
        },
    }

    out_path = RESULTS_DIR / f"{cond}_n3.json"
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"  {cond}: S1={session_summaries[0]['success_rate']:.0%} (n={session_summaries[0]['num_tasks']}) "
          f"→ S2={session_summaries[1]['success_rate']:.0%} (n={session_summaries[1]['num_tasks']}) "
          f"Δ={delta:+.0%}" if delta is not None else f"  {cond}: merged")
    return merged


def main():
    print("Merging base (n=2) + offset (n=1) results into n=3...\n")
    results = []
    for cond in CONDITIONS:
        r = merge(cond)
        if r:
            results.append(r)

    if not results:
        print("Nothing merged yet — offset run may still be in progress.")
        return

    print(f"\n{'='*70}")
    print(f"{'Condition':<22} {'S1 (n=3)':>10} {'S2 (n=3)':>10} {'Delta':>8} {'S2 Tokens':>10} {'Retrieved S2':>13}")
    print("-" * 70)
    for r in results:
        s = r["summary"]
        s1 = s.get("session_1_success_rate", 0) or 0
        s2 = s.get("session_2_success_rate", 0) or 0
        d = s.get("delta_s1_to_s2")
        tok = s.get("mean_tokens_s2") or 0
        rc = s.get("total_retrieval_calls_s2") or 0
        d_str = f"{d:+.0%}" if d is not None else "n/a"
        print(f"{r['condition']:<22} {s1:>9.0%} {s2:>10.0%} {d_str:>8} {tok:>10.0f} {rc:>13}")

    # Save combined summary
    summary_path = RESULTS_DIR / "summary_n3.json"
    summary_path.write_text(json.dumps({"n_per_session": 3, "conditions": [r["summary"] | {"name": r["condition"]} for r in results]}, indent=2))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()

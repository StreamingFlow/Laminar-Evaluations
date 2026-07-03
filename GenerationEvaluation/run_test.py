#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import statistics
import time
from collections import Counter
from datetime import datetime
from typing import Callable, Iterable

BASE_KIND = {
    "ProducerPE": "producer",
    "IterativePE": "iterative",
    "ConsumerPE": "consumer",
    "GenericPE": "generic",
    "SimpleFunctionPE": "iterative",
}


def _parse(src: str):
    try:
        return ast.parse(src, mode="exec")
    except SyntaxError:
        return None


def _class_roles(tree):
    roles = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            for b in n.bases:
                bn = b.id if isinstance(b, ast.Name) else getattr(b, "attr", None)
                if bn in BASE_KIND:
                    roles[n.name] = BASE_KIND[bn]
    return roles


def _var_to_class(tree, class_roles):
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            f = n.value.func
            cname = f.id if isinstance(f, ast.Name) else None
            if cname in class_roles and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                out[n.targets[0].id] = cname
    return out


def _connect_calls(tree):
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "connect" and len(n.args) >= 4):
            a = n.args
            name = lambda x: x.id if isinstance(x, ast.Name) else None
            const = lambda x: x.value if isinstance(x, ast.Constant) else None
            yield name(a[0]), const(a[1]), name(a[2]), const(a[3])


def extract_graph(src: str) -> dict:
    tree = _parse(src)
    if tree is None:
        return {"parse_ok": False, "role_edges": set(), "topo_edges": frozenset(),
                "n_stages": 0, "roles": Counter()}
    class_roles = _class_roles(tree)
    var_class = _var_to_class(tree, class_roles)
    role_edges, order, topo = set(), [], []
    for s, sp, d, dp in _connect_calls(tree):
        role_edges.add((class_roles.get(var_class.get(s)),
                        class_roles.get(var_class.get(d)), sp, dp))
        for v in (s, d):
            if v and v not in order:
                order.append(v)
        if s in order and d in order:
            topo.append((order.index(s), sp, order.index(d), dp))
    return {"parse_ok": True, "roles": Counter(class_roles.values()),
            "role_edges": role_edges, "topo_edges": frozenset(topo),
            "n_stages": len(order)}


def _f1(gen: Iterable, truth: Iterable) -> float:
    g, t = Counter(gen), Counter(truth)
    inter = sum((g & t).values())
    p = inter / max(sum(g.values()), 1)
    r = inter / max(sum(t.values()), 1)
    return 0.0 if p + r == 0 else round(2 * p * r / (p + r), 4)


def structural_scores(answer: str, ground_truth: str) -> dict:
    g = extract_graph(answer)
    t = extract_graph(ground_truth)
    if not g["parse_ok"]:
        return {"parse_ok": 0.0, "topology_f1": 0.0, "role_edge_f1": 0.0, "stage_count_ok": 0.0}
    return {"parse_ok": 1.0,
            "topology_f1": _f1(g["topo_edges"], t["topo_edges"]),
            "role_edge_f1": _f1(g["role_edges"], t["role_edges"]),
            "stage_count_ok": float(g["n_stages"] == t["n_stages"])}


def structural_scores_multi(answer: str, ground_truths: list[str]) -> dict:
    best = {"parse_ok": 0.0, "topology_f1": 0.0, "role_edge_f1": 0.0, "stage_count_ok": 0.0}
    for gt in ground_truths:
        s = structural_scores(answer, gt)
        if (s["topology_f1"], s["role_edge_f1"]) > (best["topology_f1"], best["role_edge_f1"]):
            best = s
    return best


# --------------------------------------------------------------------------- #
# Evaluation driver                                                           #
# --------------------------------------------------------------------------- #
# Result status buckets:
#   OK          generation returned parseable code, metrics computed
#   SYNTAX_ERR  generation returned code but it did not parse (parse_ok == 0)
#   EMPTY       generation returned nothing usable
#   GEN_ERROR   generation raised an exception
STATUS_OK, STATUS_SYNTAX, STATUS_EMPTY, STATUS_GEN = "OK", "SYNTAX_ERR", "EMPTY", "GEN_ERROR"


def run_evaluation(generate_fn: Callable[[str], str], truth_dir: str, descr_dir: str,
                   progress_cb: Callable[[dict], None] | None = None) -> list[dict]:
    results = []
    truth_files = sorted(f for f in os.listdir(truth_dir) if f.endswith(".py"))
    for py in truth_files:
        task = py[:-3]
        row = {"task": task, "status": None, "error": "", "seconds": 0.0,
               "parse_ok": 0.0, "topology_f1": 0.0, "role_edge_f1": 0.0, "stage_count_ok": 0.0}
        try:
            with open(os.path.join(truth_dir, py)) as f:
                truth_code = f.read()
            with open(os.path.join(descr_dir, task + ".txt")) as f:
                description = f.read()
        except FileNotFoundError as e:
            row["status"], row["error"] = STATUS_GEN, f"missing input: {e.filename}"
            results.append(row)
            if progress_cb:
                progress_cb(row)
            continue

        t0 = time.perf_counter()
        try:
            code = generate_fn(description)
            row["seconds"] = time.perf_counter() - t0
            if not code or not str(code).strip():
                row["status"] = STATUS_EMPTY
            else:
                scores = structural_scores_multi(str(code), [truth_code])
                row.update(scores)
                row["status"] = STATUS_OK if scores["parse_ok"] else STATUS_SYNTAX
        except Exception as e:  # generation blew up
            row["seconds"] = time.perf_counter() - t0
            row["status"], row["error"] = STATUS_GEN, f"{type(e).__name__}: {e}"
        results.append(row)
        if progress_cb:
            progress_cb(row)
    return results


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    counts = Counter(r["status"] for r in results)
    parsed = [r for r in results if r["parse_ok"] == 1.0]
    def mean(key, rows):
        vals = [r[key] for r in rows]
        return statistics.fmean(vals) if vals else 0.0
    return {
        "n_tasks": n,
        "counts": dict(counts),
        "syntactic_validity_rate": (counts[STATUS_OK] + counts[STATUS_SYNTAX] and
                                    counts[STATUS_OK] / (counts[STATUS_OK] + counts[STATUS_SYNTAX])) or 0.0,
        "mean_topology_f1": mean("topology_f1", parsed),
        "median_topology_f1": statistics.median([r["topology_f1"] for r in parsed]) if parsed else 0.0,
        "mean_role_edge_f1": mean("role_edge_f1", parsed),
        "stage_count_pass_rate": mean("stage_count_ok", parsed),
        "total_seconds": sum(r["seconds"] for r in results),
        "n_parsed": len(parsed),
    }


try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False

_STATUS_STYLE = {STATUS_OK: "green", STATUS_SYNTAX: "yellow",
                 STATUS_EMPTY: "yellow", STATUS_GEN: "red"}


def _f1_style(v: float) -> str:
    return "green" if v >= 0.8 else "yellow" if v >= 0.5 else "red"


def _check(v: float) -> str:
    return "✓" if v == 1.0 else "✗"


def render_rich(results: list[dict], agg: dict) -> None:
    console = Console()
    table = Table(title="dispel4py workflow generation — structural evaluation",
                  box=box.SIMPLE_HEAVY, header_style="bold")
    table.add_column("Task", no_wrap=True)
    table.add_column("Status", no_wrap=True, min_width=10)
    table.add_column("parse", justify="center")
    table.add_column("topo_f1", justify="right")
    table.add_column("role_f1", justify="right")
    table.add_column("stages", justify="center")
    table.add_column("time", justify="right")
    for r in results:
        ok = r["parse_ok"] == 1.0
        table.add_row(
            r["task"],
            f"[{_STATUS_STYLE[r['status']]}]{r['status']}[/]",
            f"[{'green' if ok else 'red'}]{_check(r['parse_ok'])}[/]",
            f"[{_f1_style(r['topology_f1'])}]{r['topology_f1']:.3f}[/]" if ok else "[dim]—[/]",
            f"[{_f1_style(r['role_edge_f1'])}]{r['role_edge_f1']:.3f}[/]" if ok else "[dim]—[/]",
            f"[{'green' if r['stage_count_ok'] else 'red'}]{_check(r['stage_count_ok'])}[/]" if ok else "[dim]—[/]",
            f"{r['seconds']:.1f}s",
        )
    console.print(table)

    c = agg["counts"]
    summary = (
        f"[bold]Tasks:[/] {agg['n_tasks']}   "
        f"[green]OK {c.get(STATUS_OK,0)}[/]  "
        f"[yellow]syntax {c.get(STATUS_SYNTAX,0)}[/]  "
        f"[yellow]empty {c.get(STATUS_EMPTY,0)}[/]  "
        f"[red]gen-error {c.get(STATUS_GEN,0)}[/]\n"
        f"[bold]Syntactic validity:[/] {agg['syntactic_validity_rate']*100:.0f}%   "
        f"[bold]Stage-count pass:[/] {agg['stage_count_pass_rate']*100:.0f}%\n"
        f"[bold]topology_f1[/] mean [{_f1_style(agg['mean_topology_f1'])}]{agg['mean_topology_f1']:.3f}[/] "
        f"median {agg['median_topology_f1']:.3f}   "
        f"[bold]role_edge_f1[/] mean [{_f1_style(agg['mean_role_edge_f1'])}]{agg['mean_role_edge_f1']:.3f}[/]\n"
        f"[dim](means over the {agg['n_parsed']} parsed generations · total {agg['total_seconds']:.1f}s)[/]"
    )
    console.print(Panel(summary, title="Summary", border_style="cyan", expand=False))


def render_plain(results: list[dict], agg: dict) -> None:
    cols = [("Task", 30), ("Status", 11), ("parse", 6),
            ("topo_f1", 8), ("role_f1", 8), ("stages", 7), ("time", 7)]
    header = "  ".join(name.ljust(w) for name, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        ok = r["parse_ok"] == 1.0
        cells = [
            r["task"][:30].ljust(30),
            r["status"].ljust(11),
            _check(r["parse_ok"]).center(6),
            (f"{r['topology_f1']:.3f}" if ok else "—").rjust(8),
            (f"{r['role_edge_f1']:.3f}" if ok else "—").rjust(8),
            (_check(r["stage_count_ok"]) if ok else "—").center(7),
            f"{r['seconds']:.1f}s".rjust(7),
        ]
        print("  ".join(cells))
        if r["error"]:
            print(f"    └─ {r['error']}")
    c = agg["counts"]
    print("\n" + "=" * len(header))
    print(f"Tasks: {agg['n_tasks']}   OK {c.get(STATUS_OK,0)}  "
          f"syntax {c.get(STATUS_SYNTAX,0)}  empty {c.get(STATUS_EMPTY,0)}  "
          f"gen-error {c.get(STATUS_GEN,0)}")
    print(f"Syntactic validity: {agg['syntactic_validity_rate']*100:.0f}%   "
          f"Stage-count pass: {agg['stage_count_pass_rate']*100:.0f}%")
    print(f"topology_f1  mean {agg['mean_topology_f1']:.3f}  median {agg['median_topology_f1']:.3f}   "
          f"role_edge_f1 mean {agg['mean_role_edge_f1']:.3f}")
    print(f"(means over {agg['n_parsed']} parsed generations · total {agg['total_seconds']:.1f}s)")


def render(results, agg):
    (render_rich if _HAS_RICH else render_plain)(results, agg)


def save_reports(results: list[dict], agg: dict, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"eval_{stamp}.json")
    csv_path = os.path.join(out_dir, f"eval_{stamp}.csv")
    with open(json_path, "w") as f:
        json.dump({"generated_at": stamp, "summary": agg, "results": results}, f, indent=2)
    fields = ["task", "status", "parse_ok", "topology_f1", "role_edge_f1",
              "stage_count_ok", "seconds", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})
    return json_path, csv_path


def generate_workflow_wrap():
    """Build a generate_fn backed by the real Laminar client (requires login)."""
    from laminar.client.d4pyclient import d4pClient
    from laminar.clitools.advanced_search import AdvancedSearchCommand
    from pwinput import pwinput

    client = d4pClient()
    username = os.environ.get("LAMINAR_USERNAME") or input("Username: ")
    password = os.environ.get("LAMINAR_PASSWORD") or pwinput("Password: ")
    client.login(username, password)
    command = AdvancedSearchCommand(client=client)

    def _gen(description: str) -> str:
        return command._generate(query=description, silent=True)["workflow_code"]

    return _gen


def main():
    ap = argparse.ArgumentParser(description="Structural evaluation of generated dispel4py workflows.")
    ap.add_argument("--truth-dir", default="./truth")
    ap.add_argument("--descr-dir", default="./descr")
    ap.add_argument("--out", default="./reports", help="directory for JSON/CSV reports")
    ap.add_argument("--demo", action="store_true", help="use offline fixtures, no login")
    ap.add_argument("--no-save", action="store_true", help="do not write report files")
    args = ap.parse_args()

    generate_fn = generate_workflow_wrap()

    live = None
    if _HAS_RICH:
        from rich.console import Console
        live = Console()
        live.print(f"[cyan]Running {len([f for f in os.listdir(args.truth_dir) if f.endswith('.py')])} "
                   f"tasks{' (demo)' if args.demo else ''}…[/]")

    def _tick(row):
        mark = {"OK": "✓", "SYNTAX_ERR": "!", "EMPTY": "·", "GEN_ERROR": "✗"}.get(row["status"], "?")
        line = f"  [{mark}] {row['task']:<30} {row['status']:<11} topo={row['topology_f1']:.3f}"
        (live.print(line) if live else print(line))

    results = run_evaluation(generate_fn, args.truth_dir, args.descr_dir, progress_cb=_tick)
    agg = aggregate(results)
    render(results, agg)

    if not args.no_save:
        jp, cp = save_reports(results, agg, args.out)
        msg = f"\nSaved: {jp}\n       {cp}"
        (live.print(f"[dim]{msg}[/]") if live else print(msg))


if __name__ == "__main__":
    main()
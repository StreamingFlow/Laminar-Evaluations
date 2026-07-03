#!/usr/bin/env python3
"""
Structural evaluation of generated dispel4py workflows — Textual edition.

Adds on top of the original script:
  * repeated runs (--runs N) with mean ± standard deviation on every measure,
    both per-task (across runs) and in the overall summary;
  * a Textual TUI whose right-hand pane is a dedicated window that captures
    everything the generation function prints to stdout;
  * a --headless mode that does the same multi-run evaluation without a TUI.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import statistics
import tempfile
import time
from collections import Counter, OrderedDict
from contextlib import redirect_stdout
from datetime import datetime
from typing import Callable, Iterable

BASE_KIND = {
    "ProducerPE": "producer",
    "IterativePE": "iterative",
    "ConsumerPE": "consumer",
    "GenericPE": "generic",
    "SimpleFunctionPE": "iterative",
}


# --------------------------------------------------------------------------- #
# Graph extraction + structural scoring  (unchanged from the original)        #
# --------------------------------------------------------------------------- #
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
# Single-run evaluation driver  (unchanged from the original)                 #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Multi-run helpers: mean ± standard deviation across repeated runs            #
# --------------------------------------------------------------------------- #
def mean_or0(vals) -> float:
    vals = list(vals)
    return statistics.fmean(vals) if vals else 0.0


def std_or0(vals) -> float:
    """Sample standard deviation; 0.0 when fewer than two data points."""
    vals = list(vals)
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def run_repeated(generate_fn, truth_dir, descr_dir, runs: int,
                 task_cb: Callable[[int, dict], None] | None = None,
                 run_done_cb: Callable[[int, list[dict]], None] | None = None) -> list[list[dict]]:
    """Run the whole evaluation `runs` times. Returns a list of per-run result lists."""
    all_results: list[list[dict]] = []
    for ri in range(runs):
        cb = (lambda row, ri=ri: task_cb(ri, row)) if task_cb else None
        res = run_evaluation(generate_fn, truth_dir, descr_dir, progress_cb=cb)
        all_results.append(res)
        if run_done_cb:
            run_done_cb(ri, res)
    return all_results


def per_task_stats(all_results: list[list[dict]]) -> list[dict]:
    """Aggregate each task across runs into mean ± std of every measure."""
    by_task: "OrderedDict[str, list[dict]]" = OrderedDict()
    for run in all_results:
        for row in run:
            by_task.setdefault(row["task"], []).append(row)

    out = []
    for task, rows in by_task.items():
        parsed = [r for r in rows if r["parse_ok"] == 1.0]
        topo = [r["topology_f1"] for r in parsed]
        role = [r["role_edge_f1"] for r in parsed]
        stage = [r["stage_count_ok"] for r in parsed]
        secs = [r["seconds"] for r in rows]
        out.append({
            "task": task,
            "n": len(rows),
            "ok": sum(1 for r in rows if r["status"] == STATUS_OK),
            "parse_rate": mean_or0([r["parse_ok"] for r in rows]),
            "topo_mean": mean_or0(topo), "topo_std": std_or0(topo),
            "role_mean": mean_or0(role), "role_std": std_or0(role),
            "stage_rate": mean_or0(stage),
            "time_mean": mean_or0(secs), "time_std": std_or0(secs),
        })
    return out


# Overall summary metrics we track run-to-run.
_SUMMARY_KEYS = ["syntactic_validity_rate", "mean_topology_f1", "median_topology_f1",
                 "mean_role_edge_f1", "stage_count_pass_rate", "total_seconds", "n_parsed"]


def summarize_runs(all_results: list[list[dict]]) -> dict:
    """Compute per-run aggregates, then mean ± std of each metric across runs."""
    per_run = [aggregate(r) for r in all_results]
    metrics = {}
    for k in _SUMMARY_KEYS:
        vals = [a[k] for a in per_run]
        metrics[k] = {"mean": mean_or0(vals), "std": std_or0(vals), "values": vals}
    return {"n_runs": len(all_results), "metrics": metrics,
            "per_run_counts": [a["counts"] for a in per_run]}


# --------------------------------------------------------------------------- #
# Report saving (multi-run)                                                    #
# --------------------------------------------------------------------------- #
def save_multi_reports(all_results: list[list[dict]], out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"eval_{stamp}.json")
    csv_path = os.path.join(out_dir, f"eval_{stamp}.csv")

    task_stats = per_task_stats(all_results)
    summary = summarize_runs(all_results)

    with open(json_path, "w") as f:
        json.dump({"generated_at": stamp, "n_runs": len(all_results),
                   "summary": summary, "per_task": task_stats,
                   "runs": all_results}, f, indent=2)

    fields = ["task", "n", "ok", "parse_rate", "topo_mean", "topo_std",
              "role_mean", "role_std", "stage_rate", "time_mean", "time_std"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in task_stats:
            w.writerow({k: s.get(k, "") for k in fields})
    return json_path, csv_path


# --------------------------------------------------------------------------- #
# Generation backends                                                         #
# --------------------------------------------------------------------------- #
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


def _linear_workflow_code(roles: list[str]) -> str:
    """Emit a simple linear dispel4py workflow from a list of base-class roles."""
    lines, names = [], []
    for i, r in enumerate(roles):
        cls = f"Stage{i}"
        lines.append(f"class {cls}({r}):\n    pass\n")
        names.append((cls, f"s{i}"))
    lines.append("graph = WorkflowGraph()")
    for cls, var in names:
        lines.append(f"{var} = {cls}()")
    for i in range(len(names) - 1):
        lines.append(f"graph.connect({names[i][1]}, 'output', {names[i + 1][1]}, 'input')")
    return "\n".join(lines) + "\n"


def make_demo_generate_fn(seed: int | None = None):
    """
    Offline generator for --demo. Prints chatty progress to stdout (to exercise
    the stdout window) and returns a slightly randomised workflow so that scores
    vary run-to-run (to exercise the mean ± std reporting).
    """
    rng = random.Random(seed)

    def _gen(description: str) -> str:
        print(f"[gen] prompt received ({len(description)} chars)")
        for step in ("tokenizing", "planning stages", "wiring edges", "serializing"):
            print(f"[gen] {step} …")
            time.sleep(0.04)

        n = rng.choice([2, 3, 3, 3, 4])
        roles = ["ProducerPE"] + ["IterativePE"] * (n - 2) + ["ConsumerPE"]
        if rng.random() < 0.20:                 # occasional wrong role
            roles[rng.randrange(n)] = "GenericPE"

        drop_edge = rng.random() < 0.20         # occasional missing edge
        lines, names = [], []
        for i, r in enumerate(roles):
            cls = f"Stage{i}"
            lines.append(f"class {cls}({r}):\n    pass\n")
            names.append(f"s{i}")
            lines[-1] += f"{names[-1]} = {cls}()"
        lines.insert(len(roles), "graph = WorkflowGraph()")
        skip = rng.randrange(max(1, len(names) - 1)) if drop_edge else -1
        for i in range(len(names) - 1):
            if i == skip:
                continue
            lines.append(f"graph.connect({names[i]}, 'output', {names[i + 1]}, 'input')")
        print("[gen] complete")
        return "\n".join(lines) + "\n"

    return _gen


def make_demo_fixtures(truth_dir: str, descr_dir: str) -> None:
    """Write a small self-contained set of truth/description files for --demo."""
    os.makedirs(truth_dir, exist_ok=True)
    os.makedirs(descr_dir, exist_ok=True)
    specs = {
        "wordcount": ["ProducerPE", "IterativePE", "ConsumerPE"],
        "etl_pipe":  ["ProducerPE", "IterativePE", "IterativePE", "ConsumerPE"],
        "sensor_avg": ["ProducerPE", "IterativePE", "ConsumerPE"],
        "two_stage": ["ProducerPE", "ConsumerPE"],
        "wide_map":  ["ProducerPE", "IterativePE", "IterativePE", "IterativePE", "ConsumerPE"],
    }
    for name, roles in specs.items():
        with open(os.path.join(truth_dir, name + ".py"), "w") as f:
            f.write(_linear_workflow_code(roles))
        with open(os.path.join(descr_dir, name + ".txt"), "w") as f:
            f.write(f"Build a dispel4py workflow '{name}' with {len(roles)} stages: "
                    + " -> ".join(roles) + ".")


# --------------------------------------------------------------------------- #
# Headless multi-run runner                                                   #
# --------------------------------------------------------------------------- #
def print_multi_summary(all_results: list[list[dict]]) -> None:
    stats = per_task_stats(all_results)
    summ = summarize_runs(all_results)
    cols = [("Task", 26), ("n", 4), ("OK", 4), ("parse%", 7),
            ("topo_f1 μ±σ", 16), ("role_f1 μ±σ", 16), ("stage%", 7), ("time μ", 8)]
    header = "  ".join(name.ljust(w) for name, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for s in stats:
        print("  ".join([
            s["task"][:26].ljust(26),
            str(s["n"]).ljust(4),
            str(s["ok"]).ljust(4),
            f"{s['parse_rate']*100:.0f}%".ljust(7),
            f"{s['topo_mean']:.3f}±{s['topo_std']:.3f}".ljust(16),
            f"{s['role_mean']:.3f}±{s['role_std']:.3f}".ljust(16),
            f"{s['stage_rate']*100:.0f}%".ljust(7),
            f"{s['time_mean']:.1f}s".ljust(8),
        ]))
    m = summ["metrics"]

    def pm(key, pct=False, prec=3):
        d = m[key]
        return (f"{d['mean']*100:.0f}%±{d['std']*100:.0f}%" if pct
                else f"{d['mean']:.{prec}f}±{d['std']:.{prec}f}")

    print("\n" + "=" * len(header))
    print(f"Runs: {summ['n_runs']}   "
          f"validity {pm('syntactic_validity_rate', pct=True)}   "
          f"stage-pass {pm('stage_count_pass_rate', pct=True)}")
    print(f"topology_f1 {pm('mean_topology_f1')}   role_edge_f1 {pm('mean_role_edge_f1')}   "
          f"time/run {pm('total_seconds', prec=1)}s")


def run_headless(generate_fn, truth_dir, descr_dir, runs, out_dir, save):
    def task_cb(ri, row):
        print(f"  run {ri+1}/{runs} [{row['status']:<10}] {row['task']:<26} "
              f"topo={row['topology_f1']:.3f} ({row['seconds']:.1f}s)")

    def run_done_cb(ri, res):
        print(f"# finished run {ri+1}/{runs}")

    all_results = run_repeated(generate_fn, truth_dir, descr_dir, runs,
                               task_cb=task_cb, run_done_cb=run_done_cb)
    print_multi_summary(all_results)
    if save:
        jp, cp = save_multi_reports(all_results, out_dir)
        print(f"\nSaved: {jp}\n       {cp}")


# --------------------------------------------------------------------------- #
# Textual UI                                                                  #
# --------------------------------------------------------------------------- #
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (Header, Footer, DataTable, Log, RichLog, Static,
                             Button, Input, ProgressBar, Label)
from textual import work
from rich.text import Text


class UIStream:
    """
    A file-like object standing in for stdout during evaluation. Buffers by line
    and pushes complete lines to a Textual widget from the worker thread.
    """
    def __init__(self, app: App, sink: Callable[[str], None]):
        self.app = app
        self.sink = sink
        self._buf = ""

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.app.call_from_thread(self.sink, line)
        return len(s)

    def flush(self):
        if self._buf:
            self.app.call_from_thread(self.sink, self._buf)
            self._buf = ""

    def isatty(self):
        return False


def _pm_text(mean: float, std: float) -> Text:
    color = "green" if mean >= 0.8 else "yellow" if mean >= 0.5 else "red"
    return Text(f"{mean:.3f}±{std:.3f}", style=color)


class EvalApp(App):
    CSS = """
    #body { height: 2.5fr; }
    #left { width: 1.5fr; }
    #right { width: 2fr; border-left: solid $accent; }
    .paneltitle { background: $boost; color: $text; text-style: bold; padding: 0 1; }
    #results { height: 1fr; }
    #summary { height: auto; min-height: 3; padding: 0 1; background: $panel; }
    #stdout { height: 1fr; background: $surface; }
    #controls { height: 3; padding: 0 1; align: left middle; }
    #controls Label { padding: 1 1 0 0; }
    .runsbox { width: 10; }
    #progress { width: 1fr; margin: 1 2; }
    #events { height: 8; border-top: solid $accent; padding: 0 1; }
    """

    BINDINGS = [
        ("r", "run", "Run"),
        ("c", "clear_log", "Clear stdout"),
        ("s", "save", "Save report"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, generate_fn, truth_dir, descr_dir, runs, out_dir, autosave=True):
        super().__init__()
        self.generate_fn = generate_fn
        self.truth_dir = truth_dir
        self.descr_dir = descr_dir
        self.runs = max(1, int(runs))
        self.out_dir = out_dir
        self.autosave = autosave
        self._busy = False
        self.last_results: list[list[dict]] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static("Per-task results  (mean±std across runs)", classes="paneltitle")
                yield DataTable(id="results", zebra_stripes=True, cursor_type="row")
                yield Static("No runs yet — press [b]r[/b] or Run.", id="summary")
            with Vertical(id="right"):
                yield Static("Generation stdout", classes="paneltitle")
                yield Log(id="stdout", highlight=False)
        with Horizontal(id="controls"):
            yield Label("Runs:")
            yield Input(value=str(self.runs), id="runs", type="integer", classes="runsbox")
            yield Button("Run", id="runbtn", variant="success")
            yield ProgressBar(id="progress", total=100, show_eta=False)
        yield RichLog(id="events", markup=True, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#results", DataTable)
        t.add_columns("Task", "n", "OK", "parse%",
                      "topo_f1 μ±σ", "role_f1 μ±σ", "stage%", "time μ")
        self.query_one("#progress", ProgressBar).update(total=100, progress=0)

    # ---- controls -------------------------------------------------------- #
    def action_run(self) -> None:
        self._start_run()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "runbtn":
            self._start_run()

    def action_clear_log(self) -> None:
        self.query_one("#stdout", Log).clear()

    def action_save(self) -> None:
        if not self.last_results:
            self.query_one("#events", RichLog).write("[yellow]nothing to save yet[/]")
            return
        try:
            jp, cp = save_multi_reports(self.last_results, self.out_dir)
            self.query_one("#events", RichLog).write(f"[green]saved[/] {jp}")
        except Exception as e:
            self.query_one("#events", RichLog).write(f"[red]save failed: {e}[/]")

    def _start_run(self) -> None:
        if self._busy:
            return
        try:
            runs = int(self.query_one("#runs", Input).value or "1")
        except ValueError:
            runs = 1
        self.runs = max(1, runs)
        self._busy = True
        self.query_one("#runbtn", Button).disabled = True
        self.query_one("#events", RichLog).write(f"[cyan]Starting {self.runs} run(s)…[/]")
        self.query_one("#results", DataTable).clear()
        self._evaluate(self.runs)

    # ---- background worker ---------------------------------------------- #
    @work(thread=True, exclusive=True)
    def _evaluate(self, runs: int) -> None:
        log = self.query_one("#stdout", Log)
        events = self.query_one("#events", RichLog)
        stream = UIStream(self, log.write_line)

        try:
            n_tasks = len([f for f in os.listdir(self.truth_dir) if f.endswith(".py")])
        except OSError as e:
            self.call_from_thread(events.write, f"[red]{e}[/]")
            self.call_from_thread(self._finish, None)
            return

        total = max(1, runs * n_tasks)
        self.call_from_thread(self.query_one("#progress", ProgressBar).update,
                              total=total, progress=0)

        done = 0
        all_results: list[list[dict]] = []

        def task_cb(ri, row):
            nonlocal done
            done += 1
            self.call_from_thread(self._on_task, ri, runs, row, done, total)

        def run_done_cb(ri, res):
            all_results.append(res)
            self.call_from_thread(self._refresh, list(all_results))

        try:
            with redirect_stdout(stream):
                run_repeated(self.generate_fn, self.truth_dir, self.descr_dir, runs,
                             task_cb=task_cb, run_done_cb=run_done_cb)
                stream.flush()
        except Exception as e:
            self.call_from_thread(events.write,
                                  f"[red]worker error: {type(e).__name__}: {e}[/]")

        self.last_results = all_results
        self.call_from_thread(self._finish, all_results)

    # ---- UI-thread updates ---------------------------------------------- #
    def _on_task(self, ri, runs, row, done, total) -> None:
        self.query_one("#progress", ProgressBar).update(total=total, progress=done)
        mark = {"OK": "[green]✓[/]", "SYNTAX_ERR": "[yellow]![/]",
                "EMPTY": "[yellow]·[/]", "GEN_ERROR": "[red]✗[/]"}.get(row["status"], "?")
        self.query_one("#events", RichLog).write(
            f"{mark} run {ri+1}/{runs} [b]{row['task']}[/] {row['status']} "
            f"topo={row['topology_f1']:.3f} ({row['seconds']:.1f}s)")

    def _refresh(self, all_results) -> None:
        t = self.query_one("#results", DataTable)
        t.clear()
        for s in per_task_stats(all_results):
            t.add_row(
                s["task"], str(s["n"]), str(s["ok"]),
                f"{s['parse_rate']*100:.0f}%",
                _pm_text(s["topo_mean"], s["topo_std"]),
                _pm_text(s["role_mean"], s["role_std"]),
                f"{s['stage_rate']*100:.0f}%",
                f"{s['time_mean']:.1f}s",
            )
        self._refresh_summary(all_results)

    def _refresh_summary(self, all_results) -> None:
        summ = summarize_runs(all_results)
        m = summ["metrics"]

        def part(label, key, pct=False, prec=3):
            d = m[key]
            body = (f"{d['mean']*100:.0f}%±{d['std']*100:.0f}%" if pct
                    else f"{d['mean']:.{prec}f}±{d['std']:.{prec}f}")
            return f"{label} [b]{body}[/]"

        text = ("  ·  ".join([
            f"[b]{summ['n_runs']} run(s)[/]",
            part("validity", "syntactic_validity_rate", pct=True),
            part("topo_f1", "mean_topology_f1"),
            part("role_f1", "mean_role_edge_f1"),
            part("stage-pass", "stage_count_pass_rate", pct=True),
        ]) + "\n" + "  ·  ".join([
            part("median topo", "median_topology_f1"),
            part("time/run", "total_seconds", prec=1) + "s",
        ]))
        self.query_one("#summary", Static).update(text)

    def _finish(self, all_results) -> None:
        self._busy = False
        self.query_one("#runbtn", Button).disabled = False
        if all_results:
            self.query_one("#events", RichLog).write("[green]done.[/]")
            if self.autosave:
                try:
                    jp, _ = save_multi_reports(all_results, self.out_dir)
                    self.query_one("#events", RichLog).write(f"[dim]saved {jp}[/]")
                except Exception as e:
                    self.query_one("#events", RichLog).write(f"[red]save failed: {e}[/]")


def _build_generate_fn(args):
    return make_demo_generate_fn() if args.demo else generate_workflow_wrap()


def main():
    ap = argparse.ArgumentParser(description="Structural evaluation of generated dispel4py workflows (Textual TUI).")
    ap.add_argument("--truth-dir", default="./truth")
    ap.add_argument("--descr-dir", default="./descr")
    ap.add_argument("--out", default="./reports", help="directory for JSON/CSV reports")
    ap.add_argument("--runs", type=int, default=3, help="number of repeated runs")
    ap.add_argument("--demo", action="store_true", help="offline generator, no login")
    ap.add_argument("--no-save", action="store_true", help="do not write report files")
    ap.add_argument("--headless", action="store_true", help="run without the TUI")
    args = ap.parse_args()

    truth_dir, descr_dir = args.truth_dir, args.descr_dir

    # In demo mode, synthesise fixtures if none are present so the UI is testable.
    if args.demo and (not os.path.isdir(truth_dir)
                      or not any(f.endswith(".py") for f in os.listdir(truth_dir))):
        tmp = tempfile.mkdtemp(prefix="d4py_demo_")
        truth_dir = os.path.join(tmp, "truth")
        descr_dir = os.path.join(tmp, "descr")
        make_demo_fixtures(truth_dir, descr_dir)

    # For the real backend, login happens here (before the TUI grabs the terminal).
    generate_fn = _build_generate_fn(args)

    if args.headless:
        run_headless(generate_fn, truth_dir, descr_dir, args.runs, args.out, not args.no_save)
        return

    EvalApp(generate_fn, truth_dir, descr_dir, args.runs, args.out,
            autosave=not args.no_save).run()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import re
import statistics
import tempfile
import time
from collections import Counter, OrderedDict
from contextlib import redirect_stdout
from datetime import datetime
from typing import Callable, Iterable

import networkx as nx
import numpy as np
from codebleu import calc_codebleu

BASE_KIND = {
    "ProducerPE": "producer",
    "IterativePE": "iterative",
    "ConsumerPE": "consumer",
    "GenericPE": "generic",
    "SimpleFunctionPE": "iterative",
}

NAN = float("nan")

# Set from CLI in main(). When False, port-name strings are ignored in every
# structural comparison (topology F1, role F1, GED, isomorphism).
STRICT_PORTS = False

try:

    _HAVE_GRAPH_LIBS = True
except Exception:
    _HAVE_GRAPH_LIBS = False

USE_GRAPH_SIM = _HAVE_GRAPH_LIBS
GRAPH_SIM_KEYS = ["ged", "ged_similarity", "graph_iso", "spectral_similarity"]
CODEBLEU_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
CODEBLEU_COMPONENTS = ("ngram_match_score", "weighted_ngram_match_score",
                       "syntax_match_score", "dataflow_match_score")


# --------------------------------------------------------------------------- #
# Graph extraction                                                            #
# --------------------------------------------------------------------------- #
def _parse(src: str):
    try:
        return ast.parse(src, mode="exec")
    except SyntaxError:
        return None


def _dotted(node):
    """Stable key for a Name or Attribute chain (`x`, `self.x`, `mod.C`), else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base is not None else None
    return None


def _called_name(call):
    """Name of the class/function being called (`C()` -> 'C', `m.C()` -> 'C'), else None."""
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _resolve_call_role(call, class_roles, factory_roles):
    """Role of the PE instance produced by a Call, or None if not a known PE."""
    cn = _called_name(call)
    if cn is None:
        return None
    if cn in class_roles:  # subclass of a PE base (possibly transitive)
        return class_roles[cn]
    if cn in BASE_KIND:  # direct use of a base class, e.g. SimpleFunctionPE(f)
        return BASE_KIND[cn]
    if cn in factory_roles:  # factory function that returns a PE
        return factory_roles[cn]
    return None


def _class_roles(tree):
    """
    Map class name -> role, resolving subclasses transitively so that a subclass
    of a subclass of a PE base (class B(A); class A(ProducerPE)) still resolves.
    """
    defs = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            bases = []
            for b in n.bases:
                bn = b.id if isinstance(b, ast.Name) else getattr(b, "attr", None)
                if bn:
                    bases.append(bn)
            defs[n.name] = bases
    roles, changed = {}, True
    while changed:
        changed = False
        for cls, bases in defs.items():
            if cls in roles:
                continue
            for b in bases:
                r = BASE_KIND.get(b) or roles.get(b)
                if r:
                    roles[cls] = r
                    changed = True
                    break
    return roles


def _factory_roles(tree, class_roles):
    """
    Map function name -> role for simple factories: a def whose body returns a
    PE instantiation (`def make_reader(): return ReadPE()`). First resolvable
    return wins.
    """
    out = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for ret in ast.walk(fn):
                if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Call):
                    r = _resolve_call_role(ret.value, class_roles, {})
                    if r:
                        out[fn.name] = r
                        break
    return out


def _var_roles(tree, class_roles, factory_roles):
    """
    Map variable key -> role for assignments to a Name or Attribute target,
    covering `x = C()`, `self.x = C()`, annotated `x: T = C()`, and chained
    `a = b = C()`.
    """
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            targets, value = n.targets, n.value
        elif isinstance(n, ast.AnnAssign) and isinstance(n.value, ast.Call):
            targets, value = [n.target], n.value
        else:
            continue
        role = _resolve_call_role(value, class_roles, factory_roles)
        if role is None:
            continue
        for t in targets:
            key = _dotted(t)
            if key:
                out[key] = role
    return out


def _connect_calls(tree):
    """Yield (src_expr, src_port, dst_expr, dst_port) for every *.connect(...) call."""
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "connect" and len(n.args) >= 4):
            a = n.args
            const = lambda x: x.value if isinstance(x, ast.Constant) else None
            yield a[0], const(a[1]), a[2], const(a[3])


def _endpoint(expr, var_roles, class_roles, factory_roles):
    """
    Resolve a connect endpoint expression to (node_key, role).
      * variable / attribute reference -> ("var", "self.x")    role from assignment
      * inline instantiation C()        -> ("inline", <unique>) role resolved directly
      * anything else (subscripts, …)   -> (None, None)  (edge dropped)
    Each inline call is a distinct node; the same variable used twice is one node.
    """
    if isinstance(expr, ast.Call):
        return ("inline", id(expr)), _resolve_call_role(expr, class_roles, factory_roles)
    key = _dotted(expr)
    if key is not None:
        return ("var", key), var_roles.get(key)
    return None, None


def _port_key(p) -> str:
    """Normalise a port name for comparison — dropped entirely unless --strict-ports."""
    if not STRICT_PORTS:
        return ""
    return "" if p is None else str(p)


def _canonical_ids(node_roles: list, edges: list) -> dict:
    """
    Assign each local node id an *order-invariant* canonical index via
    Weisfeiler-Lehman colour refinement. Two structurally identical workflows
    get the same canonical indices no matter what order their connect() calls
    were written in, so the topology-F1 edge sets line up correctly.
    """
    n = len(node_roles)
    if n == 0:
        return {}
    sig = [str(node_roles[v]) for v in range(n)]
    inc: list[list] = [[] for _ in range(n)]
    out: list[list] = [[] for _ in range(n)]
    for (s, sp, d, dp) in edges:
        if 0 <= s < n and 0 <= d < n:
            out[s].append((_port_key(sp), d))
            inc[d].append((_port_key(dp), s))

    for _ in range(n):  # n rounds is enough to stabilise on n nodes
        raw = []
        for v in range(n):
            im = tuple(sorted((p, sig[u]) for (p, u) in inc[v]))
            om = tuple(sorted((p, sig[w]) for (p, w) in out[v]))
            raw.append(repr((sig[v], im, om)))
        relabel = {s: i for i, s in enumerate(sorted(set(raw)))}
        sig = [str(relabel[s]) for s in raw]

    # Sort nodes by final colour; the local-id tiebreak is symmetric-safe for
    # interchangeable (automorphic) nodes, so isomorphic graphs still align.
    order = sorted(range(n), key=lambda v: (sig[v], v))
    return {local: idx for idx, local in enumerate(order)}


def extract_graph(src: str) -> dict:
    tree = _parse(src)
    if tree is None:
        return {"parse_ok": False, "role_edges": set(), "topo_edges": frozenset(),
                "edges": [], "n_stages": 0, "roles": Counter(), "node_roles": []}
    class_roles = _class_roles(tree)
    factory_roles = _factory_roles(tree, class_roles)
    var_roles = _var_roles(tree, class_roles, factory_roles)

    # Include every instantiated PE, not only instances that appear in a
    # connection. This makes extra or disconnected stages affect the metrics.
    order = [("var", key) for key in var_roles]
    key_role = {("var", key): role for key, role in var_roles.items()}
    raw = []
    for e_s, sp, e_d, dp in _connect_calls(tree):
        ks, rs = _endpoint(e_s, var_roles, class_roles, factory_roles)
        kd, rd = _endpoint(e_d, var_roles, class_roles, factory_roles)
        for k, r in ((ks, rs), (kd, rd)):
            if k is not None and k not in order:
                order.append(k)
                key_role[k] = r
        if ks is not None and kd is not None:
            raw.append((order.index(ks), sp, order.index(kd), dp))
    node_roles = [key_role.get(k) for k in order]

    canon = _canonical_ids(node_roles, raw)
    topo = frozenset((canon[s], _port_key(sp), canon[d], _port_key(dp))
                     for (s, sp, d, dp) in raw)
    role_edges = {(node_roles[s], node_roles[d], _port_key(sp), _port_key(dp))
                  for (s, sp, d, dp) in raw}

    return {"parse_ok": True, "roles": Counter(r for r in node_roles if r),
            "role_edges": role_edges, "topo_edges": topo,
            "edges": raw, "n_stages": len(order), "node_roles": node_roles}


def _f1(gen: Iterable, truth: Iterable) -> float:
    g, t = Counter(gen), Counter(truth)
    inter = sum((g & t).values())
    p = inter / max(sum(g.values()), 1)
    r = inter / max(sum(t.values()), 1)
    return 0.0 if p + r == 0 else round(2 * p * r / (p + r), 4)


# --------------------------------------------------------------------------- #
# Whole-graph similarity metrics (networkx + numpy)                           #
# --------------------------------------------------------------------------- #
def _to_digraph(graph: dict):
    """Labelled DiGraph: node label = role, edge label = (src, dst) ports (mode-aware)."""
    G = nx.DiGraph()
    for i, role in enumerate(graph.get("node_roles", [])):
        G.add_node(i, role=role if role is not None else "?")
    for (s, sp, d, dp) in graph.get("edges", []):
        G.add_edge(s, d, ports=(_port_key(sp), _port_key(dp)))
    return G


def _node_match(a, b):
    return a.get("role") == b.get("role")


def _edge_match(a, b):
    return a.get("ports") == b.get("ports")


def _graph_edit_distance(G1, G2) -> float:
    try:
        d = nx.graph_edit_distance(G1, G2, node_match=_node_match,
                                   edge_match=_edge_match, timeout=10)
    except Exception:
        d = None
    if d is None:
        try:
            d = next(nx.optimize_graph_edit_distance(
                G1, G2, node_match=_node_match, edge_match=_edge_match))
        except Exception:
            d = float(G1.number_of_nodes() + G1.number_of_edges()
                      + G2.number_of_nodes() + G2.number_of_edges())
    return float(d)


def _spectral_similarity(G1, G2) -> float:
    s1, s2 = nx.laplacian_spectrum(G1), nx.laplacian_spectrum(G2)
    L = max(len(s1), len(s2))
    if L == 0:
        return 1.0
    s1 = np.pad(s1, (0, L - len(s1)))
    s2 = np.pad(s2, (0, L - len(s2)))
    dist = float(np.linalg.norm(s1 - s2))
    denom = float(np.linalg.norm(s1) + np.linalg.norm(s2))
    return 1.0 if denom == 0 else max(0.0, 1.0 - dist / denom)


def graph_similarity_scores(answer_graph: dict, truth_graph: dict) -> dict:
    if not USE_GRAPH_SIM:
        return {"ged": NAN, "ged_similarity": NAN,
                "graph_iso": NAN, "spectral_similarity": NAN}
    G1 = _to_digraph(answer_graph)
    G2 = _to_digraph(truth_graph)
    ged = _graph_edit_distance(G1, G2)
    bound = (G1.number_of_nodes() + G1.number_of_edges()
             + G2.number_of_nodes() + G2.number_of_edges())
    ged_sim = 1.0 if bound == 0 else max(0.0, 1.0 - ged / bound)
    try:
        iso = 1.0 if nx.is_isomorphic(G1, G2, node_match=_node_match,
                                      edge_match=_edge_match) else 0.0
    except Exception:
        iso = 0.0
    spec = _spectral_similarity(G1, G2)
    return {"ged": round(ged, 4), "ged_similarity": round(ged_sim, 4),
            "graph_iso": iso, "spectral_similarity": round(spec, 4)}


def canonical_workflow_code(source: str) -> str | None:
    """Represent only instantiated PE roles and their directed connections."""
    graph = extract_graph(source)
    if not graph["parse_ok"]:
        return None

    roles = graph["node_roles"]
    edges = graph["edges"]
    canonical_ids = _canonical_ids(roles, edges)
    nodes = sorted(((canonical_ids[i], role or "UnknownPE")
                    for i, role in enumerate(roles)))
    canonical_edges = sorted((canonical_ids[src], _port_key(src_port),
                              canonical_ids[dst], _port_key(dst_port))
                             for src, src_port, dst, dst_port in edges)

    lines = [f"node_{node_id} = {role}()" for node_id, role in nodes]
    lines.append("graph = WorkflowGraph()")
    for src, src_port, dst, dst_port in canonical_edges:
        # Stable placeholder ports ensure ignored port names cannot influence
        # CodeBLEU while retaining the shape of a real connect call.
        if not STRICT_PORTS:
            src_port, dst_port = "output", "input"
        lines.append(
            f"graph.connect(node_{src}, {src_port!r}, node_{dst}, {dst_port!r})"
        )
    return "\n".join(lines) + "\n"


def codebleu_score(answer: str, ground_truth: str) -> float:
    """Run CodeBLEU on canonical workflow structure, excluding incidental code."""
    answer_workflow = canonical_workflow_code(answer)
    truth_workflow = canonical_workflow_code(ground_truth)
    if answer_workflow is None or truth_workflow is None:
        return NAN
    if not answer_workflow.strip() or not truth_workflow.strip():
        return float(answer_workflow == truth_workflow)
    try:
        scores = calc_codebleu(
            references=[truth_workflow],
            predictions=[answer_workflow],
            lang="python",
            weights=CODEBLEU_WEIGHTS,
            tokenizer=None,
        )
    except Exception:
        return NAN
    # Calculate this explicitly because some wrappers replace a zero data-flow
    # component with one, which inflates the published CodeBLEU formula.
    return sum(weight * scores[key]
               for weight, key in zip(CODEBLEU_WEIGHTS, CODEBLEU_COMPONENTS))


def structural_scores(answer: str, ground_truth: str) -> dict:
    g = extract_graph(answer)
    t = extract_graph(ground_truth)
    if not g["parse_ok"]:
        return {"parse_ok": 0.0, "topology_f1": 0.0, "role_edge_f1": 0.0,
                "stage_count_ok": 0.0, "ged": NAN, "ged_similarity": 0.0,
                "graph_iso": 0.0, "spectral_similarity": 0.0}
    scores = {"parse_ok": 1.0,
              "topology_f1": _f1(g["topo_edges"], t["topo_edges"]),
              "role_edge_f1": _f1(g["role_edges"], t["role_edges"]),
              "stage_count_ok": float(g["n_stages"] == t["n_stages"])}
    scores.update(graph_similarity_scores(g, t))
    return scores


def structural_scores_multi(answer: str, ground_truths: list[str]) -> dict:
    best = None
    for gt in ground_truths:
        s = structural_scores(answer, gt)
        if best is None or ((s["topology_f1"], s["role_edge_f1"])
                            > (best["topology_f1"], best["role_edge_f1"])):
            best = s
    if best is None:
        best = structural_scores(answer, "")
    return best


# --------------------------------------------------------------------------- #
# Single-run evaluation driver                                                #
# --------------------------------------------------------------------------- #
STATUS_OK, STATUS_SYNTAX, STATUS_EMPTY, STATUS_GEN = "OK", "SYNTAX_ERR", "EMPTY", "GEN_ERROR"


def _blank_row(task: str) -> dict:
    return {"task": task, "status": None, "error": "", "seconds": 0.0,
            "parse_ok": 0.0, "topology_f1": 0.0, "role_edge_f1": 0.0,
            "stage_count_ok": 0.0, "ged": NAN, "ged_similarity": 0.0,
            "graph_iso": 0.0, "spectral_similarity": 0.0, "codebleu": NAN}


def run_evaluation(generate_fn: Callable[[str], str], truth_dir: str, descr_dir: str,
                   progress_cb: Callable[[dict], None] | None = None) -> list[dict]:
    results = []
    truth_files = sorted(f for f in os.listdir(truth_dir) if f.endswith(".py"))
    for py in truth_files:
        task = py[:-3]
        row = _blank_row(task)
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
                row["codebleu"] = codebleu_score(str(code), truth_code)
                row["status"] = STATUS_OK if scores["parse_ok"] else STATUS_SYNTAX
        except Exception as e:
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
        vals = [r[key] for r in rows
                if not (isinstance(r[key], float) and math.isnan(r[key]))]
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
        "mean_ged": mean("ged", parsed),
        "mean_ged_similarity": mean("ged_similarity", parsed),
        "graph_iso_rate": mean("graph_iso", parsed),
        "mean_spectral_similarity": mean("spectral_similarity", parsed),
        "mean_codebleu": mean("codebleu", results),
        "total_seconds": sum(r["seconds"] for r in results),
        "n_parsed": len(parsed),
    }


# --------------------------------------------------------------------------- #
# Multi-run helpers                                                            #
# --------------------------------------------------------------------------- #
def _clean(vals):
    return [v for v in vals
            if v is not None and not (isinstance(v, float) and math.isnan(v))]


def mean_or0(vals) -> float:
    vals = _clean(vals)
    return statistics.fmean(vals) if vals else 0.0


def std_or0(vals) -> float:
    vals = _clean(vals)
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def _fmt_ged(v) -> str:
    """Format a raw GED distance for one-line logs; 'n/a' when unavailable (NaN)."""
    return "n/a" if (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def run_repeated(generate_fn, truth_dir, descr_dir, runs: int,
                 task_cb: Callable[[int, dict], None] | None = None,
                 run_done_cb: Callable[[int, list[dict]], None] | None = None) -> list[list[dict]]:
    all_results: list[list[dict]] = []
    for ri in range(runs):
        cb = (lambda row, ri=ri: task_cb(ri, row)) if task_cb else None
        res = run_evaluation(generate_fn, truth_dir, descr_dir, progress_cb=cb)
        all_results.append(res)
        if run_done_cb:
            run_done_cb(ri, res)
    return all_results


def per_task_stats(all_results: list[list[dict]]) -> list[dict]:
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
        gsim = [r["ged_similarity"] for r in parsed]
        ged = [r["ged"] for r in parsed]
        spec = [r["spectral_similarity"] for r in parsed]
        iso = [r["graph_iso"] for r in parsed]
        codebleu = [r["codebleu"] for r in rows]
        secs = [r["seconds"] for r in rows]
        out.append({
            "task": task, "n": len(rows),
            "ok": sum(1 for r in rows if r["status"] == STATUS_OK),
            "parse_rate": mean_or0([r["parse_ok"] for r in rows]),
            "topo_mean": mean_or0(topo), "topo_std": std_or0(topo),
            "role_mean": mean_or0(role), "role_std": std_or0(role),
            "stage_rate": mean_or0(stage),
            "ged_mean": mean_or0(ged), "ged_std": std_or0(ged),
            "gsim_mean": mean_or0(gsim), "gsim_std": std_or0(gsim),
            "spec_mean": mean_or0(spec), "spec_std": std_or0(spec),
            "iso_rate": mean_or0(iso),
            "codebleu_mean": mean_or0(codebleu), "codebleu_std": std_or0(codebleu),
            "time_mean": mean_or0(secs), "time_std": std_or0(secs),
        })
    return out


_SUMMARY_KEYS = ["syntactic_validity_rate", "mean_topology_f1", "median_topology_f1",
                 "mean_role_edge_f1", "stage_count_pass_rate",
                  "mean_ged", "mean_ged_similarity", "graph_iso_rate",
                  "mean_spectral_similarity", "mean_codebleu", "total_seconds", "n_parsed"]


def summarize_runs(all_results: list[list[dict]]) -> dict:
    per_run = [aggregate(r) for r in all_results]
    metrics = {}
    for k in _SUMMARY_KEYS:
        vals = [a[k] for a in per_run]
        metrics[k] = {"mean": mean_or0(vals), "std": std_or0(vals), "values": vals}
    return {"n_runs": len(all_results), "metrics": metrics,
            "per_run_counts": [a["counts"] for a in per_run]}


# --------------------------------------------------------------------------- #
# Report saving                                                                #
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
                   "strict_ports": STRICT_PORTS, "graph_sim_enabled": USE_GRAPH_SIM,
                   "summary": summary, "per_task": task_stats,
                   "runs": all_results}, f, indent=2, default=str)

    fields = ["task", "n", "ok", "parse_rate", "topo_mean", "topo_std",
              "role_mean", "role_std", "stage_rate",
               "ged_mean", "ged_std", "gsim_mean", "gsim_std",
               "spec_mean", "spec_std", "iso_rate", "codebleu_mean", "codebleu_std",
               "time_mean", "time_std"]
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
    rng = random.Random(seed)

    def _gen(description: str) -> str:
        print(f"[gen] prompt received ({len(description)} chars)")
        for step in ("tokenizing", "planning stages", "wiring edges", "serializing"):
            print(f"[gen] {step} …")
            time.sleep(0.04)

        n = rng.choice([2, 3, 3, 3, 4])
        roles = ["ProducerPE"] + ["IterativePE"] * (n - 2) + ["ConsumerPE"]
        if rng.random() < 0.20:
            roles[rng.randrange(n)] = "GenericPE"

        # Randomise port names + connect-call order so the demo exercises the
        # order-invariant / port-insensitive matching.
        oport, iport = rng.choice([("output", "input"), ("out", "in"), ("result", "data")])
        drop_edge = rng.random() < 0.20
        lines, names = [], []
        for i, r in enumerate(roles):
            cls = f"Stage{i}"
            lines.append(f"class {cls}({r}):\n    pass\n")
            names.append(f"s{i}")
            lines[-1] += f"{names[-1]} = {cls}()"
        lines.insert(len(roles), "graph = WorkflowGraph()")
        skip = rng.randrange(max(1, len(names) - 1)) if drop_edge else -1
        conns = [f"graph.connect({names[i]}, '{oport}', {names[i + 1]}, '{iport}')"
                 for i in range(len(names) - 1) if i != skip]
        if rng.random() < 0.5:
            conns.reverse()
        lines += conns
        print("[gen] complete")
        return "\n".join(lines) + "\n"

    return _gen


def make_demo_fixtures(truth_dir: str, descr_dir: str) -> None:
    os.makedirs(truth_dir, exist_ok=True)
    os.makedirs(descr_dir, exist_ok=True)
    specs = {
        "wordcount": ["ProducerPE", "IterativePE", "ConsumerPE"],
        "etl_pipe": ["ProducerPE", "IterativePE", "IterativePE", "ConsumerPE"],
        "sensor_avg": ["ProducerPE", "IterativePE", "ConsumerPE"],
        "two_stage": ["ProducerPE", "ConsumerPE"],
        "wide_map": ["ProducerPE", "IterativePE", "IterativePE", "IterativePE", "ConsumerPE"],
    }
    for name, roles in specs.items():
        with open(os.path.join(truth_dir, name + ".py"), "w") as f:
            f.write(_linear_workflow_code(roles))
        with open(os.path.join(descr_dir, name + ".txt"), "w") as f:
            f.write(f"Build a dispel4py workflow '{name}' with {len(roles)} stages: "
                    + " -> ".join(roles) + ".")


# --------------------------------------------------------------------------- #
# Headless runner                                                              #
# --------------------------------------------------------------------------- #
def print_multi_summary(all_results: list[list[dict]]) -> None:
    stats = per_task_stats(all_results)
    summ = summarize_runs(all_results)
    cols = [("Task", 22), ("n", 4), ("OK", 4), ("parse%", 7),
            ("topo_f1 μ±σ", 15), ("role_f1 μ±σ", 15),
            ("CodeBLEU μ±σ", 15), ("ged μ±σ", 15), ("spec μ±σ", 15), ("iso%", 6),
            ("stage%", 7), ("time μ", 8)]
    header = "  ".join(name.ljust(w) for name, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for s in stats:
        print("  ".join([
            s["task"][:22].ljust(22), str(s["n"]).ljust(4), str(s["ok"]).ljust(4),
            f"{s['parse_rate'] * 100:.0f}%".ljust(7),
            f"{s['topo_mean']:.3f}±{s['topo_std']:.3f}".ljust(15),
            f"{s['role_mean']:.3f}±{s['role_std']:.3f}".ljust(15),
            f"{s['codebleu_mean']:.3f}±{s['codebleu_std']:.3f}".ljust(15),
            f"{s['ged_mean']:.2f}±{s['ged_std']:.2f}".ljust(15),
            f"{s['spec_mean']:.3f}±{s['spec_std']:.3f}".ljust(15),
            f"{s['iso_rate'] * 100:.0f}%".ljust(6),
            f"{s['stage_rate'] * 100:.0f}%".ljust(7),
            f"{s['time_mean']:.1f}s".ljust(8),
        ]))
    m = summ["metrics"]

    def pm(key, pct=False, prec=3):
        d = m[key]
        return (f"{d['mean'] * 100:.0f}%±{d['std'] * 100:.0f}%" if pct
                else f"{d['mean']:.{prec}f}±{d['std']:.{prec}f}")

    print("\n" + "=" * len(header))
    print(f"Runs: {summ['n_runs']}   ports: {'strict' if STRICT_PORTS else 'ignored'}   "
          f"validity {pm('syntactic_validity_rate', pct=True)}   "
          f"stage-pass {pm('stage_count_pass_rate', pct=True)}")
    print(f"topology_f1 {pm('mean_topology_f1')}   role_edge_f1 {pm('mean_role_edge_f1')}   "
          f"CodeBLEU {pm('mean_codebleu')}   time/run {pm('total_seconds', prec=1)}s")
    if USE_GRAPH_SIM:
        print(f"ged {pm('mean_ged', prec=2)}   ged_sim {pm('mean_ged_similarity')}   "
              f"iso {pm('graph_iso_rate', pct=True)}   spectral {pm('mean_spectral_similarity')}")
    else:
        print("graph-similarity metrics disabled (install networkx + numpy)")


def run_headless(generate_fn, truth_dir, descr_dir, runs, out_dir, save):
    def task_cb(ri, row):
        print(f"  run {ri + 1}/{runs} [{row['status']:<10}] {row['task']:<22} "
              f"topo={row['topology_f1']:.3f} CodeBLEU={row['codebleu']:.3f} "
              f"ged={_fmt_ged(row['ged'])} "
              f"({row['seconds']:.1f}s)")

    def run_done_cb(ri, res):
        print(f"# finished run {ri + 1}/{runs}")

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

_ANSI_RE = re.compile(
    r"\x1B\[[0-?]*[ -/]*[@-~]"
    r"|\x1B\][^\x07\x1B]*(?:\x07|\x1B\\)"
    r"|\x1B[\x40-\x5F]"
)
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _clean_line(line: str) -> str:
    line = _ANSI_RE.sub("", line)
    if "\r" in line:
        line = line.rsplit("\r", 1)[-1]
    return _CTRL_RE.sub("", line)


class UIStream:
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
            self.app.call_from_thread(self.sink, _clean_line(line))
        return len(s)

    def flush(self):
        if self._buf:
            self.app.call_from_thread(self.sink, _clean_line(self._buf))
            self._buf = ""

    def isatty(self):
        return False


def _pm_text(mean: float, std: float) -> Text:
    color = "green" if mean >= 0.8 else "yellow" if mean >= 0.5 else "red"
    return Text(f"{mean:.3f}±{std:.3f}", style=color)


def _dist_text(mean: float, std: float) -> Text:
    """Raw GED distance (lower = closer). Not a 0..1 score, so no similarity colouring."""
    if isinstance(mean, float) and math.isnan(mean):
        return Text("n/a", style="dim")
    return Text(f"{mean:.2f}±{std:.2f}", style="cyan")


def _rate_text(rate: float) -> Text:
    color = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"
    return Text(f"{rate * 100:.0f}%", style=color)


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
        t.add_columns("Task", "n", "OK", "parse%", "topo_f1 μ±σ", "role_f1 μ±σ",
                      "CodeBLEU μ±σ", "ged μ±σ", "spec μ±σ", "iso%", "stage%", "time μ")
        self.query_one("#progress", ProgressBar).update(total=100, progress=0)
        note = "ports: strict" if STRICT_PORTS else "ports: ignored"
        if not USE_GRAPH_SIM:
            note += "  ·  graph-sim disabled (networkx/numpy missing)"
        self.query_one("#events", RichLog).write(f"[dim]{note}[/]")

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
        current_rows: list[dict] = []  # rows of the run currently in progress

        def task_cb(ri, row):
            nonlocal done
            done += 1
            current_rows.append(row)
            self.call_from_thread(self._on_task, ri, runs, row, done, total)
            # live-update the per-task table: finished runs + the in-progress one
            self.call_from_thread(self._refresh, all_results + [list(current_rows)])

        def run_done_cb(ri, res):
            all_results.append(res)
            current_rows.clear()
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

    def _on_task(self, ri, runs, row, done, total) -> None:
        self.query_one("#progress", ProgressBar).update(total=total, progress=done)
        mark = {"OK": "[green]✓[/]", "SYNTAX_ERR": "[yellow]![/]",
                "EMPTY": "[yellow]·[/]", "GEN_ERROR": "[red]✗[/]"}.get(row["status"], "?")
        self.query_one("#events", RichLog).write(
            f"{mark} run {ri + 1}/{runs} [b]{row['task']}[/] {row['status']} "
            f"topo={row['topology_f1']:.3f} CodeBLEU={row['codebleu']:.3f} "
            f"ged={_fmt_ged(row['ged'])} "
            f"({row['seconds']:.1f}s)")

    def _refresh(self, all_results) -> None:
        t = self.query_one("#results", DataTable)
        t.clear()
        for s in per_task_stats(all_results):
            t.add_row(
                s["task"], str(s["n"]), str(s["ok"]),
                f"{s['parse_rate'] * 100:.0f}%",
                _pm_text(s["topo_mean"], s["topo_std"]),
                _pm_text(s["role_mean"], s["role_std"]),
                _pm_text(s["codebleu_mean"], s["codebleu_std"]),
                _dist_text(s["ged_mean"], s["ged_std"]),
                _pm_text(s["spec_mean"], s["spec_std"]),
                _rate_text(s["iso_rate"]),
                f"{s['stage_rate'] * 100:.0f}%",
                f"{s['time_mean']:.1f}s",
            )
        self._refresh_summary(all_results)

    def _refresh_summary(self, all_results) -> None:
        summ = summarize_runs(all_results)
        m = summ["metrics"]

        def part(label, key, pct=False, prec=3):
            d = m[key]
            body = (f"{d['mean'] * 100:.0f}%±{d['std'] * 100:.0f}%" if pct
                    else f"{d['mean']:.{prec}f}±{d['std']:.{prec}f}")
            return f"{label} [b]{body}[/]"

        line1 = "  ·  ".join([
            f"[b]{summ['n_runs']} run(s)[/]",
            part("validity", "syntactic_validity_rate", pct=True),
            part("topo_f1", "mean_topology_f1"),
            part("role_f1", "mean_role_edge_f1"),
            part("CodeBLEU", "mean_codebleu"),
            part("stage-pass", "stage_count_pass_rate", pct=True),
        ])
        if USE_GRAPH_SIM:
            line2 = "  ·  ".join([
                part("ged", "mean_ged", prec=2),
                part("ged_sim", "mean_ged_similarity"),
                part("iso", "graph_iso_rate", pct=True),
                part("spectral", "mean_spectral_similarity"),
            ])
        else:
            line2 = "[dim]graph-similarity disabled[/]"
        self.query_one("#summary", Static).update(line1 + "\n" + line2)

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
    ap.add_argument("--strict-ports", action="store_true",
                    help="require exact port-name matches (default: port names ignored)")
    ap.add_argument("--no-graph-sim", action="store_true",
                    help="skip whole-graph similarity metrics (GED / isomorphism / spectral)")
    args = ap.parse_args()

    global STRICT_PORTS, USE_GRAPH_SIM
    STRICT_PORTS = args.strict_ports
    if args.no_graph_sim:
        USE_GRAPH_SIM = False
    elif not _HAVE_GRAPH_LIBS:
        USE_GRAPH_SIM = False
        print("[warn] networkx/numpy not found — graph-similarity metrics disabled. "
              "Install:  pip install networkx numpy")

    truth_dir, descr_dir = args.truth_dir, args.descr_dir

    if args.demo and (not os.path.isdir(truth_dir)
                      or not any(f.endswith(".py") for f in os.listdir(truth_dir))):
        tmp = tempfile.mkdtemp(prefix="d4py_demo_")
        truth_dir = os.path.join(tmp, "truth")
        descr_dir = os.path.join(tmp, "descr")
        make_demo_fixtures(truth_dir, descr_dir)

    generate_fn = _build_generate_fn(args)

    if args.headless:
        run_headless(generate_fn, truth_dir, descr_dir, args.runs, args.out, not args.no_save)
        return

    EvalApp(generate_fn, truth_dir, descr_dir, args.runs, args.out,
            autosave=not args.no_save).run()


if __name__ == "__main__":
    main()

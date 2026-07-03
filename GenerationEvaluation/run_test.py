import os
import ast
from collections import Counter
from typing import Iterable

from continuous_eval.metrics.code.python import PythonASTSimilarity
from laminar.client.d4pyclient import d4pClient
from laminar.clitools.advanced_search import AdvancedSearchCommand
from pwinput import pwinput

metric = PythonASTSimilarity()
client = d4pClient()
username = os.environ.get("LAMINAR_USERNAME") or input("Username: ")
password = os.environ.get("LAMINAR_PASSWORD") or pwinput("Password: ")
client.login(username, password)

generate_command = AdvancedSearchCommand(client=client)



BASE_KIND = {
    "ProducerPE": "producer",
    "IterativePE": "iterative",
    "ConsumerPE": "consumer",
    "GenericPE": "generic",
    "SimpleFunctionPE": "iterative",
}


def _parse(src: str) -> ast.AST | None:
    try:
        return ast.parse(src, mode="exec")
    except SyntaxError:
        return None


def _class_roles(tree: ast.AST) -> dict[str, str]:
    roles = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            for b in n.bases:
                bn = b.id if isinstance(b, ast.Name) else getattr(b, "attr", None)
                if bn in BASE_KIND:
                    roles[n.name] = BASE_KIND[bn]
    return roles


def _var_to_class(tree: ast.AST, class_roles: dict[str, str]) -> dict[str, str]:
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            f = n.value.func
            cname = f.id if isinstance(f, ast.Name) else None
            if cname in class_roles and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                out[n.targets[0].id] = cname
    return out


def _connect_calls(tree: ast.AST):
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "connect" and len(n.args) >= 4):
            a = n.args
            name = lambda x: x.id if isinstance(x, ast.Name) else None
            const = lambda x: x.value if isinstance(x, ast.Constant) else None
            yield name(a[0]), const(a[1]), name(a[2]), const(a[3])


def extract_graph(src: str) -> dict:
    """Return a normalized description of the workflow graph from source code."""
    tree = _parse(src)
    if tree is None:
        return {"parse_ok": False, "role_edges": set(), "topo_edges": frozenset(),
                "n_stages": 0, "roles": Counter()}
    class_roles = _class_roles(tree)
    var_class = _var_to_class(tree, class_roles)

    role_edges = set()  # role-typed, instance-name invariant
    order, topo = [], []  # position-based topology, fully name/role invariant
    for s, sp, d, dp in _connect_calls(tree):
        role_edges.add((class_roles.get(var_class.get(s)),
                        class_roles.get(var_class.get(d)), sp, dp))
        for v in (s, d):
            if v and v not in order:
                order.append(v)
        if s in order and d in order:
            topo.append((order.index(s), sp, order.index(d), dp))

    return {
        "parse_ok": True,
        "roles": Counter(class_roles.values()),
        "role_edges": role_edges,
        "topo_edges": frozenset(topo),
        "n_stages": len(order),
    }


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
        return {"parse_ok": 0.0, "topology_f1": 0.0, "role_edge_f1": 0.0,
                "stage_count_ok": 0.0}
    return {
        "parse_ok": 1.0,
        "topology_f1": _f1(g["topo_edges"], t["topo_edges"]),
        "role_edge_f1": _f1(g["role_edges"], t["role_edges"]),
        "stage_count_ok": float(g["n_stages"] == t["n_stages"]),
    }


def structural_scores_multi(answer: str, ground_truths: list[str]) -> dict:
    """Take the best structural agreement over several acceptable references."""
    best = {"parse_ok": 0.0, "topology_f1": 0.0, "role_edge_f1": 0.0, "stage_count_ok": 0.0}
    for gt in ground_truths:
        s = structural_scores(answer, gt)
        if (s["topology_f1"], s["role_edge_f1"]) > (best["topology_f1"], best["role_edge_f1"]):
            best = s
    return best


for file in os.listdir("./truth"):
    with open(f"truth/{file}", "r") as f:
        input_code = f.read()

    file = file.replace(".py", ".txt")
    with open(f"descr/{file}", "r") as f:
        input_description = f.read()

    try:
        proposed_code = generate_command._generate(query=input_description, silent=True)["workflow_code"]
        print(structural_scores_multi(proposed_code, [input_code]))
    except Exception as e:
        print(e)
        pass

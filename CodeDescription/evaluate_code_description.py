#!/usr/bin/env python3

import argparse
import ast
import asyncio
import csv
import json
import os
import random
import sys

from openai import AsyncOpenAI, APIConnectionError
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import SemanticSimilarity, Faithfulness

CORE_FIELDS = ["role", "processing", "parameters", "notes"]
PLUMBING_FIELDS = ["component_name", "component_type", "inputs", "outputs"]
ALL_DESC_FIELDS = ["component_name", "component_type", "role",
                   "inputs", "outputs", "processing", "parameters", "notes"]

CONVERT_SECOND_ARG = False

# dispel4py base classes -> (implicit input ports, implicit output ports).
# These ports are declared in the BASE class, not the emitted subclass, so they must
# be inferred from the base rather than parsed from the generated code. Adjust if
# your dispel4py version names them differently.
KNOWN_PE_PORTS = {
    "iterativepe":      ({"input"}, {"output"}),
    "simplefunctionpe": ({"input"}, {"output"}),
    "functionpe":       ({"input"}, {"output"}),
    "producerpe":       (set(),     {"output"}),
    "consumerpe":       ({"input"}, set()),
}

# Short factual contract per base class, appended to the plumbing-faithfulness context
# so claims that follow from the base class (not visible in the subclass source) are
# groundable rather than flagged as hallucinations.
BASE_CONTRACTS = {
    "iterativepe": ("IterativePE is a dispel4py processing element with exactly one "
                    "input port named 'input' and one output port named 'output'. Its "
                    "_process method receives one data item from the input port and its "
                    "return value is emitted on the output port."),
    "simplefunctionpe": ("SimpleFunctionPE is a dispel4py processing element with one "
                         "input port 'input' and one output port 'output'."),
    "producerpe": ("ProducerPE is a dispel4py processing element with no input port and "
                   "one output port 'output'."),
    "consumerpe": ("ConsumerPE is a dispel4py processing element with one input port "
                   "'input' and no output port."),
}


# ---------------------------------------------------------------------------
# Description coercion / flattening
# ---------------------------------------------------------------------------
def coerce_description(desc):
    if desc is None:
        return ""
    if isinstance(desc, (str, dict)):
        return _unwrap_desc(desc) if isinstance(desc, dict) else desc
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(desc, attr, None)
        if callable(fn):
            try:
                return _unwrap_desc(fn())
            except Exception:  # noqa: BLE001
                pass
    try:
        import dataclasses
        if dataclasses.is_dataclass(desc):
            return _unwrap_desc(dataclasses.asdict(desc))
    except Exception:  # noqa: BLE001
        pass
    d = getattr(desc, "__dict__", None)
    if isinstance(d, dict) and d:
        return _unwrap_desc(d)
    return str(desc)


def _unwrap_desc(d):
    if not isinstance(d, dict):
        return d
    if any(k in d for k in ALL_DESC_FIELDS):
        return d
    inner = [v for v in d.values() if isinstance(v, dict)]
    if len(inner) == 1 and any(k in inner[0] for k in ALL_DESC_FIELDS):
        return inner[0]
    return d


def flatten_desc(desc, fields=None) -> str:
    if isinstance(desc, str):
        return desc.strip()
    if isinstance(desc, dict):
        keys = fields if fields is not None else ALL_DESC_FIELDS
        parts = []
        for key in keys:
            val = desc.get(key)
            if val:
                parts.append(f"{key.replace('_', ' ').capitalize()}: {val}")
        if parts:
            return "\n".join(parts)
        return json.dumps(desc, ensure_ascii=False) if fields is None else ""
    return str(desc)


# ---------------------------------------------------------------------------
# PE structure: class name, base class, and IMPLICIT ports from the base class
# ---------------------------------------------------------------------------
def _unparse(node):
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""


def _call_name(func):
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _first_str_arg(call):
    for a in call.args:
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            return a.value
    return None


def extract_pe_structure(converted_code: str) -> dict:
    """Return class_name, base_classes, and the PE's input/output port sets.

    Ports come from two sources, in order of preference:
      1. explicitly declared in the subclass (GenericPE-style: _add_input('x'),
         self.inputconnections['x'] = ...), if the converter ever emits that;
      2. otherwise inferred from the base class contract (IterativePE etc.), since
         this converter's ports live in the base class and never appear in the
         subclass source.
    Degrades to empty sets if nothing is determinable."""
    info = {"class_name": None, "base_classes": [], "inputs": set(), "outputs": set()}
    try:
        tree = ast.parse(converted_code)
    except SyntaxError:
        return info

    declared_in, declared_out = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and info["class_name"] is None:
            info["class_name"] = node.name
            info["base_classes"] = [_unparse(b) for b in node.bases]
        if isinstance(node, ast.Call):
            fname = _call_name(node.func).lower()
            if fname in ("_add_input", "addinput", "_addinput", "add_input"):
                name = _first_str_arg(node)
                if name:
                    declared_in.add(name)
            elif fname in ("_add_output", "addoutput", "_addoutput", "add_output"):
                name = _first_str_arg(node)
                if name:
                    declared_out.add(name)
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Subscript):
                    container = _unparse(tgt.value).lower()
                    sl = tgt.slice
                    key = sl.value if isinstance(sl, ast.Constant) and \
                        isinstance(sl.value, str) else None
                    if key and "input" in container:
                        declared_in.add(key)
                    elif key and "output" in container:
                        declared_out.add(key)

    if declared_in or declared_out:
        info["inputs"], info["outputs"] = declared_in, declared_out
    else:
        for b in info["base_classes"]:
            key = b.split(".")[-1].lower()
            if key in KNOWN_PE_PORTS:
                ins, outs = KNOWN_PE_PORTS[key]
                info["inputs"] |= ins
                info["outputs"] |= outs
    return info


def pe_context(converted_code: str, struct: dict) -> str:
    """Converted code plus the base-class contract, so the faithfulness judge can
    ground port/type claims that follow from the base class but aren't in the
    subclass source."""
    notes = []
    for b in struct["base_classes"]:
        key = b.split(".")[-1].lower()
        if key in BASE_CONTRACTS:
            notes.append(BASE_CONTRACTS[key])
    if not notes:
        return converted_code
    return converted_code + "\n\n# dispel4py framework contract:\n# " + \
        "\n# ".join(notes)


# ---------------------------------------------------------------------------
# Deterministic plumbing checks (exact, free)
# ---------------------------------------------------------------------------
def _port_names_from_value(value):
    """Extract lowercased port names if the description field is a list/tuple of
    strings or dicts; else None."""
    if not isinstance(value, (list, tuple)):
        return None
    names = set()
    for el in value:
        if isinstance(el, str):
            names.add(el.lower())
        elif isinstance(el, dict):
            for k in ("name", "port", "id", "label"):
                if el.get(k):
                    names.add(str(el[k]).lower())
                    break
    return names


def _port_checks(kind, value, expected):
    """kind: 'input'|'output'; expected: port-name set implied by the PE.
    Emits arity/name checks when the field is structured (a list), a presence check
    when it's a string. Skips entirely when the PE has no ports of this kind (can't
    reliably verify absence from prose)."""
    if not expected:
        return {}
    out = {}
    if isinstance(value, (list, tuple)):
        out[f"{kind}_arity_ok"] = (len(value) == len(expected))
        names = _port_names_from_value(value)
        if names:
            out[f"{kind}_names_ok"] = all(any(p in n for n in names) for p in expected)
    elif isinstance(value, str):
        out[f"{kind}_present"] = bool(value.strip())
    return out


def deterministic_plumbing_checks(desc, struct: dict) -> dict:
    """Exact checks against ground truth derived from the converted PE / its base."""
    if not isinstance(desc, dict):
        return {}
    checks = {}

    if struct["class_name"]:
        checks["name_ok"] = struct["class_name"].lower() in \
            str(desc.get("component_name", "")).lower()

    said_type = str(desc.get("component_type", "")).lower()
    base_tokens = set()
    for b in struct["base_classes"]:
        t = b.split(".")[-1].lower()
        if t:
            base_tokens.add(t)
            if t.endswith("pe") and len(t) > 2:
                base_tokens.add(t[:-2])  # 'iterativepe' -> 'iterative'
    if base_tokens or said_type:
        checks["type_ok"] = ("pe" in said_type) or \
            any(tok in said_type for tok in base_tokens)

    checks.update(_port_checks("input", desc.get("inputs"), struct["inputs"]))
    checks.update(_port_checks("output", desc.get("outputs"), struct["outputs"]))
    return checks


# ---------------------------------------------------------------------------
# RAGAS plumbing
# ---------------------------------------------------------------------------
def _value(score):
    return getattr(score, "value", score)


def _is_unreachable(exc):
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, APIConnectionError):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _is_openai_new_gen_model(model: str) -> bool:
    m = model.lower()
    if len(m) >= 2 and m[0] == "o" and m[1] in "123456789":
        if len(m) == 2 or m[2] in "-_":
            return True
    if m == "codex-mini":
        return True
    if m.startswith("gpt-"):
        version_str = m[4:].split("-")[0].split("_")[0]
        try:
            return float(version_str) >= 5
        except ValueError:
            return False
    return False


def _fix_new_gen_model_args(llm, model):
    if getattr(llm, "provider", "openai") != "openai":
        return
    if not _is_openai_new_gen_model(model):
        return
    args = getattr(llm, "model_args", None)
    if not isinstance(args, dict):
        return
    args = dict(args)
    if "max_tokens" in args:
        args["max_completion_tokens"] = args.pop("max_tokens")
    args.setdefault("max_completion_tokens", 1024)
    args["temperature"] = 1.0
    args.pop("top_p", None)
    llm.model_args = args


def is_usable_reference(reference, min_words) -> bool:
    return isinstance(reference, str) and len(reference.split()) >= min_words


# ---------------------------------------------------------------------------
def load_source_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] skipping malformed line {i}: {e}", file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# Single pass: convert -> describe -> score
# ---------------------------------------------------------------------------
def process_row(obj, connector, ConvertToPE, context_queries, provider,
                user_input, sem, faith, loop, min_ref_words):
    original = obj.get("func_code_string") or obj.get("whole_func_string") or ""
    reference = obj.get("func_documentation_string") or ""
    out = {"func_name": obj.get("func_name"), "class_name": None,
           "func_url": obj.get("func_code_url")}

    try:
        converted = ConvertToPE(original, CONVERT_SECOND_ARG)
        converted_code = converted.pe
        out["class_name"] = converted.className
    except Exception as e:  # noqa: BLE001
        out["skipped"] = f"convert failed: {e!r}"
        return out

    try:
        desc = coerce_description(connector.describe(
            component_name=converted.className, kind="pe", code=converted_code,
            provider=provider, context_queries=context_queries))
    except Exception as e:  # noqa: BLE001
        if _is_unreachable(e):
            raise
        out["skipped"] = f"describe failed: {e!r}"
        return out
    if desc in (None, ""):
        out["skipped"] = "empty description"
        return out

    struct = extract_pe_structure(converted_code)
    core = flatten_desc(desc, CORE_FIELDS)
    plumb = flatten_desc(desc, PLUMBING_FIELDS)
    plumb_ctx = pe_context(converted_code, struct)

    def faithfulness(response, context, key):
        if not response or not context:
            return
        try:
            res = loop.run_until_complete(faith.ascore(
                user_input=user_input, response=response,
                retrieved_contexts=[context]))
            out[key] = _value(res)
        except Exception as e:  # noqa: BLE001
            if _is_unreachable(e):
                raise
            out[key] = None
            out[key + "_error"] = repr(e)

    # core claims vs ORIGINAL function (body is copied verbatim by the converter)
    faithfulness(core, original, "core_faithfulness")
    # plumbing claims vs CONVERTED PE + base-class contract
    faithfulness(plumb, plumb_ctx, "plumbing_faithfulness")

    # soft completeness vs reference docstring -- CORE only; thin ones skipped
    if is_usable_reference(reference, min_ref_words) and core:
        try:
            res = loop.run_until_complete(
                sem.ascore(reference=reference, response=core))
            out["core_similarity"] = _value(res)
        except Exception as e:  # noqa: BLE001
            if _is_unreachable(e):
                raise
            out["core_similarity"] = None
            out["core_similarity_error"] = repr(e)
    else:
        out["core_similarity"] = None

    out.update(deterministic_plumbing_checks(desc, struct))
    return out


def run(rows, provider, judge_model, embed_model, min_ref_words):
    from laminar.conversion.ConvertToPE import ConvertToPE
    from laminar.llms.LLMConnector import LLMConnector
    from laminar.llms.queries_templates import REQUEST_DESCRIPTION_CONTEXT_QUERIES

    connector = LLMConnector()
    user_input = ",".join(REQUEST_DESCRIPTION_CONTEXT_QUERIES)

    client = AsyncOpenAI()  # reads OPENAI_API_KEY
    llm = llm_factory(judge_model, client=client)
    _fix_new_gen_model_args(llm, judge_model)
    embeddings = embedding_factory(
        "openai", model=embed_model, client=client, interface="modern")
    sem = SemanticSimilarity(embeddings=embeddings)
    faith = Faithfulness(llm=llm)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        try:
            loop.run_until_complete(client.models.list())
        except APIConnectionError as e:
            raise SystemExit(f"OpenAI is not reachable: {e!r}")
        scored = []
        for idx, obj in enumerate(rows, 1):
            scored.append(process_row(
                obj, connector, ConvertToPE, REQUEST_DESCRIPTION_CONTEXT_QUERIES,
                provider, user_input, sem, faith, loop, min_ref_words))
            print(f"\rprocessed {idx}/{len(rows)}", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    return scored


# ---------------------------------------------------------------------------
NUMERIC_KEYS = ["core_faithfulness", "plumbing_faithfulness", "core_similarity"]
BOOL_KEYS = ["name_ok", "type_ok", "input_arity_ok", "output_arity_ok",
             "input_names_ok", "output_names_ok", "input_present", "output_present"]


def summarize(scored):
    n_skipped = sum(1 for s in scored if s.get("skipped"))
    print(f"\nProcessed {len(scored)} row(s); {n_skipped} skipped.")
    print("\nMean scores (ignoring failed / skipped rows):")
    for k in NUMERIC_KEYS:
        vals = [s[k] for s in scored if isinstance(s.get(k), (int, float))]
        if vals:
            print(f"  {k:22s}: {sum(vals)/len(vals):.4f}  (n={len(vals)})")
        else:
            print(f"  {k:22s}: no successful scores")
        errs = [s[k + "_error"] for s in scored if s.get(k + "_error")]
        if errs:
            print(f"    {len(errs)} row(s) failed; example: {errs[0]}")
    print("\nDeterministic plumbing pass-rates (rows where the fact was checkable):")
    for k in BOOL_KEYS:
        vals = [s[k] for s in scored if isinstance(s.get(k), bool)]
        if vals:
            print(f"  {k:22s}: {sum(vals)/len(vals):.2%}  (n={len(vals)})")


def write_csv(scored, out_path):
    fields = (["func_name", "class_name", "func_url"] + NUMERIC_KEYS + BOOL_KEYS
              + ["core_faithfulness_error", "plumbing_faithfulness_error",
                 "core_similarity_error", "skipped"])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for s in scored:
            w.writerow(s)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", help="Path to python_all.jsonl (source functions)")
    ap.add_argument("--out", default="pe_results.csv", help="Output CSV path")
    ap.add_argument("--count", type=int, default=None,
                    help="Evaluate this many RANDOM rows instead of all")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for --count")
    ap.add_argument("--provider", default="openai", help="Description provider")
    ap.add_argument("--judge-model", default="gpt-5.4-mini",
                    help="LLM used by Faithfulness")
    ap.add_argument("--embed-model", default="text-embedding-3-small",
                    help="Embedding model for SemanticSimilarity")
    ap.add_argument("--min-reference-words", type=int, default=8,
                    help="Compute core_similarity only when the reference docstring "
                         "has >= this many words. Use 0 to always compute it.")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY in your environment first.")

    rows = load_source_rows(args.jsonl)
    if not rows:
        sys.exit("No valid rows in the source file.")
    if args.count is not None and args.count < len(rows):
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.count]
        print(f"Sampling {len(rows)} random rows (seed={args.seed})", file=sys.stderr)
    else:
        print(f"Processing {len(rows)} rows", file=sys.stderr)

    scored = run(rows, args.provider, args.judge_model, args.embed_model,
                 args.min_reference_words)
    write_csv(scored, args.out)
    summarize(scored)
    print(f"\nPer-row scores written to {args.out}")


if __name__ == "__main__":
    main()
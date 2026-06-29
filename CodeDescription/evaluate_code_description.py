#!/usr/bin/env python3
"""
Evaluate generated code descriptions against reference docstrings with RAGAS.

Input: a JSONL file where each line is an object with:
  - "ID"                  : the source code (string)
  - "entry_expected_desc" : the reference description / docstring (string)
  - "entry_obtained_desc" : the generated description. Either a plain string,
                            or a structured dict (component_name, role, inputs,
                            outputs, processing, parameters, notes, ...).

How each field maps to RAGAS:
  response           = the (flattened) generated description  -> entry_obtained_desc
  reference          = the ground-truth docstring             -> entry_expected_desc
  retrieved_contexts = [the source code]                       -> ID
  user_input         = a fixed instruction

Metrics (RAGAS 0.4.x "collections" API):
  - SemanticSimilarity : embedding cosine similarity between generated & reference.
                         No LLM judge -> cheap and deterministic. Note your
                         generated text is far richer than the terse docstring,
                         so absolute values run moderate even for good output.
  - Faithfulness       : LLM judge. Are the claims in the generated description
                         actually grounded in the CODE (vs hallucinated)? This does
                         not use the reference at all, so it still gives signal even
                         when your docstrings are thin. Often the most useful metric
                         for code-description quality.

Evaluation is sequential: rows are scored one line at a time, and within each row
the metrics are run one after another. A single event loop is created up front and
reused for every metric call so the shared AsyncOpenAI client stays valid across
rows (the metrics' synchronous .score() spins up a fresh loop per call, which would
break a reused async client).

Requires:  OPENAI_API_KEY in the environment.
Usage:     python eval_code_descriptions.py data.jsonl --out code_description_results.csv
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys

from openai import AsyncOpenAI, APIConnectionError
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    SemanticSimilarity,
    Faithfulness,
)

from laminar.llms.queries_templates import REQUEST_DESCRIPTION_CONTEXT_QUERIES

# Descriptive fields of the structured obtained-description, in reading order.
# "model"/"provider" are metadata about *who generated it*, not part of the
# description, so they are intentionally excluded.
DESC_FIELDS = [
    "component_name",
    "component_type",
    "role",
    "inputs",
    "outputs",
    "processing",
    "parameters",
    "notes",
]

USER_INPUT = ",".join(REQUEST_DESCRIPTION_CONTEXT_QUERIES)


def flatten_obtained_desc(obtained) -> str:
    """Turn the structured generated description into one readable string."""
    if isinstance(obtained, str):
        return obtained.strip()
    if isinstance(obtained, dict):
        parts = []
        for key in DESC_FIELDS:
            val = obtained.get(key)
            if val:
                label = key.replace("_", " ").capitalize()
                parts.append(f"{label}: {val}")
        # Fall back to dumping anything unexpected but present.
        if not parts:
            return json.dumps(obtained, ensure_ascii=False)
        return "\n".join(parts)
    return str(obtained)


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] skipping malformed line {i}: {e}", file=sys.stderr)
                continue
            rows.append(
                {
                    "code": obj["ID"],
                    "reference": obj["entry_expected_desc"],
                    "response": flatten_obtained_desc(obj["entry_obtained_desc"]),
                }
            )
    return rows


def is_usable_reference(reference, min_words) -> bool:
    """Whether a ground-truth reference carries enough signal to score against.

    Very terse stubs like "DiskReader implementation." describe a name, not
    behaviour, so embedding-similarity against a rich generated description scores
    low for reasons that have nothing to do with the generation's quality. Dropping
    them keeps those rows from skewing the semantic_similarity mean.
    """
    if not isinstance(reference, str):
        return False
    return len(reference.split()) >= min_words


def _value(score):
    """collections metrics return an object with .value (and sometimes .reason)."""
    return getattr(score, "value", score)


def _is_unreachable(exc):
    """True if exc (or anything in its cause/context chain) is a connection error.

    RAGAS / instructor may wrap the original openai.APIConnectionError, so we walk
    the chain instead of a single isinstance check.
    """
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, APIConnectionError):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _is_openai_new_gen_model(model: str) -> bool:
    """True for GPT-5+/o-series models that require max_completion_tokens.

    RAGAS 0.4.x detects these to rename max_tokens -> max_completion_tokens, force
    temperature=1.0, and drop top_p. But its parser does int(version), so a decimal
    name like 'gpt-5.4-mini' raises and is misclassified as a legacy model. We redo
    the check with float() so decimal minor versions are handled.
    """
    m = model.lower()
    # o-series reasoning models: o1..o9, optionally followed by - or _
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
    """Work around the RAGAS detection bug for GPT-5+/o-series judge models.

    These models reject 'max_tokens' (need 'max_completion_tokens'), require
    temperature=1.0, and don't accept top_p. When RAGAS's own detector misses the
    model (e.g. 'gpt-5.4-mini'), it sends the legacy params and the call 400s, so we
    rewrite the args ourselves. No-op for legacy models and non-OpenAI providers.
    """
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


def score_row(row, sem, faith, loop):
    """Score a single row, one metric after another, on the shared event loop."""
    out = {}

    # SemanticSimilarity: embedding cosine similarity, no LLM judge.
    try:
        res = loop.run_until_complete(
            sem.ascore(reference=row["reference"], response=row["response"])
        )
        out["semantic_similarity"] = _value(res)
    except Exception as e:  # noqa: BLE001 - record and keep going
        if _is_unreachable(e):
            raise  # OpenAI not reachable -> abort the whole run
        out["semantic_similarity"] = None
        out["semantic_similarity_error"] = repr(e)

    # Faithfulness: are the generated claims grounded in the code?
    try:
        res = loop.run_until_complete(
            faith.ascore(
                user_input=USER_INPUT,
                response=row["response"],
                retrieved_contexts=[row["code"]],
            )
        )
        out["faithfulness"] = _value(res)
    except Exception as e:  # noqa: BLE001 - record and keep going
        if _is_unreachable(e):
            raise  # OpenAI not reachable -> abort the whole run
        out["faithfulness"] = None
        out["faithfulness_error"] = repr(e)

    return out


def run(rows, judge_model, embed_model):
    client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
    llm = llm_factory(judge_model, client=client)
    _fix_new_gen_model_args(llm, judge_model)
    embeddings = embedding_factory(
        "openai", model=embed_model, client=client, interface="modern"
    )

    sem = SemanticSimilarity(embeddings=embeddings)
    faith = Faithfulness(llm=llm)

    # One event loop, reused for every metric call across all rows.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Fail fast if OpenAI is unreachable, before doing any scoring work.
        try:
            loop.run_until_complete(client.models.list())
        except APIConnectionError as e:
            raise SystemExit(f"OpenAI is not reachable: {e!r}")

        scored = []
        for idx, row in enumerate(rows, 1):
            scored.append(score_row(row, sem, faith, loop))
            print(f"\rscored {idx}/{len(rows)}", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    return scored


def summarize(scored):
    keys = ["semantic_similarity", "faithfulness"]
    print("\nMean scores (ignoring failed rows):")
    for k in keys:
        vals = [s[k] for s in scored if s.get(k) is not None]
        if vals:
            print(f"  {k:22s}: {sum(vals) / len(vals):.4f}  (n={len(vals)})")
        else:
            print(f"  {k:22s}: no successful scores")
        # Surface why rows failed, so the cause isn't hidden in the CSV.
        errs = [s[k + "_error"] for s in scored if s.get(k + "_error")]
        if errs:
            print(f"    {len(errs)} row(s) failed; example error: {errs[0]}")


def write_csv(rows, scored, out_path):
    fields = [
        "semantic_similarity",
        "faithfulness",
        "semantic_similarity_error",
        "faithfulness_error",
        "reference",
        "response",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r, s in zip(rows, scored):
            w.writerow({**s, "reference": r["reference"], "response": r["response"]})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", help="Path to the input JSONL file")
    ap.add_argument("--out", default="ragas_results.csv", help="Output CSV path")
    ap.add_argument("--count", type=int, default=None,
                    help="Evaluate this many RANDOM entries instead of all of them")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for --count, so the sample is reproducible")
    ap.add_argument("--judge-model", default="gpt-5.4-mini",
                    help="LLM used by Faithfulness")
    ap.add_argument("--embed-model", default="text-embedding-3-small",
                    help="Embedding model for SemanticSimilarity")
    ap.add_argument("--min-reference-words", type=int, default=15,
                    help="Skip rows whose ground-truth reference has fewer than "
                         "this many words (filters out thin stubs like 'DiskReader "
                         "implementation.' that skew similarity). Use 0 to keep all.")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY in your environment first.")

    rows = load_rows(args.jsonl)
    if not rows:
        sys.exit("No valid rows found in the input file.")

    # Drop rows whose ground-truth reference is too thin to score against, before
    # sampling, so a --count sample is drawn from usable rows only.
    if args.min_reference_words > 0:
        kept, dropped = [], []
        for r in rows:
            target = kept if is_usable_reference(
                r["reference"], args.min_reference_words) else dropped
            target.append(r)
        if dropped:
            print(f"Skipped {len(dropped)} row(s) with a reference under "
                  f"{args.min_reference_words} words. Examples:", file=sys.stderr)
            for r in dropped[:3]:
                print(f"  - {r['reference']!r}", file=sys.stderr)
        rows = kept
        if not rows:
            sys.exit("All rows were filtered out by --min-reference-words.")

    # Pick a random subset if --count was given (and the file has more than that).
    if args.count is not None and args.count < len(rows):
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.count]
        print(f"Loaded {len(rows)} random rows (seed={args.seed}) from {args.jsonl}",
              file=sys.stderr)
    else:
        print(f"Loaded {len(rows)} rows from {args.jsonl}", file=sys.stderr)

    scored = run(rows, args.judge_model, args.embed_model)
    write_csv(rows, scored, args.out)
    summarize(scored)
    print(f"\nPer-row scores written to {args.out}")


if __name__ == "__main__":
    main()
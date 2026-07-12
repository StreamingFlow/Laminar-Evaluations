#!/usr/bin/env python3
"""Evaluate Laminar's generated PE descriptions.

Two measurements:

  [1] FAITHFULNESS -- is the description TRUE of the code?      (LLM judge, per row)
        ref_faithfulness       : the ORIGINAL DOCSTRING's claims, checked against the
                                 source it documents. A gate on the reference, not a
                                 score for the describer -- a stale or vacuous docstring
                                 makes everything downstream noise.
        generated_faithfulness : the GENERATED DESCRIPTION's claims, checked against the
                                 original function AND the converted PE. Precision;
                                 catches hallucination.

  [2] FINDABILITY -- does the description make the PE FINDABLE?  (local registry search)
        Register every described PE locally, then query the registry with the ORIGINAL
        DOCSTRING and see whether the right PE comes back.
          OK      : the PE is the top hit
          PARTIAL : it is in the top-k, but other PEs beat it
          MISS    : it is not in the top-k at all
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console  # kept: pre-launch warnings + score rendering
from rich.table import Table
from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (DataTable, Footer, Header, Label, ProgressBar,
                             Static, TabbedContent, TabPane, TextArea)

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from laminar.conversion.ConvertToPE import ConvertToPE
from laminar.llms.LLMConnector import LLMConnector
from laminar.llms.queries_templates import REQUEST_DESCRIPTION_CONTEXT_QUERIES

from sentence_transformers import SentenceTransformer, util


CORE_FIELDS = ["role", "processing", "parameters", "notes"]
PROSE_HEADINGS = ["summary", "role", "parameters"]  # COMPONENT TYPE is plumbing
ALL_DESC_FIELDS = ["component_name", "component_type", "role", "inputs", "outputs",
                   "processing", "parameters", "notes", "tags", "description"]
IO_FIELDS = ["inputs", "outputs"]

CONVERT_SECOND_ARG = False

FOUND, PARTIAL, MISS = 0, 1, 2
TAGS = ("OK", "PARTIAL", "MISS")
TAG_STYLE = {"OK": "bold green", "PARTIAL": "bold yellow", "MISS": "bold red"}

BOILERPLATE_RE = re.compile(
    r"^\W*(todo|fixme|tbd|xxx|n/?a|none|see (above|below|docs?)|deprecated|"
    r"docstring|no description|placeholder|\.{3})\W*$", re.I)

FIELD_LINE_RE = re.compile(r"^\s*(:param|:type|:returns?|:rtype|:raises|@param|"
                           r"@return|@type|Args:|Returns:|Raises:)", re.I)


def _prose_words(reference: str) -> int:
    """Words that are actual prose, ignoring :param:/@param blocks -- a docstring that is
    nothing but a parameter table describes no behaviour and makes a useless query."""
    return sum(len(line.split()) for line in reference.splitlines()
               if not FIELD_LINE_RE.match(line))


def reference_quality(reference, func_name, min_words):
    """(usable, code, detail). Deterministic, no LLM."""
    if not isinstance(reference, str) or not reference.strip():
        return False, "empty", "docstring is empty"
    ref = reference.strip()
    if BOILERPLATE_RE.match(ref):
        return False, "boilerplate", "docstring is boilerplate"
    n = _prose_words(ref)
    if n < min_words:
        return False, "thin", f"docstring too short ({n} prose words < {min_words})"
    # "def get_user_name" / "Get user name." -- says nothing the name didn't.
    name_tokens = set(re.split(r"[_\W]+", (func_name or "").lower())) - {""}
    ref_tokens = set(re.split(r"[_\W]+", ref.lower())) - {""}
    if name_tokens and ref_tokens <= name_tokens:
        return False, "name_restatement", "docstring only restates the function name"
    return True, "ok", "ok"


_HEADING_RE = re.compile(
    r"^[ \t]*(COMPONENT TYPE|SUMMARY|ROLE|PARAMETERS|NOTES|INPUTS|OUTPUTS)"
    r"[ \t]*:?[ \t]*$", re.I | re.M)
_FENCE_RE = re.compile(r"^```(?:json)?[ \t]*\n?|\n?```$", re.I)


def split_headed_prose(text) -> dict[str, str]:
    """'SUMMARY\n...\n\nROLE\n...' -> {'summary': '...', 'role': '...'}. Returns {}
    when there are no headings at all, so callers can treat the text as one blob."""
    if not isinstance(text, str) or not text.strip():
        return {}
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return {}
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if body:
            out[m.group(1).strip().lower()] = body
    return out


def _maybe_json(s: str):
    """A JSON object that arrives as *text* -- possibly in ``` fences -- is a payload, not
    prose. Indexing the raw braces would be nonsense."""
    t = _FENCE_RE.sub("", s.strip()).strip()
    if not (t.startswith("{") and t.endswith("}")):
        return s.strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return s.strip()
    return _unwrap_desc(obj) if isinstance(obj, dict) else s.strip()


def coerce_description(desc):
    if desc is None:
        return ""
    if isinstance(desc, str):
        return _maybe_json(desc)
    if isinstance(desc, dict):
        return _unwrap_desc(desc)
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(desc, attr, None)
        if callable(fn):
            try:
                return _unwrap_desc(fn())
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
    # {"result": "{\"inputs\": ...}"} -- still a payload; unwrap rather than stringify.
    strings = [v for v in d.values() if isinstance(v, str) and v.strip()]
    if len(d) == 1 and len(strings) == 1:
        obj = _maybe_json(strings[0])
        if isinstance(obj, dict):
            return obj
    return d


def _io_lines(desc: dict) -> list[str]:
    lines = []
    for k in IO_FIELDS:
        v = desc.get(k)
        if not v:
            continue
        lines.append(f"{k.capitalize()}: " +
                     ("; ".join(str(x) for x in v) if isinstance(v, (list, tuple))
                      else str(v)))
    return lines


def flatten_core(desc, include_io: bool = False) -> str:
    """The semantic content, either schema, as flat text. This is what gets JUDGED and
    what gets INDEXED. Returns "" when there is nothing semantic at all, so the caller can
    skip the row rather than register an empty description."""
    if isinstance(desc, str):
        return desc.strip()
    if not isinstance(desc, dict):
        return str(desc)

    parts = [f"{k.replace('_', ' ').capitalize()}: {desc[k]}"
             for k in CORE_FIELDS if desc.get(k)]

    if not parts:  # prose schema
        blob = desc.get("description")
        if isinstance(blob, str) and blob.strip():
            sections = split_headed_prose(blob)
            if sections:
                parts = [f"{h.capitalize()}: {sections[h]}"
                         for h in PROSE_HEADINGS if sections.get(h)]
                # Headings we did not anticipate still carry meaning -- keep them rather
                # than throw the model's work away.
                parts += [f"{h.capitalize()}: {sections[h]}" for h in sections
                          if h not in PROSE_HEADINGS and h != "component type"]
            else:
                parts = [blob.strip()]  # no headings -- take the whole thing

    if include_io:
        parts += _io_lines(desc)
    return "\n".join(p for p in parts if p.strip())


def flatten_full(desc) -> str:
    """Every field the describer produced, for the CSV and for eyeballing."""
    if isinstance(desc, str):
        return desc.strip()
    if not isinstance(desc, dict):
        return str(desc)

    def render(v):
        if isinstance(v, (list, tuple)):
            return "\n".join(f"  - {x}" for x in v)
        if isinstance(v, dict):
            return "\n".join(f"  - {k}: {val}" for k, val in v.items())
        return str(v)

    keys = [k for k in ALL_DESC_FIELDS if desc.get(k)]
    keys += [k for k in desc if k not in ALL_DESC_FIELDS and desc.get(k)]
    return "\n\n".join(f"{k.replace('_', ' ').upper()}\n{render(desc[k])}" for k in keys)


# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------
def _value(score):
    return getattr(score, "value", score)


def _is_openai_new_gen_model(model: str) -> bool:
    m = model.lower()
    if len(m) >= 2 and m[0] == "o" and m[1] in "123456789" and (len(m) == 2 or m[2] in "-_"):
        return True
    if m == "codex-mini":
        return True
    if m.startswith("gpt-"):
        try:
            return float(m[4:].split("-")[0].split("_")[0]) >= 5
        except ValueError:
            return False
    return False


def _fix_new_gen_model_args(llm, model):
    if getattr(llm, "provider", "openai") != "openai" or not _is_openai_new_gen_model(model):
        return
    args = getattr(llm, "model_args", None)
    if not isinstance(args, dict):
        return
    args = dict(args)
    if "max_tokens" in args:
        args["max_completion_tokens"] = args.pop("max_tokens")
    # 1024 truncates the judge mid-decomposition on long docstrings and the row is lost
    # to IncompleteOutputException.
    args["max_completion_tokens"] = max(args.get("max_completion_tokens") or 0, 4096)
    args["temperature"] = 1.0  # new-gen models reject anything else, so the judge is
    args.pop("top_p", None)  # NON-DETERMINISTIC; expect run-to-run drift
    llm.model_args = args


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def load_source_rows(path, warn=print):
    rows, bad = [], 0
    with open(path, "rb") as f:  # binary: decode per line, not per buffer
        for raw in f:
            try:
                line = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                bad += 1
                continue
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            # "4" is valid JSON but not a function: no func_code_string to convert.
            if not isinstance(obj, dict):
                bad += 1
                continue
            rows.append(obj)
    if bad:
        warn(f"[warn] skipped {bad} unreadable line(s); kept {len(rows)}")
    return rows


def select_rows(rows, shuffle=False, seed=0, warn=print):
    if shuffle:
        rows = list(rows)
        random.Random(seed).shuffle(rows)
    order = f"random order, seed={seed}" if shuffle else "file order"
    warn(f"Evaluating {len(rows)} row(s) available from file ({order})")
    return rows


@dataclass
class Result:
    idx: int
    func_name: str | None = None
    class_name: str | None = None  # the PE's name -- what search must return
    original: str = ""
    reference: str = ""  # original docstring: the phase-B QUERY
    converted_code: str = ""  # kept, so `--index code` has something to index
    generated_core: str = ""  # judged AND indexed
    description: str = ""  # full payload, rendered, for the CSV
    skipped: str | None = None  # pipeline failure -- no PE, no description
    excluded: str | None = None  # usable output, unusable reference
    exclusion_code: str = ""
    scores: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    registered: bool = False  # made it into the local registry
    verdict: int | None = None  # FOUND / PARTIAL / MISS, from phase B
    rank: int | None = None  # 0-based position in the hits; None = absent
    hits: list[str] = field(default_factory=list)

    @property
    def counted(self) -> bool:
        """Feeds the aggregates: a real PE, a trusted docstring, a usable description."""
        return not self.skipped and not self.excluded

    @property
    def found(self) -> bool | None:
        """True if this PE turned up anywhere in the top-k search results."""
        if self.verdict is None:
            return None
        return self.verdict in (FOUND, PARTIAL)

    def as_csv_row(self):
        return {
            "func_name": self.func_name, "class_name": self.class_name,
            "counted": self.counted, "skipped": self.skipped,
            "excluded": self.excluded, "exclusion_code": self.exclusion_code,
            "registered": self.registered,
            "ref_faithfulness": self.scores.get("ref_faithfulness"),
            "generated_faithfulness": self.scores.get("generated_faithfulness"),
            "found": "" if self.found is None else ("yes" if self.found else "no"),
            "verdict": None if self.verdict is None else TAGS[self.verdict],
            "rank": self.rank,  # 0-based; blank = never returned at all
            "hits": " | ".join(self.hits),
            "errors": " | ".join(f"{k}: {v}" for k, v in self.errors.items()),
            "description": self.description,
        }


CSV_FIELDS = ["func_name", "class_name", "counted", "skipped", "excluded",
              "exclusion_code", "registered", "ref_faithfulness",
              "generated_faithfulness", "found", "verdict", "rank", "hits", "errors",
              "description"]


def _fmt_exc(e: BaseException) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip()


def result_status(res: "Result") -> tuple[str, str]:
    """(label, rich-style) for one PE, used by both the table and the status bar."""
    if res.skipped:
        return "SKIP", "red"
    if res.excluded:
        return "EXCL", "yellow"
    if res.verdict is None:
        return "…", "dim"
    tag = TAGS[res.verdict]
    return tag, TAG_STYLE[tag]


def found_cell(res: "Result") -> Text:
    """Explicit yes/no 'was this PE found' cell for the table."""
    if res.verdict is None:
        return Text("…", style="dim")
    if res.found:
        return Text("yes", style="green")
    return Text("no", style="red")


# ---------------------------------------------------------------------------
# Local registry
# ---------------------------------------------------------------------------
@dataclass
class Entry:
    name: str  # the PE class name -- what a search result is identified by
    code: str  # the converted PE source
    description: str  # the generated description


class LocalRegistry:
    """An in-memory PE registry with a semantic search over it.

    The ranking is done by sentence-transformers' util.semantic_search;
    """

    def __init__(self, model_name: str, index: str = "description"):
        self.model_name = model_name
        self.index = index  # "description" | "code" | "both"
        self.entries: list[Entry] = []
        self.built = False

    def add(self, name: str, code: str, description: str) -> None:
        self.entries.append(Entry(name=name, code=code, description=description))

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def chance(self) -> float:
        """Random-baseline OK rate. An OK rate near this is measuring nothing."""
        return 1.0 / len(self.entries) if self.entries else float("nan")

    def _indexed_text(self, e: Entry) -> str:
        if self.index == "code":
            return e.code
        if self.index == "both":
            return f"{e.description}\n\n{e.code}"
        return e.description

    def build(self) -> None:
        """Embed the whole registry once. Called after phase A, before any search, so
        every query faces the same index."""


        if not self.entries:
            raise RuntimeError("local registry is empty -- nothing was registered")
        self.model = SentenceTransformer(self.model_name)
        self.emb = self.model.encode(
            [self._indexed_text(e) for e in self.entries],
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        self.built = True

    def search(self, query: str, top_k: int) -> list[dict]:
        """Score-sorted hits, shaped like the server's: [{"name": ..., "score": ...}]."""

        if not self.built:
            raise RuntimeError("registry index not built")
        q = self.model.encode([query], convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        hits = util.semantic_search(q, self.emb, top_k=min(top_k, len(self.entries)))[0]
        return [{"name": self.entries[h["corpus_id"]].name, "score": float(h["score"])}
                for h in hits]

    def dump(self, path: str) -> None:
        """Write the registry out, so a bad verdict can be inspected rather than guessed
        at. Includes the code even when only the description was indexed."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"name": e.name, "description": e.description, "code": e.code}
                       for e in self.entries], f, indent=2)


def search_one(registry: LocalRegistry, res: Result, top_k: int) -> None:
    """Query the registry with the ORIGINAL DOCSTRING and look for this PE by name.

    top_k is the cut-off for the verdict AND for how deep we look: a PE buried below it is
    a MISS, because nobody scrolls.
    """
    hits = registry.search(res.reference, top_k)
    names = [h["name"] for h in hits]
    res.hits = names
    res.rank = names.index(res.class_name) if res.class_name in names else None
    if res.rank == 0:
        res.verdict = FOUND
    elif res.rank is not None:
        res.verdict = PARTIAL
    else:
        res.verdict = MISS


# ---------------------------------------------------------------------------
# PHASE A: convert -> gate -> describe -> judge -> register (locally)
# ---------------------------------------------------------------------------
class Describer:
    def __init__(self, provider, judge_model, min_ref_words, min_ref_faith,
                 score_io=False):
        self.provider = provider
        self.judge_model = judge_model
        self.min_ref_words = min_ref_words
        self.min_ref_faith = min_ref_faith
        self.score_io = score_io

    async def setup(self):
        self.ConvertToPE = ConvertToPE
        self.connector = LLMConnector()
        self.context_queries = REQUEST_DESCRIPTION_CONTEXT_QUERIES
        self.user_input = ",".join(REQUEST_DESCRIPTION_CONTEXT_QUERIES)

        llm = llm_factory(self.judge_model, client=AsyncOpenAI())  # reads OPENAI_API_KEY
        _fix_new_gen_model_args(llm, self.judge_model)
        self.faith = Faithfulness(llm=llm)

    async def _faith(self, res: Result, response, contexts, key):
        """Faithfulness of `response` against `contexts`. RAGAS checks each claim against
        the union, so extra context can only help a claim be supported, never hurt it."""
        contexts = [c for c in contexts if c and c.strip()]
        if not response or not contexts:
            return
        try:
            score = await self.faith.ascore(user_input=self.user_input,
                                            response=response,
                                            retrieved_contexts=contexts)
            res.scores[key] = _value(score)
        except Exception as e:  # noqa: BLE001
            res.scores[key] = None
            res.errors[key] = repr(e)

    async def process(self, idx: int, obj: dict, registry: LocalRegistry) -> Result:
        res = Result(
            idx=idx,
            func_name=obj.get("func_name"),
            original=obj.get("func_code_string") or obj.get("whole_func_string") or "",
            reference=obj.get("func_documentation_string") or "",
        )

        try:
            converted = await asyncio.to_thread(
                self.ConvertToPE, res.original, CONVERT_SECOND_ARG)
            res.converted_code = converted.pe or ""
            res.class_name = converted.className
        except Exception as e:  # noqa: BLE001
            res.skipped = f"convert failed: {e!r}"
            return res

        if not res.converted_code.strip() or not res.class_name:
            res.skipped = "no converted PE (empty body or no class name)"
            return res

        usable, code, detail = reference_quality(
            res.reference, res.func_name, self.min_ref_words)
        if not usable:
            res.exclusion_code, res.excluded = code, detail
            return res

        await self._faith(res, res.reference, [res.original], "ref_faithfulness")

        rf = res.scores.get("ref_faithfulness")
        if not isinstance(rf, (int, float)):
            res.exclusion_code = "unscorable"
            res.excluded = "docstring could not be scored"
            return res
        if rf < self.min_ref_faith:
            res.exclusion_code = "ungrounded"
            res.excluded = (f"docstring not grounded in its own source "
                            f"({rf:.2f} < {self.min_ref_faith})")
            return res

        try:
            raw = await asyncio.to_thread(lambda: self.connector.describe(
                component_name=res.class_name, kind="pe", code=res.converted_code,
                provider=self.provider, context_queries=self.context_queries))
        except Exception as e:  # noqa: BLE001
            res.skipped = f"describe failed: {e!r}"
            return res

        desc = coerce_description(raw)
        res.description = flatten_full(desc)
        res.generated_core = flatten_core(desc, include_io=self.score_io)
        if not res.generated_core:
            keys = ", ".join(sorted(desc)) if isinstance(desc, dict) else type(desc).__name__
            res.skipped = f"no core fields in description (got: {keys})"
            return res

        await self._faith(res, res.generated_core,
                          [res.original, res.converted_code], "generated_faithfulness")

        registry.add(res.class_name, res.converted_code, res.generated_core)
        res.registered = True
        return res


def _mean(results, key):
    vals = [r.scores.get(key) for r in results
            if isinstance(r.scores.get(key), (int, float))]
    return (sum(vals) / len(vals)) if vals else None


def _fmt(v):
    return "  -  " if not isinstance(v, (int, float)) else f"{v:.3f}"


def build_summary(results, registry, top_k):
    """The three headline numbers, computed once and shared by the app and the console."""
    counted = [r for r in results if r.counted]
    searched = [r for r in counted if r.verdict is not None]
    total = len(searched)
    found_top = sum(1 for r in searched if r.verdict == FOUND)
    matched = sum(1 for r in searched if r.verdict in (FOUND, PARTIAL))
    miss = sum(1 for r in searched if r.verdict == MISS)
    return {
        "ref_faithfulness": _mean(counted, "ref_faithfulness"),
        "generated_faithfulness": _mean(counted, "generated_faithfulness"),
        "total_searched": total,
        "found_top": found_top,   # exact top hit (verdict == FOUND)
        "matched": matched,       # anywhere in top-k (FOUND or PARTIAL)
        "miss": miss,
        "registry_size": len(registry) if registry else 0,
        "chance": registry.chance if registry and registry.entries else float("nan"),
        "top_k": top_k,
    }


def render_summary_table(s) -> Table:
    """The three final scores, as a full Rich table (used for the console report)."""
    total = s["total_searched"] or 0
    t = Table(title="Final scores", show_header=False, box=None, padding=(0, 3))
    t.add_column(style="bold")
    t.add_column(justify="right")
    t.add_row("① RAGAS  docstring → source faithfulness",
              _fmt(s["ref_faithfulness"]))
    t.add_row("② RAGAS  description → source faithfulness",
              _fmt(s["generated_faithfulness"]))
    matched_pct = f"  ({100.0 * s['matched'] / total:.0f}%)" if total else ""
    t.add_row(f"③ Search matches (sentence-transformers, top-{s['top_k']})",
              f"{s['matched']} / {total}{matched_pct}")
    t.add_row("     ├ exact top hit", str(s["found_top"]))
    t.add_row("     └ misses", str(s["miss"]))
    chance = (f"{100.0 * s['chance']:.1f}%"
              if isinstance(s["chance"], float) and s["chance"] == s["chance"] else "?")
    t.add_row("registry size / random-chance baseline",
              f"{s['registry_size']}  /  {chance}")
    return t


def render_summary_cell(s) -> Text:
    """Compact three-score report for the dedicated cell in the bottom bar.

    Rendered as plain Text (not a Rich Table) so it can never lay itself out wider
    than the cell -- Static will wrap/clip it to the fixed cell width instead of
    spilling past the right edge of the screen.
    """
    total = s["total_searched"] or 0
    matched_pct = f" ({100.0 * s['matched'] / total:.0f}%)" if total else ""
    chance = (f"{100.0 * s['chance']:.0f}%"
              if isinstance(s["chance"], float) and s["chance"] == s["chance"] else "?")

    t = Text()
    t.append("FINAL RESULTS\n", style="bold cyan")
    t.append("① doc>src  ", style="bold")
    t.append(f"{_fmt(s['ref_faithfulness'])}\n")
    t.append("② gen>src  ", style="bold")
    t.append(f"{_fmt(s['generated_faithfulness'])}\n")
    t.append("③ matched  ", style="bold")
    t.append(f"{s['matched']}/{total}{matched_pct}\n")
    t.append("reg/chance ", style="dim")
    t.append(f"{s['registry_size']} / {chance}", style="dim")
    return t


def write_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r.as_csv_row())


# ---------------------------------------------------------------------------
# Textual UI
# ---------------------------------------------------------------------------
class PEEvalApp(App):
    """Interactive dashboard for the PE-description evaluation."""

    CSS = """
    #main { height: 1fr; }
    #pe-table { width: 45%; border-right: solid $accent; }
    #details { width: 55%; }
    TextArea { height: 1fr; }
    .detail-scroll { padding: 1 2; height: 1fr; }

    #status-bar {
        height: auto;
        max-height: 8;
        dock: bottom;
        background: $panel;
        border-top: solid $accent;
    }
    #status-main { height: auto; }
    #status-left { width: 1fr; height: auto; }
    #status-line { height: auto; padding: 1 1 0 1; }
    #bars { height: auto; padding-bottom: 1; }
    #bars > Horizontal { height: auto; padding: 0 1; align-vertical: middle; }
    #bars Label { width: 10; content-align: left middle; }
    #bars ProgressBar { width: 1fr; }

    /* the dedicated final-results cell, on the right of the bottom bar */
    #final-cell {
        width: 30;
        height: auto;
        padding: 1 1;
        border-left: solid $accent;
        background: $boost;
        overflow-x: hidden;
        overflow-y: auto;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, rows: list[dict], args) -> None:
        super().__init__()
        self.rows = rows
        self.args = args
        self.results: list[Result] = []
        self.by_key: dict[str, Result] = {}
        self.registry: LocalRegistry | None = None
        self.final_summary: dict | None = None  # printed to the console after exit
        self._phase_a_total = args.count if args.count else len(rows)

    # -- layout --------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            table = DataTable(id="pe-table", cursor_type="row", zebra_stripes=True)
            table.add_column("Function", key="func")
            table.add_column("Status", key="status")
            table.add_column("Found", key="found")
            table.add_column("Gen>Src", key="gen")
            yield table
            with TabbedContent(id="details"):
                with TabPane("Original Code", id="tab-orig-code"):
                    yield TextArea.code_editor(
                        "", language="python", read_only=True, id="orig-code")
                with TabPane("Original Description", id="tab-orig-desc"):
                    with VerticalScroll(classes="detail-scroll"):
                        yield Static("", id="orig-desc")
                with TabPane("Converted PE", id="tab-conv-code"):
                    yield TextArea.code_editor(
                        "", language="python", read_only=True, id="conv-code")
                with TabPane("Generated Description", id="tab-gen-desc"):
                    with VerticalScroll(classes="detail-scroll"):
                        yield Static("", id="gen-desc")
                with TabPane("Scores", id="tab-scores"):
                    with VerticalScroll(classes="detail-scroll"):
                        yield Static("", id="scores")
        with Vertical(id="status-bar"):
            with Horizontal(id="status-main"):
                with Vertical(id="status-left"):
                    yield Static("Starting…", id="status-line")
                    with Vertical(id="bars"):
                        with Horizontal():
                            yield Label("Convert & Score")
                            yield ProgressBar(total=self._phase_a_total, id="bar-a")
                        with Horizontal():
                            yield Label("Search")
                            yield ProgressBar(total=None, id="bar-b")
                # dedicated final-results cell on the right of the bottom bar
                yield Static(Text("final results\npending…", style="dim"),
                             id="final-cell")
        yield Footer()

    def on_mount(self) -> None:
        self.run_evaluation()

    # -- table plumbing ------------------------------------------------------
    def add_result(self, res: Result) -> None:
        """Record every result (for CSV + tally), but only show a table row for
        PEs that actually converted."""
        self.results.append(res)

        if not res.converted_code.strip() or not res.class_name:
            return

        key = str(res.idx)
        self.by_key[key] = res
        label, style = result_status(res)
        table = self.query_one("#pe-table", DataTable)
        first_row = table.row_count == 0
        table.add_row(
            res.func_name or "?",
            Text(label, style=style),
            found_cell(res),
            _fmt(res.scores.get("generated_faithfulness")),
            key=key,
        )
        if first_row:  # populate the panes for the first PE that lands
            self.show_details(res)

    def refresh_result(self, res: Result) -> None:
        """Update an existing row (used after phase B assigns a verdict)."""
        key = str(res.idx)
        label, style = result_status(res)
        table = self.query_one("#pe-table", DataTable)
        table.update_cell(key, "status", Text(label, style=style))
        table.update_cell(key, "found", found_cell(res))
        table.update_cell(
            key, "gen", _fmt(res.scores.get("generated_faithfulness")))
        if self._current_key() == key:
            self.show_details(res)

    def _current_key(self) -> str | None:
        table = self.query_one("#pe-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            return key.value
        except Exception:  # noqa: BLE001
            return None

    def on_data_table_row_highlighted(
            self, event: DataTable.RowHighlighted) -> None:
        res = self.by_key.get(event.row_key.value)
        if res is not None:
            self.show_details(res)

    def show_details(self, res: Result) -> None:
        self.query_one("#orig-code", TextArea).load_text(res.original or "")
        self.query_one("#conv-code", TextArea).load_text(res.converted_code or "")
        self.query_one("#orig-desc", Static).update(res.reference or "[dim](none)[/]")
        self.query_one("#gen-desc", Static).update(
            res.description or "[dim](not described)[/]")
        self.query_one("#scores", Static).update(self._render_scores(res))

    def _render_scores(self, res: Result) -> Table:
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold")
        t.add_column()
        t.add_row("docstring > source", _fmt(res.scores.get("ref_faithfulness")))
        t.add_row("generated > source",
                  _fmt(res.scores.get("generated_faithfulness")))
        if res.found is None:
            found = Text("…", style="dim")
        else:
            found = (Text("yes", style="green") if res.found
                     else Text("no", style="red"))
        t.add_row("found in top-k", found)
        verdict = "-" if res.verdict is None else TAGS[res.verdict]
        t.add_row("verdict", verdict)
        t.add_row("rank", "-" if res.rank is None else str(res.rank))
        t.add_row("registered", "yes" if res.registered else "no")
        if res.hits:
            t.add_row("top hits", ", ".join(res.hits))
        if res.skipped:
            t.add_row("skipped", Text(res.skipped, style="red"))
        if res.excluded:
            t.add_row("excluded", Text(res.excluded, style="yellow"))
        for k, v in res.errors.items():
            t.add_row(f"error:{k}", Text(v, style="red"))
        return t

    def set_status(self, text: str) -> None:
        self.query_one("#status-line", Static).update(text)

    def set_final_cell(self, renderable) -> None:
        self.query_one("#final-cell", Static).update(renderable)

    def update_tally(self) -> None:
        counted = [r for r in self.results if r.counted]
        searched = [r for r in counted if r.verdict is not None]
        n = len(searched) or 1
        ok = sum(1 for r in searched if r.verdict == FOUND)
        part = sum(1 for r in searched if r.verdict == PARTIAL)
        miss = sum(1 for r in searched if r.verdict == MISS)
        skipped = sum(1 for r in self.results if r.skipped)
        excluded = sum(1 for r in self.results if not r.skipped and r.excluded)
        reg = len(self.registry) if self.registry else 0
        self.set_status(
            f"[bold]doc>src[/] {_fmt(_mean(counted, 'ref_faithfulness'))}  "
            f"[bold]gen>src[/] {_fmt(_mean(counted, 'generated_faithfulness'))}   "
            f"[green]OK {ok} ({100.0*ok/n:.0f}%)[/] "
            f"[yellow]PART {part} ({100.0*part/n:.0f}%)[/] "
            f"[red]MISS {miss} ({100.0*miss/n:.0f}%)[/]   "
            f"counted {len(counted)} / skip {skipped} / excl {excluded}"
            + (f"   registry {reg}" if reg else "")
        )

    @work(exclusive=True)
    async def run_evaluation(self) -> None:
        args = self.args
        self.set_status("Setting up describer…")
        describer = Describer(args.provider, args.judge_model,
                              args.min_reference_words, args.min_ref_faithfulness,
                              score_io=args.score_io)
        await describer.setup()

        registry = LocalRegistry(args.rank_model, index=args.index)
        self.registry = registry
        registered: list[Result] = []
        bar_a = self.query_one("#bar-a", ProgressBar)

        self.sub_title = "Phase A — convert / describe"
        for i, obj in enumerate(self.rows, 1):
            res = await describer.process(i, obj, registry)
            self.add_result(res)
            if res.registered:
                registered.append(res)
            # advance per-registered when --count, else per-row
            if args.count is None or res.registered:
                bar_a.advance(1)
            self.update_tally()
            if args.count is not None and len(registered) >= args.count:
                break

        if not registered:
            self.set_status("[red]Nothing registered — no PE survived phase A.")
            write_csv(self.results, args.out)
            self.final_summary = build_summary(self.results, registry, args.top_k)
            self.set_final_cell(render_summary_cell(self.final_summary))
            return

        self.sub_title = (f"Phase B — searching {len(registry)} PEs "
                          f"(index: {args.index})")
        self.set_status(f"Building index over {len(registry)} PEs…")
        await asyncio.to_thread(registry.build)

        bar_b = self.query_one("#bar-b", ProgressBar)
        bar_b.update(total=len(registered), progress=0)
        for res in registered:
            try:
                await asyncio.to_thread(search_one, registry, res, args.top_k)
            except Exception as e:  # noqa: BLE001 (a failed query must not kill the run)
                res.errors["search"] = repr(e)
                res.verdict = MISS
                self.log(f"[search] {res.func_name}: {_fmt_exc(e)}")


            tag = "?" if res.verdict is None else TAGS[res.verdict]
            rank = "-" if res.rank is None else res.rank
            self.log(
                f"[found] {res.func_name} [{res.class_name}]: "
                f"{'FOUND' if res.found else 'NOT FOUND'} "
                f"(verdict={tag}, rank={rank})")
            self.refresh_result(res)
            bar_b.advance(1)
            self.update_tally()

        write_csv(self.results, args.out)
        self.final_summary = build_summary(self.results, registry, args.top_k)
        self.set_final_cell(render_summary_cell(self.final_summary))

        chance = f"{100.0 * registry.chance:.1f}%" if registry.entries else "?"
        self.set_status(
            f"[green]Done.[/] {len(registry)} PEs indexed "
            f"(chance {chance}).  Results → {args.out}"
            + (f"  Registry → {args.registry_out}" if args.registry_out else "")
        )
        if args.registry_out:
            registry.dump(args.registry_out)


def run(rows, args) -> None:
    app = PEEvalApp(rows, args)
    app.run()

    if app.final_summary is not None:
        console = Console()
        console.print()
        console.print(render_summary_table(app.final_summary))
        console.print(f"\nResults written to [bold]{args.out}[/]")
        if args.registry_out:
            console.print(f"Registry written to [bold]{args.registry_out}[/]")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", help="Path to python_all.jsonl (source functions)")
    ap.add_argument("--out", default="pe_eval_results.csv", help="Output CSV")
    ap.add_argument("--registry-out", default=None,
                    help="Also dump the local registry (name, description, code) as JSON, "
                         "so a bad verdict can be inspected instead of guessed at.")
    ap.add_argument("--count", type=int, default=None,
                    help="Evaluate until THIS MANY Processing Elements are registered. NOTE: "
                         "this also shrinks the registry, so each PE has fewer rivals to beat. "
                         "Finding your PE among 20 is not a test.")
    ap.add_argument("--random", action="store_true", help="Shuffle before selecting")
    ap.add_argument("--seed", type=int, default=0, help="Seed for --random")
    ap.add_argument("--provider", default="openai", help="Description provider")
    ap.add_argument("--judge-model", default="gpt-5.4-mini",
                    help="LLM used by Faithfulness")
    ap.add_argument("--min-reference-words", type=int, default=8,
                    help="Docstrings with fewer prose words than this (ignoring :param: "
                         "blocks) are excluded -- they make useless search queries.")
    ap.add_argument("--min-ref-faithfulness", type=float, default=0.5,
                    help="Docstrings whose own claims are grounded in their source below "
                         "this are excluded.")
    ap.add_argument("--score-io", action="store_true",
                    help="Fold declared inputs/outputs into the description that is judged "
                         "and indexed. Under the prose schema those arrays are the only "
                         "place the I/O types appear, so without this a hallucinated type "
                         "is invisible to generated_faithfulness.")
    ap.add_argument("--index", default="description",
                    choices=["description", "code", "both"],
                    help="What the local registry indexes. 'description' is the thing "
                         "under test. 'code' is the BASELINE: run it and compare -- if the "
                         "source alone finds the PEs just as well, the description is "
                         "adding nothing. 'both' concatenates them.")
    ap.add_argument("--rank-model", default="all-mpnet-base-v2",
                    help="sentence-transformers model backing the registry search. Local; "
                         "no API calls. all-MiniLM-L6-v2 is ~5x faster, a little worse.")
    ap.add_argument("--top-k", type=int, default=5,
                    help="A PE found below this rank counts as a MISS -- nobody scrolls. "
                         "Rank 1 is OK, ranks 2..k are PARTIAL.")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY in your environment first.")

    console = Console()
    rows = load_source_rows(args.jsonl, warn=lambda m: console.print(m, style="yellow"))
    if not rows:
        sys.exit("No valid rows in the source file.")

    rows = select_rows(rows, shuffle=args.random, seed=args.seed, warn=lambda m: console.print(m, style="dim"))

    run(rows, args)


if __name__ == "__main__":
    main()
import csv
import os
from dataclasses import dataclass
from io import StringIO

from laminar.clitools.advanced_search import AdvancedSearchCommand
from pwinput import pwinput

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


@dataclass
class Example:
    user_input: str
    label: str
    match_type: str
    target: str
    matches: list  # list[tuple[str, str | None]]:
    #   (workflow, pe_name) for PEs
    #   (workflow, None)    for workflows
    #   []                  for no-match


# label prefix -> (match_type, target)
_PREFIXES = {
    "PE_MATCH": ("single", "pe"),
    "WORKFLOW_MATCH": ("single", "workflow"),
    "MULTI_PE_MATCH": ("multi", "pe"),
    "MULTI_WORKFLOW_MATCH": ("multi", "workflow"),
    "NO_MATCH_PE": ("none", "pe"),
    "NO_MATCH_WORKFLOW": ("none", "workflow"),
}


def _parse_token(token: str, target: str) -> tuple[str, str | None]:
    token = token.strip()
    if target == "pe":
        workflow, pe_name = token.rsplit(".", 1)  # "workflow.PE"
        return workflow, pe_name
    return token, None  # workflow target: bare name


def parse_label(label: str):
    prefix, _, payload = label.partition(":")  # NO_MATCH_* has no colon
    prefix, payload = prefix.strip(), payload.strip()

    try:
        match_type, target = _PREFIXES[prefix]
    except KeyError:
        raise ValueError(f"Unknown label prefix {prefix!r} in {label!r}")

    if match_type == "none":
        return match_type, target, []
    tokens = payload.split("|") if match_type == "multi" else [payload]
    return match_type, target, [_parse_token(t, target) for t in tokens]


def parse_row(user_input: str, label: str) -> Example:
    match_type, target, matches = parse_label(label)
    return Example(user_input.strip(), label.strip(), match_type, target, matches)


def load(source) -> list[Example]:
    if isinstance(source, str):
        source = open(source, newline="", encoding="utf-8")
    with source as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        return [parse_row(inp, label) for inp, label in reader if inp or label]


def result_to_match(result: dict, target: str) -> tuple[str, str | None]:
    """Map one advanced_search() result dict onto the (workflow, pe_name)
    shape used in Example.matches.

    >>> ADAPT THESE KEYS to whatever advanced_search actually returns. <
    """
    workflow = result.get("workflow") or result.get("workflow_name")
    if target == "pe":
        pe_name = result.get("pe_name") or result.get("pe") or result.get("name")
        return workflow, pe_name
    return workflow, None  # workflow target: drop the PE part


FOUND = 0
PARTIAL = 1
NO_MATCH = 2


def _expected_names(ex: Example) -> list[str]:
    """Names to compare the returned object against:
    the PE name for PE queries, the workflow name for workflow queries."""
    if ex.target == "pe":
        return [pe for (_wf, pe) in ex.matches]
    return [wf for (wf, _pe) in ex.matches]


def classify(ex: Example, results: list[dict]) -> int:
    names = [r["name"] for r in results]   # already score-sorted

    if ex.target == "pe":
        expected = [pe for (_wf, pe) in ex.matches]
        if not expected:                       # NO_MATCH_PE
            return FOUND if not names else NO_MATCH
        if not names:
            return NO_MATCH
        if names[0] in expected:               # top result is expected
            return FOUND
        if any(n in expected for n in names[1:]):
            return PARTIAL
        return NO_MATCH

    # ex.target == "workflow": score on presence vs. expectation
    match_expected = bool(ex.matches)
    has_results = bool(names)
    return FOUND if match_expected == has_results else NO_MATCH


def check_example(searchCommandInt: AdvancedSearchCommand, ex: Example):
    """Run one example and return (verdict, top_names) without printing,
    so the live dashboard owns all rendering."""
    results = searchCommandInt._search(query=ex.user_input, silent=True) or []
    verdict = classify(ex, results)
    names = [r["name"] for r in results][:3]
    return verdict, names


# --------------------------------------------------------------------------- #
# Live dashboard
# --------------------------------------------------------------------------- #

_TAGS = ("OK", "PARTIAL", "MISS")
_TAG_STYLE = {"OK": "bold green", "PARTIAL": "bold yellow", "MISS": "bold red"}


def _fmt_expected(matches: list) -> str:
    if not matches:
        return "(none)"
    return ", ".join(f"{wf}.{pe}" if pe else str(wf) for wf, pe in matches)


def _fmt_names(names: list) -> str:
    return ", ".join(names) if names else "(none)"


class LiveStats:
    """Holds the running tally and renders the dashboard each tick."""

    def __init__(self, total: int):
        self.total = total
        self.found = 0
        self.partial = 0
        self.miss = 0
        self.recent: list = []  # newest first, capped for display
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Running search tests"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            expand=True,
        )
        self.task = self.progress.add_task("tests", total=total)

    @property
    def done(self) -> int:
        return self.found + self.partial + self.miss

    def record(self, verdict: int, ex: Example, names: list) -> None:
        if verdict == FOUND:
            self.found += 1
        elif verdict == PARTIAL:
            self.partial += 1
        else:
            self.miss += 1
        tag = _TAGS[verdict]
        self.recent.insert(0, (tag, ex, names))
        del self.recent[12:]
        self.progress.advance(self.task)

    def _summary_panel(self) -> Panel:
        done = self.done or 1  # avoid div-by-zero before first result
        # strict accuracy = exact hits; weighted gives partials half credit
        accuracy = 100.0 * self.found / done
        weighted = 100.0 * (self.found + 0.5 * self.partial) / done

        grid = Table.grid(expand=True, padding=(0, 2))
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        grid.add_column(justify="right")

        grid.add_row(
            Text("OK", style="bold green"),
            Text(str(self.found), style="bold green"),
            Text(f"{100.0 * self.found / done:5.1f}%", style="green"),
        )
        grid.add_row(
            Text("PARTIAL", style="bold yellow"),
            Text(str(self.partial), style="bold yellow"),
            Text(f"{100.0 * self.partial / done:5.1f}%", style="yellow"),
        )
        grid.add_row(
            Text("MISS", style="bold red"),
            Text(str(self.miss), style="bold red"),
            Text(f"{100.0 * self.miss / done:5.1f}%", style="red"),
        )
        grid.add_row(Text(""), Text(""), Text(""))
        grid.add_row(
            Text("Accuracy", style="bold"),
            Text(f"{accuracy:5.1f}%", style="bold cyan"),
            Text(f"weighted {weighted:4.1f}%", style="cyan"),
        )

        return Panel(grid, title="Stats", border_style="blue", padding=(1, 2))

    def _recent_panel(self) -> Panel:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(width=8)             # status
        table.add_column(width=9)             # target
        table.add_column(ratio=1, overflow="fold")  # got
        table.add_column(ratio=1, overflow="fold")  # expected

        table.add_row(
            Text("STATUS", style="dim"),
            Text("TARGET", style="dim"),
            Text("GOT", style="dim"),
            Text("EXPECTED", style="dim"),
        )
        for tag, ex, names in self.recent:
            table.add_row(
                Text(tag, style=_TAG_STYLE[tag]),
                Text(ex.target, style="white"),
                Text(_fmt_names(names), style="white"),
                Text(_fmt_expected(ex.matches), style="dim"),
            )

        return Panel(table, title="Recent results", border_style="grey37", padding=(1, 2))

    def render(self) -> Group:
        return Group(self._summary_panel(), self._recent_panel(), self.progress)

    def final_summary(self) -> Panel:
        done = self.done or 1
        accuracy = 100.0 * self.found / done
        weighted = 100.0 * (self.found + 0.5 * self.partial) / done
        body = Text.assemble(
            ("Examples run : ", "bold"), (f"{self.done}\n", "white"),
            ("Correct (OK) : ", "bold green"), (f"{self.found}\n", "green"),
            ("Partial      : ", "bold yellow"), (f"{self.partial}\n", "yellow"),
            ("Miss         : ", "bold red"), (f"{self.miss}\n", "red"),
            ("\n", ""),
            ("Accuracy     : ", "bold"), (f"{accuracy:.1f}%", "bold cyan"),
            ("   weighted ", "dim"), (f"{weighted:.1f}%", "cyan"),
        )
        return Panel(body, title="Final", border_style="green", padding=(1, 2))


def run_tests(examples: list[Example]):
    from laminar.client.d4pyclient import d4pClient

    console = Console()

    client = d4pClient()
    username = os.environ.get("LAMINAR_USERNAME") or input("Username: ")
    password = os.environ.get("LAMINAR_PASSWORD") or pwinput("Password: ")
    client.login(username, password)

    searchCommandInterface = AdvancedSearchCommand(client)

    stats = LiveStats(len(examples))
    with Live(stats.render(), console=console, refresh_per_second=12) as live:
        for ex in examples:
            verdict, names = check_example(searchCommandInterface, ex)
            stats.record(verdict, ex, names)
            live.update(stats.render())

    console.print(stats.final_summary())


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
    run_tests(load(path))
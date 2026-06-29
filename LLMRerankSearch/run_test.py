import argparse
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


# --------------------------------------------------------------------------- #
# Results export + plot
# --------------------------------------------------------------------------- #

# semantic colours, matching the OK / PARTIAL / MISS scheme of the dashboard
OK_COLOR = "#2f9e44"       # Correct - green
PARTIAL_COLOR = "#f08c00"  # Partial - amber
MISS_COLOR = "#e03131"     # Miss    - red

# CSV column order is fixed: type,count with these row labels
_CSV_FIELDS = ("Correct", "Partial", "Miss")


def save_results_csv(found: int, partial: int, miss: int, path: str = "results.csv") -> None:
    """Write the tally as `type,count` (rows: Correct, Partial, Miss)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "count"])
        writer.writerow(["Correct", found])
        writer.writerow(["Partial", partial])
        writer.writerow(["Miss", miss])


def load_results_csv(path: str = "results.csv") -> tuple[int, int, int]:
    """Read a `type,count` results CSV back into (found, partial, miss)."""
    counts = {"Correct": 0, "Partial": 0, "Miss": 0}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "type" not in reader.fieldnames or "count" not in reader.fieldnames:
            raise ValueError(f"{path!r} must have a 'type,count' header")
        for row in reader:
            key = (row["type"] or "").strip()
            if key in counts:
                counts[key] = int(row["count"])
    return counts["Correct"], counts["Partial"], counts["Miss"]


def _draw_brace(ax, x0, x1, y, text, *, color="#343a40", lw=1.6,
                       depth=0.22, text_gap=0.12, fontsize=12):
    if x1 <= x0:
        return

    # Define the 4 points of the square bracket
    x_coords = [x0, x0, x1, x1]
    y_coords = [y, y - depth, y - depth, y]

    # Draw the bracket (note the corrected solid_joinstyle)
    ax.plot(x_coords, y_coords, color=color, lw=lw, clip_on=False,
            solid_capstyle="round", solid_joinstyle="miter")

    # Add the centered text
    ax.text((x0 + x1) / 2.0, y - depth - text_gap, text,
            ha="center", va="top", fontsize=fontsize, color=color, fontweight="bold")


def plot_results(found: int, partial: int, miss: int, path: str = "results.png",
                 *, title: str | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot "
              "(install with: pip install matplotlib)")
        return

    total = found + partial + miss
    denom = total or 1

    fig, ax = plt.subplots(figsize=(9.5, 2.6))

    segments = [
        ("Perfect match", found, OK_COLOR),
        ("Partial match", partial, PARTIAL_COLOR),
        ("Miss", miss, MISS_COLOR),
    ]

    bar_y, bar_h = 0.0, 0.6
    left = 0
    for _label, count, color in segments:
        if count <= 0:
            continue
        ax.barh(bar_y, count, left=left, height=bar_h, color=color,
                edgecolor="white", linewidth=1.5, zorder=3)
        pct = 100.0 * count / denom
        # only label inside the segment if there's room; otherwise skip
        if count / denom >= 0.04:
            ax.text(left + count / 2.0, bar_y, f"{_label}: {count}\n({pct:.0f}%)",
                    ha="center", va="center", color="white",
                    fontsize=12, fontweight="bold", zorder=4)
        left += count

    # brace under Correct + Partial, labelled with their sum
    good = found + partial
    if good > 0:
        _draw_brace(ax, 0, good, bar_y - bar_h / 2 - 0.06,
                    f"Result found: {good} \n({100.0 * good / denom:.0f}%)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in segments]


    ax.set_xlim(0, denom)
    ax.set_ylim(-1.15, bar_h / 2 + 0.1)
    ax.set_yticks([])
    ax.set_xlabel("Test cases", fontsize=12)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    plt.title("Laminar 3.0 advanced search accuracy", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {path}")


def run_tests(examples: list[Example], *, csv_path: str = "results.csv",
              plot_path: str | None = "code_search_accuracy.pdf"):
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

    # persist + visualise
    save_results_csv(stats.found, stats.partial, stats.miss, csv_path)
    console.print(f"Saved results -> {csv_path}")
    if plot_path:
        plot_results(stats.found, stats.partial, stats.miss, plot_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run the advanced-search eval, export a type,count CSV "
                    "and a stacked-bar plot of Correct / Partial / Miss."
    )
    parser.add_argument("dataset", nargs="?", default="dataset.csv",
                        help="input examples CSV (default: dataset.csv)")
    parser.add_argument("--csv", default="results.csv",
                        help="path for the type,count results CSV (default: results.csv)")
    parser.add_argument("--plot", default="code_search_accuracy.pdf",
                        help="path for the plot; format from extension, "
                             ".png/.pdf/.svg (default: code_search_accuracy.pdf)")
    parser.add_argument("--plot-only", action="store_true",
                        help="skip the evaluation; read --csv and just render the plot")
    parser.add_argument("--no-plot", action="store_true",
                        help="run the eval and write the CSV, but don't render a plot")
    args = parser.parse_args()

    if args.plot_only:
        found, partial, miss = load_results_csv(args.csv)
        plot_results(found, partial, miss, args.plot)
        return

    examples = load(args.dataset)
    run_tests(examples, csv_path=args.csv,
              plot_path=None if args.no_plot else args.plot)


if __name__ == "__main__":
    main()
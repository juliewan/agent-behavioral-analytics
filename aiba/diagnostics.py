"""
Graphical detection diagnostics (companion to report.py's text summary).

Renders one shareable dashboard PNG from the same findings the text report
prints:

  1. headline tiles:      injected attacks, scenarios caught by >=1 detector,
                          total findings, total false positives
  2. precision / recall:  grouped bars per detector (matches report.py's
                          per-detector eval, i.e. evaluate() over the event
                          window, so the numbers line up with the console)
  3. detector heatmap:    which detector fired on which injected attack, and
                          which stayed silent. A silent (missed) cell is the
                          point of the panel, so it reads as neutral gray,
                          while a hit takes a blue step scaled by how many
                          findings it raised. A slim red strip on the right
                          carries each detector's false-positive count.

Pure add-on: matplotlib is an optional [viz] extra. The rule-based pipeline and
the text report complete without it (run_all.py degrades gracefully), mirroring
how the [embed] extra is optional for goal_drift / card_drift.

Colors are the validated data-viz reference palette (categorical blue+aqua for
the two P/R series, a single-hue blue sequential ramp for the heatmap, and the
reserved status green/red for caught / false-positive). Every sub-3:1 fill is
backed by a direct value label so identity never rests on color alone.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from aiba.detectors.base import Finding, evaluate, is_attack
from aiba.schema import AgentCard, Event

# --- validated reference palette (light surface) ---------------------------
SURFACE = "#fcfcfb"      # chart surface
SILENT = "#ecebe6"       # a detector stayed silent on this attack (neutral)
INK = "#0b0b0b"          # primary text
INK2 = "#52514e"         # secondary text
MUTED = "#898781"        # axis / tick labels
GRID = "#e1e0d9"         # hairline gridline
SER_RECALL = "#2a78d6"   # categorical slot 1 (blue)
SER_PREC = "#1baf7a"     # categorical slot 2 (aqua)
GOOD = "#0ca30c"         # status: caught
CRIT = "#d03b3b"         # status: false positive
SEQ_BLUE = ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]  # 100->700 ramp


def compute(events: list[Event],
            all_findings: dict[str, list[Finding]],
            cards: Optional[list[AgentCard]] = None) -> dict:
    """Structured diagnostics from the same findings report.py consumes.

    per_detector uses evaluate() (event-window recall) so the bars match the
    console. The heatmap universe additionally folds in agent-card attack
    scenarios (typosquat / capability_escalation / endpoint_hijack live on
    cards, not events) so card_drift's wins are visible.
    """
    per_detector = {name: evaluate(fs, events) for name, fs in all_findings.items()}

    scenarios = {e.injected_scenario for e in events if is_attack(e.injected_scenario)}
    if cards:
        scenarios |= {c.injected for c in cards if is_attack(c.injected)}
    scenarios |= {f.ground_truth for fs in all_findings.values()
                  for f in fs if is_attack(f.ground_truth)}

    counts = {name: Counter(f.ground_truth for f in fs if is_attack(f.ground_truth))
              for name, fs in all_findings.items()}
    false_pos = {name: sum(1 for f in fs if not is_attack(f.ground_truth))
                 for name, fs in all_findings.items()}

    # detectors best-recall-first; scenarios most-caught-first (nice gradient)
    detectors = sorted(all_findings, key=lambda n: (-per_detector[n]["recall"], n))
    scen_hits = {s: sum(1 for n in detectors if counts[n][s]) for s in scenarios}
    scen_order = sorted(scenarios, key=lambda s: (-scen_hits[s], s))

    event_attacks = {e.injected_scenario for e in events if is_attack(e.injected_scenario)}
    caught_any = {s for s in event_attacks if any(counts[n][s] for n in detectors)}

    return {
        "detectors": detectors,
        "scenarios": scen_order,
        "counts": counts,
        "false_pos": false_pos,
        "per_detector": per_detector,
        "matrix": [[counts[n][s] for s in scen_order] for n in detectors],
        "headline": {
            "injected": len(event_attacks),
            "caught_any": len(caught_any),
            "findings": sum(len(fs) for fs in all_findings.values()),
            "false_pos": sum(false_pos.values()),
        },
    }


def _pretty(name: str) -> str:
    return name.replace("_", " ")


# --- terminal renderer (dev loop; stdlib only, no matplotlib) --------------

def _supports_color() -> bool:
    import os
    import sys
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    return sys.stdout.isatty()


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def render_terminal(diag: dict, color: Optional[bool] = None) -> str:
    """ANSI dashboard for the dev loop: P/R bars, a caught-vs-silent heatmap,
    and a false-positive column. 24-bit color when the terminal supports it
    (honors NO_COLOR / FORCE_COLOR), plain glyphs otherwise."""
    use_color = _supports_color() if color is None else color

    def fg(hx, s):
        if not use_color:
            return s
        r, g, b = _hex(hx)
        return f"\x1b[38;2;{r};{g};{b}m{s}\x1b[0m"

    def cell_bg(hx, s):
        if not use_color:
            return s
        r, g, b = _hex(hx)
        return f"\x1b[48;2;{r};{g};{b}m{s}\x1b[0m"

    dets, scens = diag["detectors"], diag["scenarios"]
    counts, fp, pr = diag["counts"], diag["false_pos"], diag["per_detector"]
    h = diag["headline"]
    out: list[str] = []

    rule = fg(MUTED, "─" * 66)
    out.append("")
    out.append(fg(INK, "━━ detection diagnostics ") + fg(MUTED, "━" * 42))
    fps = fg(CRIT if h["false_pos"] else GOOD, str(h["false_pos"]))
    out.append("  " + fg(SER_RECALL, f"{h['caught_any']}/{h['injected']}") + fg(MUTED, " attacks caught   ")
               + fg(INK, str(h["findings"])) + fg(MUTED, " findings   ")
               + fps + fg(MUTED, " false positives"))

    # precision / recall bars
    out.append("")
    out.append(fg(INK2, "precision / recall per detector") + fg(MUTED, "   (recall ")
               + fg(SER_RECALL, "█") + fg(MUTED, "  precision ") + fg(SER_PREC, "█") + fg(MUTED, ")"))
    W = 22
    dname = max(len(_pretty(d)) for d in dets)

    def bar(frac, hx):
        blocks = " ▏▎▍▌▋▊▉"
        full = frac * W
        n = int(full)
        rem = full - n
        s = "█" * n + (blocks[int(rem * 8)] if n < W and rem > 0 else "")
        return fg(hx, s.ljust(W, " ")) if use_color else s.ljust(W, "·")

    for d in dets:
        r, p = pr[d]["recall"], pr[d]["precision"]
        out.append(f"  {_pretty(d):<{dname}}  R {bar(r, SER_RECALL)} {r:.2f}")
        out.append(f"  {'':<{dname}}  P {bar(p, SER_PREC)} {p:.2f}")

    # heatmap: detectors × scenarios (numbered columns + legend beneath)
    out.append("")
    out.append(fg(INK2, "which detector caught which injected attack")
               + fg(MUTED, "   (shade = # findings · gray = silent)"))
    vmax = max((counts[d][s] for d in dets for s in scens), default=1) or 1
    idx = "".join(f"{i + 1:>3}" for i in range(len(scens)))
    out.append(f"  {'':<{dname}}  {idx}   {fg(MUTED, 'FP')}")
    for d in dets:
        row = []
        for s in scens:
            v = counts[d][s]
            if not v:
                row.append(cell_bg(SILENT, " · ") if use_color else " · ")
            else:
                shade = SEQ_BLUE[min(int((v - 1) / max(vmax - 1, 1) * (len(SEQ_BLUE) - 1)), len(SEQ_BLUE) - 1)]
                txt = f" {v} " if v < 10 else f"{v:>2} "
                row.append(cell_bg(shade, fg("#ffffff", txt)) if use_color else f" {v} ")
        f = fp[d]
        fptxt = fg(CRIT, f"{f:>2}") if f else fg(GOOD, " 0")
        out.append(f"  {_pretty(d):<{dname}}  " + "".join(row) + f"   {fptxt}")

    out.append("")
    out.append(fg(MUTED, "  columns:"))
    for i, s in enumerate(scens, 1):
        out.append(fg(MUTED, f"    {i:>2}. ") + fg(INK2, _pretty(s)))
    return "\n".join(out)


def print_terminal(events, all_findings, cards=None) -> None:
    print(render_terminal(compute(events, all_findings, cards)))


def render(diag: dict, out_path: str, title: str = "Agent Behavioral Analytics") -> str:
    """Draw the dashboard PNG. Returns out_path."""
    import matplotlib
    matplotlib.use("Agg")  # headless, deterministic
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "svg.fonttype": "none",
    })

    dets, scens = diag["detectors"], diag["scenarios"]
    M = np.array(diag["matrix"], dtype=float)
    nd, ns = len(dets), len(scens)

    fig = plt.figure(figsize=(max(11.0, 2.2 + 0.62 * ns), 4.2 + 0.52 * nd))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.05, 0.16 + 0.34 * nd, 0.5 + 0.42 * nd],
        hspace=0.42, left=0.16, right=0.97, top=0.9, bottom=0.14)

    fig.suptitle(f"{title} — detection diagnostics", x=0.16, ha="left",
                 fontsize=17, fontweight="bold", color=INK)
    fig.text(0.16, 0.925, "how each detector scored, and which injected attacks it caught vs. slept on",
             ha="left", fontsize=10.5, color=INK2)

    _tiles(fig, gs[0], diag["headline"], FancyBboxPatch)
    _pr_bars(fig.add_subplot(gs[1]), diag, np)
    _heatmap(fig, gs[2], diag, M, nd, ns, np, LinearSegmentedColormap)

    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def _tiles(fig, cell, h, FancyBboxPatch) -> None:
    ax = fig.add_subplot(cell)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    tiles = [
        (f"{h['caught_any']}/{h['injected']}", "attacks caught", "by ≥1 detector", SER_RECALL),
        (str(h["findings"]), "total findings", "raised across suite", INK),
        (str(h["false_pos"]), "false positives", "on benign / controls", CRIT if h["false_pos"] else GOOD),
        (str(h["injected"]), "injected attacks", "in the eval window", INK2),
    ]
    w, gap = 0.235, 0.02
    for i, (num, label, sub, accent) in enumerate(tiles):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch(
            (x, 0.05), w, 0.9, boxstyle="round,pad=0.006,rounding_size=0.02",
            linewidth=1, edgecolor=GRID, facecolor="#ffffff",
            mutation_aspect=0.35, transform=ax.transData, clip_on=False))
        ax.add_patch(FancyBboxPatch(
            (x, 0.05), 0.012, 0.9, boxstyle="square,pad=0",
            linewidth=0, facecolor=accent, transform=ax.transData, clip_on=False))
        ax.text(x + 0.03, 0.66, num, fontsize=24, fontweight="bold", color=accent, va="center")
        ax.text(x + 0.032, 0.31, label, fontsize=10.5, color=INK, va="center", fontweight="medium")
        ax.text(x + 0.032, 0.16, sub, fontsize=8.5, color=MUTED, va="center")


def _pr_bars(ax, diag, np) -> None:
    dets = diag["detectors"]
    pr = diag["per_detector"]
    y = np.arange(len(dets))
    recall = [pr[d]["recall"] for d in dets]
    prec = [pr[d]["precision"] for d in dets]
    hh = 0.36

    ax.barh(y + hh / 2 + 0.02, recall, height=hh, color=SER_RECALL, label="recall", zorder=3)
    ax.barh(y - hh / 2 - 0.02, prec, height=hh, color=SER_PREC, label="precision", zorder=3)

    for yi, v in zip(y + hh / 2 + 0.02, recall):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", ha="left", fontsize=8.5, color=INK2)
    for yi, v in zip(y - hh / 2 - 0.02, prec):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", ha="left", fontsize=8.5, color=INK2)

    ax.set_yticks(y)
    ax.set_yticklabels([_pretty(d) for d in dets], fontsize=10, color=INK)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"], fontsize=8.5)
    ax.invert_yaxis()  # best recall on top
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_title("precision & recall per detector", fontsize=12, fontweight="bold",
                 color=INK, loc="left", pad=8)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False,
              fontsize=9, ncol=2, handlelength=1.1, borderaxespad=0.2)


def _heatmap(fig, cell, diag, M, nd, ns, np, LinearSegmentedColormap) -> None:
    dets, scens = diag["detectors"], diag["scenarios"]
    fp = diag["false_pos"]

    sub = cell.subgridspec(1, 2, width_ratios=[ns, 2.2], wspace=0.04)
    ax = fig.add_subplot(sub[0])
    axf = fig.add_subplot(sub[1], sharey=ax)

    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    Mm = np.ma.masked_where(M == 0, M)
    vmax = max(M.max(), 1)
    ax.set_facecolor(SILENT)  # silent cells show through the mask
    mesh = ax.pcolormesh(Mm, cmap=cmap, vmin=1, vmax=vmax,
                         edgecolors=SURFACE, linewidth=2.5)

    for r in range(nd):
        for c in range(ns):
            v = M[r, c]
            if v:
                frac = (v - 1) / (vmax - 1) if vmax > 1 else 1.0
                ax.text(c + 0.5, r + 0.5, f"{int(v)}", ha="center", va="center",
                        fontsize=8.5, color="#ffffff" if frac > 0.45 else INK)

    ax.set_xticks(np.arange(ns) + 0.5)
    ax.set_xticklabels([_pretty(s) for s in scens], rotation=40, ha="right",
                       fontsize=8.5, color=INK2)
    ax.set_yticks(np.arange(nd) + 0.5)
    ax.set_yticklabels([_pretty(d) for d in dets], fontsize=10, color=INK)
    ax.set_ylim(nd, 0)  # first detector on top
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("which detector caught which injected attack",
                 fontsize=12, fontweight="bold", color=INK, loc="left", pad=8)

    # false-positive strip
    fp_vals = [fp[d] for d in dets]
    y = np.arange(nd) + 0.5
    axf.barh(y, fp_vals, height=0.7,
             color=[CRIT if v else GOOD for v in fp_vals], zorder=3)
    for yi, v in zip(y, fp_vals):
        axf.text(v + max(fp_vals + [1]) * 0.04, yi, str(v), va="center",
                 ha="left", fontsize=8.5, color=INK2)
    axf.set_xlim(0, max(fp_vals + [1]) * 1.35)
    axf.set_ylim(nd, 0)
    axf.set_xticks([])
    axf.tick_params(left=False, labelleft=False, length=0)
    for s in axf.spines.values():
        s.set_visible(False)
    axf.set_title("false\npositives", fontsize=9, color=INK2, loc="left", pad=8)


def write_report(events: list[Event],
                 all_findings: dict[str, list[Finding]],
                 out_path: str,
                 cards: Optional[list[AgentCard]] = None,
                 title: str = "Agent Behavioral Analytics") -> str:
    """compute -> render in one call (the entry point run_all.py uses)."""
    return render(compute(events, all_findings, cards), out_path, title)

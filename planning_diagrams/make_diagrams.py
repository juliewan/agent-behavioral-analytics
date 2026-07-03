"""Render README diagrams for agent-behavioral-analytics as PNGs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# palette — muted, legible on GitHub light & dark (white canvas)
INK      = "#1f2933"   # main text
SUB      = "#52606d"   # secondary text
STAGE_F  = "#eef2f6"   # stage box fill
STAGE_E  = "#9aa5b1"   # stage box edge
DET_F    = "#e3ecf7"   # detector bank fill
DET_E    = "#5b7aa6"
FLAG_F   = "#fdeaea"   # detector flag chip
FLAG_E   = "#c0392b"
FLAG_T   = "#8f2318"
GT_F     = "#e7f2ea"   # ground-truth / eval green
GT_E     = "#4e8a5f"
ARROW    = "#7b8794"
EMB      = "#8a6d3b"   # [embed] marker

plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, fill, edge, r=0.12, lw=1.6):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fill, ec=edge, lw=lw, zorder=2)
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=ARROW, lw=2.2, style="-|>", ms=16):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=ms, color=color, lw=lw, zorder=1,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def txt(ax, x, y, s, size=11, color=INK, weight="normal", ha="center",
        va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=3, **kw)


# ---------------------------------------------------------------- diagram 1
def pipeline():
    fig, ax = plt.subplots(figsize=(14.5, 5.6), dpi=200)
    ax.set_xlim(0, 145); ax.set_ylim(0, 56); ax.axis("off")
    fig.patch.set_facecolor("white")

    # sources (stacked, left)
    box(ax, 2, 31, 22, 12, STAGE_F, STAGE_E)
    txt(ax, 13, 39.4, "synth/traffic.py", 11.5, weight="bold")
    txt(ax, 13, 35.2, "benign multi-agent\ntraffic (seeded)", 9.5, SUB)

    box(ax, 2, 13, 22, 12, STAGE_F, STAGE_E)
    txt(ax, 13, 21.4, "synth/scenarios.py", 11.5, weight="bold")
    txt(ax, 13, 17.2, "attack injectors\n(ground-truth labels)", 9.5, SUB)

    # events.jsonl
    box(ax, 32, 20, 22, 16, STAGE_F, STAGE_E)
    txt(ax, 43, 31.6, "events.jsonl", 12, weight="bold")
    txt(ax, 43, 26.2, "unified event schema\n+ agent_cards.jsonl\n(OTel GenAI-style)", 9.5, SUB)

    arrow(ax, 24.4, 37, 32, 31)
    arrow(ax, 24.4, 19, 32, 25)

    # baseline
    box(ax, 62, 20, 20, 16, STAGE_F, STAGE_E)
    txt(ax, 72, 31.6, "baselines", 12, weight="bold")
    txt(ax, 72, 25.8, "clean window:\nper-agent / per-tool /\nper-edge stats", 9.5, SUB)
    arrow(ax, 54.4, 28, 62, 28)

    # detector bank
    box(ax, 90, 6, 28, 44, DET_F, DET_E)
    txt(ax, 104, 46.6, "detector bank", 12, weight="bold", color="#2c4a6e")
    dets = ["action hallucination", "loops / recursion",
            "memory", "sequence / rare edges",
            "goal drift \u2020", "tool health",
            "cost", "card drift (+rug-pull \u2020)"]
    y0 = 42.0
    for i, d in enumerate(dets):
        txt(ax, 92.2, y0 - i * 4.6, "\u2022 " + d, 10, INK, ha="left")
    txt(ax, 92.2, 3.0, "\u2020 optional [embed] extra", 8.5, EMB, ha="left")
    arrow(ax, 82.4, 28, 90, 28)

    # findings + report
    box(ax, 126, 29, 17, 14, STAGE_F, STAGE_E)
    txt(ax, 134.5, 39.2, "ranked", 11.5, weight="bold")
    txt(ax, 134.5, 36.1, "findings", 11.5, weight="bold")
    txt(ax, 134.5, 32.2, "additive risk\nscores", 9, SUB)

    box(ax, 126, 9, 17, 14, GT_F, GT_E)
    txt(ax, 134.5, 19.2, "report.py", 11.5, weight="bold")
    txt(ax, 134.5, 13.8, "correlation +\nP/R eval vs\ninjected labels", 9, SUB)

    arrow(ax, 118.4, 34, 126, 36)
    arrow(ax, 134.5, 29, 134.5, 23.4)

    fig.savefig(f"{OUT}/pipeline.png", bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)


# ------------------------------------------------------- incident chain base
def chain(fname, title, steps):
    """steps: list of (event_text, flag_text or None)"""
    n = len(steps)
    H = 10 + n * 13
    fig, ax = plt.subplots(figsize=(10.5, H / 9.5), dpi=200)
    ax.set_xlim(0, 105); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor("white")

    txt(ax, 52.5, H - 4, title, 13.5, weight="bold")

    y = H - 10
    for i, (event, flag) in enumerate(steps):
        by = y - 10
        box(ax, 4, by, 62, 10, STAGE_F, STAGE_E)
        txt(ax, 7, by + 5, f"{i+1}", 15, "#c0d0e0", weight="bold")
        txt(ax, 11, by + 5, event, 10, INK, ha="left")
        if flag:
            box(ax, 72, by + 2, 29, 6, FLAG_F, FLAG_E, r=0.3, lw=1.4)
            txt(ax, 86.5, by + 5, flag, 9.2, FLAG_T, weight="bold")
            arrow(ax, 66.4, by + 5, 71.6, by + 5, color=FLAG_E, lw=1.6, ms=11)
        if i < n - 1:
            arrow(ax, 35, by - 0.4, 35, by - 2.8, lw=2.4)
        y -= 13

    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)


chain_hijack = [
    ("Scheduler-Agent's endpoint replaced in its card\n(endpoint not in baseline set)",
     "CARD DRIFT"),
    ("Impostor endpoint is slow and flaky\n(latency + failure-rate z-score spike)",
     "TOOL HEALTH"),
    ("Downstream calls fail / return empty; agent claims\nsuccess and continues the plan",
     "ACTION HALLUCINATION"),
    ("Retries spiral: repetitive n-grams, 0% errors,\nballooning tokens, 3-minute session",
     "LOOPS + COST"),
]

chain_poison = [
    ("Agent ingests external web content; next span is\nexec_shell, an edge absent from baseline",
     "SEQUENCE (risk edge)"),
    ("Planner-step similarity to original goal cliffs\nat the same point (single-step drop)",
     "GOAL DRIFT"),
    ("Harvested credential written to all_agents,\nno-TTL memory (regex class: key/secret)",
     "MEMORY \u00b7 M3"),
    ("Agent bulk-deletes prior memories to cover\ntracks: burst chained off the M3 hit",
     "MEMORY \u00b7 M2"),
]

pipeline()
chain("incident_endpoint_hijack.png",
      "Correlated incident \u2014 endpoint hijack chain", chain_hijack)
chain("incident_memory_poisoning.png",
      "Correlated incident \u2014 memory-poisoning chain", chain_poison)
print("done:", os.listdir(OUT))

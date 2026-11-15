"""Append draft-eval leaderboard rows from scorecard JSONs (no hand-typed numbers).

    uv run python scripts/add_rows.py --exp ONB-FRA-full --config configs/ONB-FRA-full.yaml \
        --wall 81min docs/scorecards/onboard/FRA/ONB-FRA-full@best.FRA.val.full.json

Rows go at the end of docs/LEADERBOARD.md unless --section names a heading,
in which case they are appended to that section's table (created if absent).
"""
import argparse
import json
from pathlib import Path

LB = Path("docs/LEADERBOARD.md")
HEADER = ("| EXP | Model | Set(s) | Split | Stats | Top-1 | Top-3 | Expert Top-1 | Skill-gap | NLL "
          "| Config | SHA | Wall-clock | Date |\n"
          "|-----|-------|--------|-------|-------|-------|-------|--------------|-----------|-----"
          "|--------|-----|------------|------|\n")


def row(card: dict, exp: str, config: str, wall: str) -> str:
    m = card["metrics"]
    return (f"| {exp} | {card['model']} | {card['set']} | {card['split']} | {card['stats']} "
            f"| {m['top1']:.4f} | {m['top3']:.4f} | {m['expert_top1']:.4f} | {m['skill_gap']:+.4f} "
            f"| {m['nll']:.4f} | {config} | {card['git_sha']} | {wall} | {card['generated_at'][:10]} |\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cards", nargs="+", type=Path)
    p.add_argument("--exp", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--wall", required=True)
    p.add_argument("--section", default=None, help="markdown heading text (without #s)")
    a = p.parse_args()
    rows = "".join(row(json.loads(c.read_text()), a.exp, a.config, a.wall) for c in a.cards)
    text = LB.read_text().rstrip("\n") + "\n"
    if a.section and any(l.lstrip("#").strip() == a.section for l in text.splitlines()):
        lines = text.splitlines(keepends=True)
        i = next(k for k, l in enumerate(lines) if l.lstrip("#").strip() == a.section)
        j = i + 1
        while j < len(lines) and not lines[j].startswith("#"):
            j += 1
        while lines[j - 1].strip() == "":
            j -= 1
        lines.insert(j, rows)
        text = "".join(lines)
    elif a.section:
        text += f"\n## {a.section}\n\n{HEADER}{rows}"
    else:
        text += rows
    LB.write_text(text)
    print(f"{len(a.cards)} row(s) added")


if __name__ == "__main__":
    main()

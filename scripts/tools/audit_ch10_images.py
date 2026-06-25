#!/usr/bin/env python3
"""Audit Chapter 10 image placement in docs/KNOWLEDGE.md."""

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / "docs" / "KNOWLEDGE.md"


def run_num_from_prefix(fname: str):
    if m := re.match(r"run(\d+)_r(\d+)_", fname):
        return int(m.group(1)), int(m.group(2))
    if m := re.match(r"run(\d+)_", fname):
        return int(m.group(1)), None
    return None, None


def section_run_num(name: str):
    if m := re.match(r"Run (\d+) R(\d+)", name):
        return int(m.group(1)), int(m.group(2))
    if m := re.match(r"Run (\d+)", name):
        return int(m.group(1)), None
    return None, None


def classify(fname: str) -> str:
    if fname.startswith("official_h1_flat"):
        return "official_flat"
    if fname.startswith("official_h1"):
        return "official_rough"
    if fname.startswith("ch5_"):
        return "ch5"
    if fname.startswith("appendix_"):
        return "appendix"
    if fname.startswith("tensorboard_") or fname.startswith("play_my_h1"):
        return "legacy_run12"
    if fname.startswith("run"):
        return "run"
    return "other"


def matches(sec_name: str, fname: str, caption: str) -> bool:
    cat = classify(fname)
    if cat in ("ch5", "appendix"):
        return "附录" in sec_name
    if cat == "official_flat":
        return "官方 H1 Flat" in sec_name
    if cat == "official_rough":
        return "官方 H1 Rough" in sec_name
    if cat == "legacy_run12":
        return section_run_num(sec_name)[0] in (1, 2)
    if cat == "run":
        pr, ps = run_num_from_prefix(fname)
        sr, ss = section_run_num(sec_name)
        if ps is not None:
            return pr == sr and ps == ss
        if pr == sr:
            return True
        if pr and (f"Run {pr}/" in caption or f"（Run {pr}/" in caption):
            return True
        return False
    return True


def main() -> int:
    md = MD.read_text()
    start = md.find("## 第 10 章")
    if start < 0:
        print("Chapter 10 not found", file=sys.stderr)
        return 1
    lines = md[start:].splitlines()

    sections = []
    for i, line in enumerate(lines):
        if line.startswith("### "):
            sections.append({"name": line[4:].strip(), "start": i})

    images = []
    for si, sec in enumerate(sections):
        end = sections[si + 1]["start"] if si + 1 < len(sections) else len(lines)
        for li in range(sec["start"], end):
            m = re.search(r"!\[([^\]]*)\]\(\./images/([^)]+)\)", lines[li])
            if m:
                cap, fn = m.group(1), m.group(2)
                images.append(
                    {
                        "sec": sec["name"],
                        "fn": fn,
                        "cap": cap,
                        "ok": matches(sec["name"], fn, cap),
                    }
                )

    mismatches = [x for x in images if not x["ok"]]
    dups = defaultdict(list)
    for x in images:
        dups[x["fn"]].append(x["sec"])

    print(f"Total images: {len(images)}")
    print(f"Correct placement: {len(images) - len(mismatches)}")
    print(f"Mismatches: {len(mismatches)}")
    for x in mismatches:
        print(f"  [{x['sec']}] {x['fn']}")
    print("Duplicates:")
    for fn, secs in dups.items():
        if len(secs) > 1:
            print(f"  {fn}: {secs}")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())

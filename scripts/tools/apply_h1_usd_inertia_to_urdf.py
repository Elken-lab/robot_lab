#!/usr/bin/env python3
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Apply Isaac h1_minimal.usd rigid-body props to h1_minimal.urdf (Route B P2)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _parse_export(path: Path) -> dict[str, dict]:
    bodies: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, rest = line.split("\t", 1)
        mass = float(re.search(r"m=([-\d.eE+]+)", rest).group(1))
        com = [float(x) for x in re.search(r"com=([^ \t]+)", rest).group(1).split(",")]
        ixx, ixy, ixz, iyy, iyz, izz = [float(x) for x in re.search(r"I=([^ \t]+)", rest).group(1).split(",")]
        bodies[name] = {"mass": mass, "com": com, "inertia": (ixx, ixy, ixz, iyy, iyz, izz)}
    return bodies


def _fmt(v: float) -> str:
    if abs(v) < 1e-4 and v != 0:
        return f"{v:.8E}"
    return f"{v:g}"


def _inertial_xml(props: dict) -> str:
    cx, cy, cz = props["com"]
    ixx, ixy, ixz, iyy, iyz, izz = props["inertia"]
    return (
        "    <inertial>\n"
        f'      <origin xyz="{_fmt(cx)} {_fmt(cy)} {_fmt(cz)}" rpy="0 0 0"/>\n'
        f'      <mass value="{_fmt(props["mass"])}"/>\n'
        f'      <inertia ixx="{_fmt(ixx)}" ixy="{_fmt(ixy)}" ixz="{_fmt(ixz)}" '
        f'iyy="{_fmt(iyy)}" iyz="{_fmt(iyz)}" izz="{_fmt(izz)}"/>\n'
        "    </inertial>"
    )


def apply(urdf_path: Path, export_path: Path) -> list[str]:
    bodies = _parse_export(export_path)
    text = urdf_path.read_text(encoding="utf-8")
    updated: list[str] = []
    for name, props in bodies.items():
        pattern = (
            rf'(<link name="{re.escape(name)}">\s*)'
            r"<inertial>.*?</inertial>"
        )
        repl = rf"\1{_inertial_xml(props)}"
        new_text, n = re.subn(pattern, repl, text, count=1, flags=re.DOTALL)
        if n:
            text = new_text
            updated.append(name)
    urdf_path.write_text(text, encoding="utf-8")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "source/robot_lab/data/Robots/unitree/h1_description/urdf/h1_minimal.urdf",
    )
    parser.add_argument("--export", type=Path, default=Path("/tmp/h1_usd_rigid_bodies.txt"))
    args = parser.parse_args()
    updated = apply(args.urdf, args.export)
    print(f"Updated {len(updated)} links in {args.urdf}")
    for name in updated:
        print(f"  - {name}")


if __name__ == "__main__":
    main()

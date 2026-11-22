"""Check relative links and in-page anchors in the repository's Markdown files.

External (http/https) links are not fetched, so the check is deterministic
and works offline. Exit status 1 if any relative link target is missing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SKIP_DIRS = {".git", ".venv", "venv", "node_modules"}


def slug(heading: str) -> str:
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s", "-", text)


def anchors(path: Path) -> set[str]:
    found = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"#{1,6}\s+(.*)", line)
        if match:
            found.add(slug(match.group(1)))
    return found


def main() -> int:
    errors = []
    files = [
        p for p in ROOT.rglob("*.md") if not SKIP_DIRS.intersection(p.parts)
    ]
    for md in sorted(files):
        text = md.read_text(encoding="utf-8-sig")
        in_fence = False
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("```"):
                in_fence = not in_fence
            if in_fence:
                continue
            for target in LINK.findall(line):
                if re.match(r"^[a-z][a-z0-9+.-]*:", target):
                    continue  # http(s), mailto, ...
                path_part, _, fragment = target.partition("#")
                dest = md if not path_part else (md.parent / path_part).resolve()
                if not dest.exists():
                    errors.append(f"{md.relative_to(ROOT)}:{number}: missing {target}")
                elif fragment and dest.suffix == ".md" and fragment not in anchors(dest):
                    errors.append(f"{md.relative_to(ROOT)}:{number}: missing anchor #{fragment}")
    for message in errors:
        print(message)
    print(f"checked {len(files)} Markdown files, {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

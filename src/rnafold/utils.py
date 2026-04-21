from __future__ import annotations

from typing import Dict, List, Set, Tuple

RNA_ALPHABET = "ACGU"

_WATSON_CRICK = {("A", "U"), ("U", "A"), ("C", "G"), ("G", "C")}
_WOBBLE = {("G", "U"), ("U", "G")}
PAIRS: Set[Tuple[str, str]] = _WATSON_CRICK | _WOBBLE


def is_valid_rna(sequence: str) -> bool:
    return bool(sequence) and all(c in RNA_ALPHABET for c in sequence.upper())


def can_pair(b1: str, b2: str) -> bool:
    return (b1.upper(), b2.upper()) in PAIRS


def pairs_to_dotbracket(pairs: List[Tuple[int, int]], n: int) -> str:
    dot = ["."] * n
    for i, j in pairs:
        if i < j:
            dot[i], dot[j] = "(", ")"
    return "".join(dot)


def dotbracket_to_pairs(structure: str) -> List[Tuple[int, int]]:
    stack: List[int] = []
    pairs: List[Tuple[int, int]] = []
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if not stack:
                raise ValueError(f"unmatched ')' at position {i}")
            j = stack.pop()
            pairs.append((j, i))
        elif c != ".":
            raise ValueError(f"unexpected character {c!r} at position {i}")
    if stack:
        raise ValueError("unmatched '('")
    return sorted(pairs)


def is_nested(pairs: List[Tuple[int, int]]) -> bool:
    ps = sorted(pairs)
    for a, (i, j) in enumerate(ps):
        for k, l in ps[a + 1:]:
            if i < k < j < l:
                return False
    return True


def load_known_structures(path) -> List[Tuple[str, str, str]]:
    """Parse our `data/rna_known_structures.fasta` format.

    Returns a list of (name, sequence, dot-bracket_structure_or_empty).
    """
    from pathlib import Path
    text = Path(path).read_text()
    records: List[Tuple[str, str, str]] = []
    header: str = ""
    parts: List[str] = []
    all_entries: List[Tuple[str, str]] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header:
                all_entries.append((header, "".join(parts)))
            header = line[1:].strip()
            parts = []
        else:
            parts.append(line.strip())
    if header:
        all_entries.append((header, "".join(parts)))
    i = 0
    while i < len(all_entries):
        name, seq = all_entries[i]
        if i + 1 < len(all_entries) and all_entries[i + 1][0].startswith(name + "_structure") or (
            i + 1 < len(all_entries) and "structure" in all_entries[i + 1][0].lower()
            and name.split()[0] in all_entries[i + 1][0]
        ):
            records.append((name, seq.upper(), all_entries[i + 1][1]))
            i += 2
        else:
            records.append((name, seq.upper(), ""))
            i += 1
    return records

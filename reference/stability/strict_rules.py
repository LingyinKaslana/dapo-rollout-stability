"""Conservative exact-repeat triage; negative means 'needs review', not normal.

Offsets are Python Unicode character offsets into the original response. This
module deliberately does not infer semantic drift, correctness, or instability
from length, token probabilities, global n-gram rates, or non-English text.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re

VERSION = "strict-exact-repetition-v1"
DECORATION = set("-_=~*#·─━═/\\|.()[]{}+")


def _substantive(unit: str) -> bool:
    """Exclude decoration, numeric expansions, and tiny algebraic building blocks."""
    compact = re.sub(r"\s+", "", unit)
    letters = {c for c in compact if c.isalpha()}
    if len(compact) < 32 or len(letters) < 3:
        return False
    # A sum can intentionally spell out several identical summands. In that
    # case repetition alone cannot establish stalled reasoning.
    nonempty_lines = [x.strip() for x in unit.splitlines() if x.strip()]
    additive_lines = []
    for line in nonempty_lines:
        stripped = re.sub(r"\\(?:q?quad|,|;|!| )", "", line).lstrip("& ")
        additive_lines.append(stripped.startswith(("+", "-")))
    return not all(additive_lines)


def _evidence(text, kind, start, end, count, unit, **extra):
    return {
        "type": kind,
        "start": start,
        "end": end,
        "characters": end - start,
        "repeat_count": count,
        "unit": unit[:600],
        "unit_characters": len(unit),
        "snippet": text[start:min(end, start + 600)],
        "context_before": text[max(0, start - 100):start],
        "context_after": text[end:min(len(text), end + 100)],
        **extra,
    }


def classify_obvious(text: str, question: str = "") -> dict:
    """Select only sustained, exact adjacent repetition or extreme character runs.

    All numbers and formula characters are preserved. Whitespace-only changes
    are ignored for line/block comparisons. In particular, changing numerals in
    step headings are *not* stripped. Four repeated blocks are needed, spanning
    >=500 original characters and >=320 nonspace characters. Outputs containing
    only ordinary wrong math, distinct enumerated cases, or repeated decoration
    are not selected.
    """
    evidence = []
    if not isinstance(text, str):
        raise TypeError("text must be str")

    # Repeated rows are legitimate in matrices, tables, and geometric diagrams.
    # Leave their interpretation to the LLM instead of calling them a loop.
    table_spans = []
    table_stack = []
    for match in re.finditer(r"\\(begin|end)\{(array|[bBpPvV]?matrix|smallmatrix|tabular\*?)\}", text):
        if match[1] == "begin":
            table_stack.append((match[2], match.start()))
        elif table_stack and table_stack[-1][0] == match[2]:
            _, start = table_stack.pop()
            table_spans.append((start, match.end()))
    table_spans.extend((start, len(text)) for _, start in table_stack)

    def in_table(start, end):
        return any(start >= a and end <= b for a, b in table_spans)

    # A long integer or separator is insufficient. Repeated non-ASCII letters,
    # replacement characters, NULs, and @ are much less ambiguous in math output.
    for match in re.finditer(r"([^\s])\1{95,}", text):
        char = match[1]
        start, end = match.span()
        if char.isdigit() or char in DECORATION:
            continue
        threshold = 96 if (ord(char) > 127 or char in "@\x00") else (128 if char.isalpha() else 256)
        if end - start < threshold or match[0] in question:
            continue
        evidence.append(_evidence(text, "character_run", start, end, end - start,
                                  char, codepoint=f"U+{ord(char):04X}"))

    # Work on all nonempty lines. Adjacent line blocks must be identical, so
    # calculations separated by changed expressions cannot become a loop match.
    lines = []
    for match in re.finditer(r"[^\r\n]+", text):
        raw = match[0]
        norm = re.sub(r"\s+", " ", raw).strip()
        if norm:
            lines.append((norm, match.start(), match.end()))
    values = [line[0] for line in lines]
    n = len(values)
    consumed_to = 0
    for i in range(n):
        if i < consumed_to:
            continue
        for width in range(1, min(64, (n - i) // 4) + 1):
            block = values[i:i + width]
            if block != values[i + width:i + 2 * width]:
                continue
            count = 2
            while i + (count + 1) * width <= n and values[i + count * width:i + (count + 1) * width] == block:
                count += 1
            if count < 4:
                continue
            unit = "\n".join(block)
            start, end = lines[i][1], lines[i + count * width - 1][2]
            nonspace = len(re.sub(r"\s+", "", unit)) * count
            if end - start < 500 or nonspace < 320 or not _substantive(unit) or in_table(start, end):
                continue
            # A question explicitly containing the complete repeated sequence
            # may legitimately ask the model to reproduce it.
            if re.sub(r"\s+", " ", text[start:end]).strip() in re.sub(r"\s+", " ", question):
                continue
            evidence.append(_evidence(text, "exact_line_or_block_loop", start, end,
                                      count, unit, lines_per_repeat=width,
                                      normalization="whitespace only; numbers unchanged"))
            consumed_to = i + count * width
            break

    # Inline prose may have no line breaks. Consecutive identical long sentences
    # are safe to select; e.g. varying numerical cases are kept as distinct units.
    sentences = []
    sentence_start = 0
    # Delimiter scanning is linear even for a megabyte-long punctuation-free
    # line; an unanchored greedy sentence regex could become quadratic.
    for match in re.finditer(r"[.!?](?=\s|$)|\n", text):
        if match[0] != "\n":
            raw = text[sentence_start:match.end()]
            norm = re.sub(r"\s+", " ", raw).strip()
            if norm:
                sentences.append((norm, sentence_start, match.end()))
        sentence_start = match.end()
    i = 0
    while i < len(sentences):
        j = i + 1
        while j < len(sentences) and sentences[j][0] == sentences[i][0] and not text[sentences[j - 1][2]:sentences[j][1]].strip():
            j += 1
        count = j - i
        if count >= 5:
            unit, start, _ = sentences[i]
            end = sentences[j - 1][2]
            if end - start >= 500 and _substantive(unit) and not in_table(start, end):
                evidence.append(_evidence(text, "exact_sentence_loop", start, end, count, unit,
                                          normalization="whitespace only; numbers unchanged"))
        i = j

    # Keep maximal independent spans so line and sentence detectors do not
    # multiply-count the same passage. This does not change the binary decision.
    selected = []
    for item in sorted(evidence, key=lambda x: x["characters"], reverse=True):
        if not any(item["start"] >= x["start"] and item["end"] <= x["end"] for x in selected):
            selected.append(item)
    selected.sort(key=lambda x: x["start"])
    return {
        "version": VERSION,
        "obvious": bool(selected),
        "types": sorted({x["type"] for x in selected}),
        "evidence": selected,
        "interpretation": "Positive: exact repeated output selected by conservative rules. Negative: requires semantic review; not a normal-output label.",
    }


def _self_test():
    cases = [
        ("ordinary_wrong_answer", "The answer is 3 + 5 = 9. Therefore the result is 9.", False),
        ("decorative_separator", "Analysis\n" + "-" * 800 + "\nFinal answer is 4.", False),
        ("long_integer", "The integer is " + "8" * 500 + ".", False),
        ("repeated_matrix_rows", "\\begin{array}{ccccccccc}\n" + ("\\circ & \\circ & \\circ & \\circ & \\circ & \\circ & \\circ & \\circ & \\circ \\\\\n" * 20) + "\\end{array}", False),
        ("repeated_summands", "S = \\begin{align*}\n" + ("&\\quad + 4 \\int_0^1 \\frac{x^2 + y^2}{1 + z^2} \\, dx \\\\\n" * 20) + "\\end{align*}", False),
        ("distinct_numerical_cases", "\n".join(f"For case {i}, the resulting coefficient is {i + 1} and the sum is {i + 2}." for i in range(30)), False),
        ("changing_formula_corrections", "\n".join("This is incorrect. Let's use the correct method:\n" + f"f(x) = {i} x^2 + {i + 1} x + {i + 2}." for i in range(30)), False),
        ("single_symbols", "I conclude: " + "@" * 1500, True),
        ("repeated_nulls", "The answer is 4." + "\x00" * 300, True),
        ("repeated_letter", "Now: " + "廋" * 1000, True),
        ("line_loop", ("The same conclusion follows and we shall check the same expression once again.\n" * 20), True),
        ("block_loop", ("We apply the same formula again.\n\\[ f(x) = \\frac{1}{x^2 + 1} = \\frac{1}{x^2 + 1} \\]\n" * 8), True),
        ("sentence_loop", "The same conclusion follows and we shall check the same expression once again. " * 20, True),
        ("two_passes_only", "The same conclusion follows and we shall check the same expression once again.\n" * 2, False),
    ]
    out = []
    for name, text, expected in cases:
        actual = classify_obvious(text)
        assert actual["obvious"] == expected, (name, actual)
        for item in actual["evidence"]:
            assert text[item["start"]:min(item["end"], item["start"] + 600)] == item["snippet"]
        out.append({"name": name, "expected": expected, "actual": actual["obvious"]})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    result = {"version": VERSION, "self_tests": _self_test()}
    if args.audit:
        reports = Path(__file__).resolve().parent.parent
        current = json.load(gzip.open(reports / "anomaly_screen10000_20261104/candidate_results.json.gz", "rt"))["candidates"]
        old = json.load(open(reports / "anomaly_screen1000_20261104/results.json"))
        labels = {x["ordinal"]: x for x in json.load(open(reports / "anomaly_screen1000_20261104/manual_review_merged.json"))}
        old_questions = {x["ordinal"]: "\n".join(p["content"] for p in x["prompt"]) for x in old["cases"]}
        selected = []
        for row in current:
            question = "\n".join(p["content"] for p in row["prompt"])
            verdict = classify_obvious(row["text"], question)
            if verdict["obvious"]:
                selected.append({"ordinal": row["ordinal"], "qid": row["qid"], "tokens": row["tokens"], **verdict})
        prior_selected = []
        for row in old["records"]:
            verdict = classify_obvious(row["text"], old_questions[row["ordinal"]])
            if verdict["obvious"]:
                prior_selected.append({"ordinal": row["ordinal"], "manual_label": labels.get(row["ordinal"]), **verdict})
        result.update({
            "current_candidates": len(current),
            "current_obvious_count": len(selected),
            "current_type_counts_overlap": dict(Counter(t for x in selected for t in x["types"])),
            "current_selected": selected,
            "old_total": len(old["records"]),
            "old_selected": prior_selected,
            "old_selected_manual_decisions": dict(Counter((x["manual_label"] or {}).get("decision", "unreviewed") for x in prior_selected)),
            "limitations": [
                "Rules select only exact sustained repeats, not semantic drift or arbitrary gibberish.",
                "Only whitespace is normalized; variable formulas and enumerated cases are not collapsed.",
                "Numeric and decoration runs are deferred even if they might be anomalous.",
                "Prior manual labels cover a selected subset, so no population precision or recall estimate follows.",
                "The 10k pass evaluates the 1482 prefiltered candidates; it cannot prove remaining outputs normal.",
            ],
        })
        target = Path(__file__).with_name("strict_rules_audit.json")
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("current_selected", "old_selected")}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

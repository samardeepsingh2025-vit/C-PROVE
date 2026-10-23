"""Number extraction + normalisation for the numeric-inconsistency check.

Reviewer concern: a transcript may say "twenty-five" while the summary says "25" (or "25 euros" vs "twenty five euro").
Absence of the *string* must therefore never be read as fabrication. We map every numeric mention in both texts
to a canonical VALUE (digits, spelled-out numbers, decimals "twelve point five", percentages, k/thousand/million,
"half"/"a quarter" are NOT treated as numbers) and compare values.

  numeric_report(transcript_text, claim_text) -> {"claim": [...], "unseen": [...], "checked": bool}
A claim number counts as supported if its value occurs in the transcript, optionally also as the sum/difference of two
transcript values only when `allow_arith` is set (OFF by default; used in sensitivity tests).
"""
from __future__ import annotations
import re
from functools import lru_cache

_UNITS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_MULT = {"hundred": 100, "thousand": 1000, "million": 10 ** 6}
_NUMWORDS = set(_UNITS) | set(_TENS) | set(_MULT)
_UNIT_NOUNS = {"euro", "euros", "pound", "pounds", "dollar", "dollars", "quid", "bucks", "percent", "pence", "cents", "cent",
               "hour", "hours", "minute", "minutes", "day", "days", "week", "weeks", "month", "months", "year", "years", "people",
               "buttons", "button", "units", "pieces", "colours", "colors", "features", "options", "meetings", "million", "thousand", "hundred"}
_DIG = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(k|m|%|percent)?(?![\w])", re.I)


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+(?:[.,]\d+)*%?", text.lower().replace("-", " "))


def _parse_words(toks: list[str], i: int):
    """Parse a spelled-out number starting at toks[i]. Returns (value, next_index) or None."""
    n = len(toks)
    total, cur, j, seen = 0, 0, i, False
    while j < n:
        t = toks[j]
        if t == "and" and seen and j + 1 < n and toks[j + 1] in _NUMWORDS:
            j += 1; continue
        if t == "a" and not seen and j + 1 < n and toks[j + 1] in _MULT:
            cur, seen, j = 1, True, j + 1; continue
        if t in _UNITS and (cur % 100 == 0 or (cur % 100 >= 20 and cur % 10 == 0)):
            cur += _UNITS[t]; seen = True; j += 1
        elif t in _TENS and (cur % 100 == 0):
            cur += _TENS[t]; seen = True; j += 1
        elif t in _MULT and seen:
            if t == "hundred":
                cur = (cur or 1) * 100
            else:
                total += (cur or 1) * _MULT[t]; cur = 0
            j += 1
        else:
            break
    if not seen:
        return None
    val = total + cur
    # "twelve point five"  /  "point five"
    if j + 1 < n and toks[j] == "point" and toks[j + 1] in _UNITS:
        k, frac = j + 1, ""
        while k < n and toks[k] in _UNITS and _UNITS[toks[k]] < 10:
            frac += str(_UNITS[toks[k]]); k += 1
        if frac:
            return float(f"{val}.{frac}"), k
    return float(val), j


def extract_values(text: str, claim_side: bool = False) -> list[float]:
    """All numeric values mentioned in `text`. On the claim side a bare 'one' ('one of the designers') is ignored
    unless followed by a unit noun, to avoid false alarms."""
    vals: list[float] = []
    for m in _DIG.finditer(text):
        whole = m.group(1).replace(",", "")
        v = float(whole + ("." + m.group(2) if m.group(2) else ""))
        suf = (m.group(3) or "").lower()
        nxt = re.match(r"\s*(hundred|thousand|million)\b", text[m.end():].lower())
        if nxt and not suf:
            v *= _MULT[nxt.group(1)]
        if suf == "k":
            v *= 1000
        elif suf == "m":
            v *= 10 ** 6
        vals.append(v)
    toks = _tok(re.sub(r"\d+(?:[.,]\d+)*", " ", text))   # words only (digits already handled)
    i = 0
    while i < len(toks):
        if toks[i] in _NUMWORDS or (toks[i] == "a" and i + 1 < len(toks) and toks[i + 1] in _MULT):
            r = _parse_words(toks, i)
            if r:
                v, j = r
                bare_one = (j == i + 1 and toks[i] in ("one", "oh"))
                if not (claim_side and bare_one and not (j < len(toks) and toks[j] in _UNIT_NOUNS)):
                    vals.append(v)
                i = max(j, i + 1); continue
        i += 1
    return vals


def spoken_composites(text: str) -> list[float]:
    """Spoken money/decimal forms in the TRANSCRIPT: 'twelve fifty', 'twelve euros fifty', '12 euros 50' -> 12.5;
    'twelve and a half' -> 12.5. Added to the set of values the transcript supports (never to the claim side)."""
    toks = re.findall(r"\d+(?:\.\d+)?|[a-z]+", text.lower())
    seq, i = [], 0     # (value, start, end) numbers in order
    while i < len(toks):
        t = toks[i]
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            seq.append((float(t), i, i + 1)); i += 1; continue
        if t in _NUMWORDS:
            r = _parse_words(toks, i)
            if r:
                seq.append((r[0], i, max(r[1], i + 1))); i = max(r[1], i + 1); continue
        i += 1
    out = []
    for (a, _, ea), (b, sb, _) in zip(seq, seq[1:]):
        gap = toks[ea:sb]
        if a == int(a) and 1 <= a < 1000 and b == int(b) and 1 <= b <= 99 and len(gap) <= 1 and all(g in _UNIT_NOUNS for g in gap):
            out.append(a + b / 100.0)
    for (a, _, ea) in seq:
        if toks[ea:ea + 3] == ["and", "a", "half"] or toks[ea:ea + 2] == ["and", "half"]:
            out.append(a + 0.5)
    return out


@lru_cache(maxsize=512)
def _transcript_values(text: str) -> frozenset:
    return frozenset(round(v, 6) for v in extract_values(text) + spoken_composites(text))


def numeric_report(transcript_text: str, claim_text: str, allow_arith: bool = False) -> dict:
    cv = [round(v, 6) for v in extract_values(claim_text, claim_side=True)]
    tv = _transcript_values(transcript_text)
    unseen = []
    for v in cv:
        if v in tv:
            continue
        if allow_arith and any(round(a + b, 6) == v or round(abs(a - b), 6) == v for a in tv for b in tv):
            continue
        unseen.append(v)
    return {"claim": cv, "unseen": unseen, "checked": bool(cv)}

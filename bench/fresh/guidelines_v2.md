# Annotation guidelines — claim verification on meeting summaries (v2)

**Your job:** decide whether each short claim from an AI-written meeting summary is true *according to the meeting transcript*.
Use only the transcript. Your own knowledge of the topic, or what sounds reasonable, does not count.
You see the **full transcript** (right side, lines numbered `[n]`), the summary (left) and one claim at a time. Use Ctrl+F or the search box to find lines.
You do NOT see any model output. Work alone and do not discuss items with the other annotator until adjudication.

## 1. Label (choose exactly one)
| Label | Use when | Example (transcript: *"ID: I propose 35 euros." "PM: too high, let's say 25 euros." "ID: ok, 25."*) |
|---|---|---|
| **SUPPORTED** | The transcript states or clearly implies the claim, **in its final state** (not later changed). | "The team set the price at 25 euros." |
| **CONTRADICTED** | The transcript says something incompatible with the claim: other number/entity/person, opposite polarity, or the claim states the old state of a decision that was later changed. | "The price was set at 35 euros." |
| **CONFLICTING** | The transcript contains real evidence both for and against; people disagree and it is never resolved. | "The team agreed on a scroll wheel" when one person insists on it and another rejects it and no decision follows. |
| **INSUFFICIENT** | The transcript neither supports nor contradicts the claim: topic not discussed, too vague, or you cannot decide. This is NOT the same as false. | "The price includes shipping." (never mentioned) |

Rules of thumb
- **Final state wins.** If something is proposed and later rejected/revised, "decided X" for the old X is CONTRADICTED.
- **Proposed ≠ decided.** "The team decided on X" when X was only suggested and not accepted is CONTRADICTED (status error). If it was suggested and nobody objected but nobody confirmed, choose INSUFFICIENT.
- **Who matters.** If the claim names a person/role, the speaker must match. Wrong person = CONTRADICTED (speaker swap). Roles in AMI: Project Manager, Marketing, Industrial Designer, User Interface.
- **Numbers are exact.** Different number, unit or currency than the transcript = CONTRADICTED. "twenty-five" and "25" are the same. Rounding/approximation that the speakers themselves make is fine; one you make up is not.
- **Paraphrase is fine.** Do not penalise different words with the same meaning.
- **Jokes / abandoned ideas** are not decisions. Mark a claim that presents them as decided/serious as CONTRADICTED.
- **Not in the meeting at all** (invented fact/action item) with nothing related in the transcript → INSUFFICIENT, error type "fabrication". If the transcript says the opposite → CONTRADICTED.
- **Several parts in the claim:** all parts must hold for SUPPORTED. If one part is wrong → CONTRADICTED (and mark the claim *compound* below).

## 2. Evidence lines (required for SUPPORTED / CONTRADICTED / CONFLICTING)
Type the line numbers that you used (e.g. `12, 14-16, 40`). Include **all** lines needed, including the later line that revises or confirms. For INSUFFICIENT leave empty, or list lines that come closest.

## 3. Error type (if the label is not SUPPORTED)
number/entity · negation · speaker_attribution · time_shift · revision_ignored (old decision stated as final) · status_flip (proposed vs decided vs rejected) · fabrication (not in meeting) · dependency (claim is only true if another wrong claim is true) · other.

## 4. Atomicity (every claim)
- **atomic** — one checkable fact.
- **compound** — contains two or more independent facts/decisions (should have been split).
- **fragment** — cannot be checked on its own (e.g. "this was agreed", dangling pronoun).

## 5. Whole summary (once per summary, after its claims)
- Rating 1–5 for overall factual accuracy (1 = many serious errors, 3 = some errors, 5 = fully accurate).
- "Would you want a human to check this before using it?" yes/no.
- Missed: list important decisions/action items from the meeting that the summary left out, and any summary statement that no claim captured.

## 6. Process
1. Pilot: both annotators label the same 20 claims, then meet **once** to discuss the rules (this file may be clarified; labels are then frozen).
2. Label everything independently, in the given order. Do not search the internet. About 1–2 min per claim.
3. A third person (or you two together, recording the reason) resolves disagreements in the adjudication step.
4. Use *Undo last* if you mis-click. Save often; the tool resumes where you stopped.

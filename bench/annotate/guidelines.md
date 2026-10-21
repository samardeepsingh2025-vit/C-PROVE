# MeetVerify-Human — Annotation Guidelines (v1.0)

You will see **one atomic claim** taken from an AI-written meeting summary, the
**retrieved transcript evidence**, and a **wider transcript window**. Decide
whether the claim is faithful to the meeting. Use only the transcript. Do not use
outside knowledge, and do not guess what the speakers "probably meant".

## 1. Labels

| Label | Use when |
|---|---|
| **SUPPORTED** | Every part of the claim is backed by the transcript: who, what, when, how much, and the status (proposed / agreed / decided / rejected / assigned / deferred). The claim also reflects the meeting's **latest** position on the topic. |
| **UNSUPPORTED** | At least one part is contradicted, **or** the claim states a position that was later changed in the meeting, **or** it invents content (fabrication). |
| **UNVERIFIABLE** | The claim is too vague to check (for example "The meeting was productive") or is pure meta-commentary. Use this sparingly; it should cover fewer than 5% of items. |

Rule of thumb: if you have to write "yes, but …", the claim is UNSUPPORTED.

## 2. Error types (pick one when UNSUPPORTED; choose the most specific)

1. **entity**: a wrong object, number, price, colour or feature. "price of 20 euros", but the transcript says 25.
2. **negation**: the polarity is flipped. "did not agree", but they agreed.
3. **attribution_swap**: the right content is given to the wrong person. "Marketing proposed kinetic batteries", but the Industrial Designer proposed them.
4. **time_shift**: a wrong deadline, date or meeting order. "by Monday", but the deadline was Friday.
5. **revision_ignored**: the claim states an earlier value or decision that was later revised in the meeting. 25 euros was proposed, but 12.50 was finally decided, and the claim says "decided 25".
6. **status_flip**: the right content with the wrong status. "decided", but it was only proposed. "rejected", but it was accepted. "left open", but it was decided.
7. **dependency_cascade**: the claim is wrong **because** it inherits an error from another claim it depends on. Example: "after the prototype on Monday", where Monday is itself wrong.
8. **fabrication**: nothing in the meeting corresponds to the claim.
9. **other**: anything else; explain it in the note.

## 3. Status definitions

- **proposed**: someone suggested it and there is no group acceptance yet.
- **agreed**: at least one other participant explicitly accepts ("yeah", "fine", "agreed") and nobody objects.
- **decided**: explicitly finalised ("that's decided", "let's go with", "final decision"), usually by the PM, or agreed with no later reopening.
- **rejected**: explicitly dropped.
- **assigned**: a task is given to a named owner.
- **deferred**: explicitly postponed or left open.

"Agreed" for "decided" is acceptable (SUPPORTED). "Proposed" for "decided" is **not** acceptable.

## 4. Attribution

- Roles are the names: Project Manager (PM), Marketing (ME), Industrial Designer (ID), User Interface (UI).
- "The group / team decided" is SUPPORTED if at least two participants accept and nobody objects.
- For action items, the owner must match. The person who assigned the task does not have to be named.

## 5. Evidence ids

List the utterance ids (in brackets) that prove your label. For UNSUPPORTED, include the utterance that **contradicts** the claim (for example, the later revision).

## 6. Optional: edges (20% of items, marked "double")

If this claim relates to another claim in the shown summary context, write `type:claim_id` entries separated by `;`. Types:
`before`, `causes`, `task_depends`, `revises`, `conflicts`, `coref`.
These are the gold Claim Dependency Graph edges used for ablation 7.

## 7. Process

1. Calibration round: 50 shared items, then a discussion of disagreements.
2. Main round: about 1,500 claims. 15% are double-annotated for agreement; the target is Cohen's κ ≥ 0.6.
3. Ties are adjudicated by a third annotator.
4. Expected speed is about 60–80 claims per hour. Take a break every 45 minutes.

## 8. Worked examples

| Claim | Transcript gist | Label | Error |
|---|---|---|---|
| The group decided on a kinetic battery. | ID suggests kinetic, UI and ME agree, PM: "that's decided" | SUPPORTED | none |
| The group decided on a price of 25 euros. | ME proposes 25, PM later changes to 12.50 (final) | UNSUPPORTED | revision_ignored |
| Marketing proposed a kinetic battery. | The ID suggested it | UNSUPPORTED | attribution_swap |
| The case material was decided as rubber. | PM: "we will not decide the case material today" | UNSUPPORTED | status_flip |
| The ID will prepare a prototype by Monday. | "by Friday" | UNSUPPORTED | time_shift |
| The team had a productive discussion. | (none) | UNVERIFIABLE | – |

# Supplementary material
Prompt version: `cprove-v1.0`. All prompts are used verbatim with Qwen2.5-7B-Instruct, 4-bit quantised (Q4_K_M, Ollama tag `qwen2.5:7b-instruct-q4_K_M`), temperature 0, at most 768 generated tokens, context length requested 8192.

## S1. Claim decomposition (system)

```text
You decompose sentences from a MEETING SUMMARY into atomic claims and fill a frame for each.
Rules:
1. One claim = one checkable proposition (one agent, one act, one object). Split coordinations ("A and B decided X and Y" -> 4 claims only if each is independently checkable; otherwise 2).
2. Resolve pronouns using the participant list and previous sentences. Never invent names not in the sentence/context.
3. status: proposed (suggested/raised), agreed (group accepted), decided (final decision), rejected, assigned (action item given to someone), deferred (postponed/left open), revised (changed an earlier position), stated (plain fact/report).
4. polarity = 0 only if the claim is negated ("did not", "rejected the idea of" stays polarity 1 with status rejected).
5. agent = who performs/owns the act (speaker or role). For assignments put the owner in "recipient" and the assigner (if stated) in "agent".
6. Copy time expressions and numbers verbatim into time / quantity.
7. Keep claim text short and self-contained. Output JSON only.
```

## S1. Claim decomposition (user template)

```text
Participants: {participants}
Previous summary sentences (context, do NOT decompose): {context}
Sentence to decompose: "{sentence}"
```

## S2. Qwen zero-shot judge (system)

```text
You verify one claim from a meeting summary against transcript excerpts.
A claim is SUPPORTED only if every slot is supported: who (speaker/owner), what, when, the status (proposed vs decided vs rejected), and it is the LATEST position (not one later revised).
UNSUPPORTED if any slot is contradicted or if the excerpts show it was later changed.
INSUFFICIENT if the excerpts do not mention it at all.
Cite utterance ids. faithful_prob is your probability the claim is fully faithful. Output JSON only.
```

## S2. Qwen zero-shot judge (user template)

```text
Claim: "{claim}"
Frame: agent={agent}; act={act}; object={obj}; time={time}; status={status}; polarity={polarity}; quantity={quantity}; recipient={recipient}

Transcript excerpts (chronological):
{evidence}

Related claims in the same summary and their current belief:
{neighbours}
```

## S3. Claim dependency edge prompt (system)

```text
Given two claims from the same meeting summary, output the single strongest relation from claim A to claim B:
before (A happens earlier than B), causes (A is the reason for B), task_depends (B's task needs A done first),
revises (B changes/overrides A), conflicts (A and B cannot both be true), coref (same event described twice), none.
Output JSON only.
```

## S4. Summary generation (system)

```text
You write abstractive meeting summaries. Cover: main topics, decisions (with who decided), action items (with owners and deadlines), and any changes of plan. 8-15 sentences. Use participant roles as names.
```

## S5. Decision-status check (system)

```text
You label the STATUS of one item in a meeting. Statuses: proposed (someone suggested it, no response yet); discussed (talked about, no decision); accepted (the participants agreed or decided to do it); rejected (decided against it); deferred (postponed to later or another person); revised (an earlier decision was changed); unresolved (disagreement or open question remains). Answer from the quoted text only. If the quoted transcript does not talk about the item, transcript_status is not_discussed.
```

## S6. Annotation guidelines (v1)

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

## 6. How these guidelines were applied

The labels used in the paper were produced in two steps. An LLM drafted a label for every claim; one author then checked each draft against the transcript using the definitions above. A separate 50-claim subset was labelled independently by two annotators with these definitions (binary Cohen's $\kappa = 0.80$); 28\% of the original labels changed after adjudication (paper subsections "Annotation Protocol and Label Reliability" and "Sensitivity to Label Noise"). The guidelines were written before annotation as the target specification; claim-dependency edges were not annotated.

## 7. Worked examples

| Claim | Transcript gist | Label | Error |
|---|---|---|---|
| The group decided on a kinetic battery. | ID suggests kinetic, UI and ME agree, PM: "that's decided" | SUPPORTED | none |
| The group decided on a price of 25 euros. | ME proposes 25, PM later changes to 12.50 (final) | UNSUPPORTED | revision_ignored |
| Marketing proposed a kinetic battery. | The ID suggested it | UNSUPPORTED | attribution_swap |
| The case material was decided as rubber. | PM: "we will not decide the case material today" | UNSUPPORTED | status_flip |
| The ID will prepare a prototype by Monday. | "by Friday" | UNSUPPORTED | time_shift |
| The team had a productive discussion. | (none) | UNVERIFIABLE | – |
"""Corpus loaders -> list[Meeting] (+ reference summaries).

Supported sources (put them under data/, see README "Data"):
  ami_nxt   AMI manual annotations v1.6.2 (NXT XML): words, segments, dialogue acts,
            abstractive summaries (abstract / decisions / actions / problems), speaker roles.
            https://groups.inf.ed.ac.uk/ami/download/  -> ami_public_manual_1.6.2.zip
  qmsum     QMSum JSONL (contains AMI "Product" and ICSI "Academic" meetings with transcripts)
            https://github.com/Yale-LILY/QMSum  (data/ALL/jsonl/{train,val,test}.jsonl)
  tofueval  TofuEval MeetingBank annotations (sentence-level factual consistency labels)
            https://github.com/amazon-science/tofueval  + MeetingBank transcripts (HF: huuuyeah/meetingbank)
  jsonl     our own normalised format: {"meeting_id", "utterances":[{uid,speaker,text,start,end}], "meta", "summaries"}
  csv       one row per utterance: meeting_id, speaker, text[, start, end]
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from ..schema import Meeting, Utterance

NITE = "{http://nite.sourceforge.net/}"
AMI_ROLE = {"PM": "Project Manager", "ME": "Marketing", "ID": "Industrial Designer", "UI": "User Interface"}
# AMI DA classes (da-types.xml ami_da_1..16) -> coarse labels used by the status model
AMI_DA = {"ami_da_1": "backchannel", "ami_da_2": "stall", "ami_da_3": "fragment", "ami_da_4": "inform",
          "ami_da_5": "elicit-inform", "ami_da_6": "suggest", "ami_da_7": "offer", "ami_da_8": "elicit-offer-or-suggestion",
          "ami_da_9": "assess", "ami_da_11": "comment-about-understanding", "ami_da_12": "elicit-assessment",
          "ami_da_13": "elicit-comment-understanding", "ami_da_14": "be-positive", "ami_da_15": "be-negative",
          "ami_da_16": "other"}


def _href_range(href: str):
    m = re.search(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?", href)
    return (m.group(1), m.group(2) or m.group(1)) if m else (None, None)


def _ami_roles(root: Path) -> dict:
    f = root / "corpusResources" / "meetings.xml"
    roles = defaultdict(dict)
    if f.exists():
        for mt in ET.parse(f).getroot().iter("meeting"):
            obs = mt.get("observation")
            for sp in mt.iter("speaker"):
                r = sp.get("role")
                roles[obs][sp.get("nxt_agent")] = AMI_ROLE.get(r, r or sp.get("nxt_agent"))
    return roles


def load_ami_nxt(root: str | os.PathLike, meeting_ids: list[str] | None = None, max_meetings=None):
    root = Path(root)
    roles = _ami_roles(root)
    word_files = sorted(glob.glob(str(root / "words" / "*.words.xml")))
    by_meeting = defaultdict(list)
    for wf in word_files:
        mid, agent = Path(wf).name.split(".")[:2]
        by_meeting[mid].append(agent)
    mids = sorted(by_meeting)
    if meeting_ids:
        mids = [m for m in mids if m in set(meeting_ids)]
    if max_meetings:
        mids = mids[:max_meetings]
    meetings, refs = [], {}
    for mid in mids:
        segs = []
        for agent in by_meeting[mid]:
            words = {}
            order = []
            for w in ET.parse(root / "words" / f"{mid}.{agent}.words.xml").getroot():
                wid = w.get(NITE + "id")
                order.append(wid)
                if w.tag == "w":
                    words[wid] = (w.text or "", float(w.get("starttime", 0) or 0), float(w.get("endtime", 0) or 0),
                                  w.get("punc") == "true")
            pos = {wid: i for i, wid in enumerate(order)}
            # dialogue acts (optional)
            da_of_word = {}
            daf = root / "dialogueActs" / f"{mid}.{agent}.dialog-act.xml"
            if daf.exists():
                for d in ET.parse(daf).getroot().iter("dact"):
                    t = None
                    for p in d.iter(NITE + "pointer"):
                        t = AMI_DA.get(_href_range(p.get("href"))[0], None)
                    for ch in d.iter(NITE + "child"):
                        a, b = _href_range(ch.get("href"))
                        if a in pos and b in pos:
                            for wid in order[pos[a]:pos[b] + 1]:
                                da_of_word[wid] = t
            sf = root / "segments" / f"{mid}.{agent}.segments.xml"
            for s in ET.parse(sf).getroot().iter("segment"):
                for ch in s.iter(NITE + "child"):
                    a, b = _href_range(ch.get("href"))
                    if a not in pos or b not in pos:
                        continue
                    ids = order[pos[a]:pos[b] + 1]
                    toks = [words[i] for i in ids if i in words]
                    if not toks:
                        continue
                    text = ""
                    for t, _, _, punc in toks:
                        text += t if (punc or not text) else " " + t
                    das = [da_of_word.get(i) for i in ids if da_of_word.get(i)]
                    segs.append((toks[0][1], toks[-1][2], roles.get(mid, {}).get(agent, agent), text.strip(),
                                 max(set(das), key=das.count) if das else None))
        segs.sort(key=lambda x: x[0])
        utts = [Utterance(f"{mid}_u{i:04d}", spk, txt, st, en, i, da) for i, (st, en, spk, txt, da) in enumerate(segs) if txt]
        meetings.append(Meeting(mid, utts, {"corpus": "ami", "participants": sorted(set(u.speaker for u in utts))}))
        refs[mid] = load_ami_abstractive(root, mid)
    return meetings, refs


def load_ami_abstractive(root: Path, mid: str) -> dict:
    f = Path(root) / "abstractive" / f"{mid}.abssumm.xml"
    if not f.exists():
        return {}
    r = ET.parse(f).getroot()
    out = {}
    for sec in ("abstract", "actions", "decisions", "problems"):
        node = r.find(sec)
        out[sec] = [s.text.strip() for s in node.iter("sentence")] if node is not None else []
    return out


def load_qmsum(path_glob: str, corpus_filter: str | None = None, max_meetings=None):
    """QMSum JSONL. corpus_filter: 'ami' (Product) | 'icsi' (Academic) | None."""
    meetings, refs = [], {}
    seen = set()
    for f in sorted(glob.glob(path_glob)):
        with open(f, encoding="utf-8") as fh:
            for k, line in enumerate(fh):
                d = json.loads(line)
                tr = d["meeting_transcripts"]
                mid = d.get("meeting_id") or f"{Path(f).stem}_{k}"
                spk = {t["speaker"] for t in tr}
                kind = "ami" if any(s in AMI_ROLE.values() for s in spk) else "icsi" if any(re.match(r"(PhD|Professor|Grad|Postdoc|Undergrad)", s) for s in spk) else "other"
                if corpus_filter and kind != corpus_filter:
                    continue
                if mid in seen:
                    continue
                seen.add(mid)
                utts = [Utterance(f"{mid}_u{i:04d}", t["speaker"], t["content"], 0, 0, i) for i, t in enumerate(tr)]
                meetings.append(Meeting(mid, utts, {"corpus": kind, "participants": sorted(spk)}))
                refs[mid] = {"abstract": [q["answer"] for q in d.get("general_query_list", [])]}
                if max_meetings and len(meetings) >= max_meetings:
                    return meetings, refs
    return meetings, refs


def load_tofueval(ann_csv: str, transcripts_jsonl: str | None = None):
    """Returns (meetings, rows) where rows are sentence-level labels:
    {doc_id, summ_id, sent_idx, sentence, label(1 faithful/0), error_type}.
    TofuEval column names have changed across releases, so we match loosely."""
    rows = []
    with open(ann_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            low = {k.lower().strip(): v for k, v in r.items()}
            doc = low.get("doc_id") or low.get("document_id") or low.get("source_id")
            sent = low.get("summary_sentence") or low.get("sentence") or low.get("summary sentence")
            lab = (low.get("annotation") or low.get("label") or low.get("sent_label") or "").strip().lower()
            rows.append({"doc_id": doc, "summ_id": low.get("summ_id") or low.get("summary_id") or doc,
                         "sent_idx": int(low.get("sent_idx") or low.get("sent_id") or 0), "sentence": sent,
                         "label": 1 if lab in ("yes", "1", "true", "consistent", "supported") else 0,
                         "error_type": low.get("type") or low.get("error_type") or ""})
    meetings = []
    if transcripts_jsonl:
        meetings = [m for m in load_meeting_jsonl(transcripts_jsonl) if m.meeting_id in {r["doc_id"] for r in rows}]
    return meetings, rows


def load_meeting_jsonl(path) -> list[Meeting]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            utts = [Utterance(**u) if isinstance(u, dict) and "uid" in u else
                    Utterance(f"{d['meeting_id']}_u{i:04d}", u["speaker"], u["text"], u.get("start", 0), u.get("end", 0), i)
                    for i, u in enumerate(d["utterances"])]
            for i, u in enumerate(utts):
                u.idx = i
            out.append(Meeting(d["meeting_id"], utts, d.get("meta", {})))
    return out


def save_meetings_jsonl(meetings: list[Meeting], path, refs: dict | None = None):
    from dataclasses import asdict
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for m in meetings:
            d = {"meeting_id": m.meeting_id, "utterances": [asdict(u) for u in m.utterances], "meta": m.meta}
            if refs and m.meeting_id in refs:
                d["references"] = refs[m.meeting_id]
            f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")


def load_csv(path) -> list[Meeting]:
    rows = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["meeting_id"]].append(r)
    out = []
    for mid, rs in rows.items():
        utts = [Utterance(f"{mid}_u{i:04d}", r["speaker"], r["text"], float(r.get("start") or 0), float(r.get("end") or 0), i)
                for i, r in enumerate(rs)]
        out.append(Meeting(mid, utts, {"participants": sorted({u.speaker for u in utts})}))
    return out


def merge_short(m: Meeting, min_words=3) -> Meeting:
    """Merge consecutive same-speaker fragments and drop pure backchannels — AMI
    segments are very short; this keeps retrieval windows meaningful."""
    out = []
    for u in m.utterances:
        if out and out[-1].speaker == u.speaker and (len(out[-1].text.split()) < 25):
            out[-1].text += " " + u.text
            out[-1].end = u.end
            continue
        out.append(Utterance(u.uid, u.speaker, u.text, u.start, u.end, 0, u.dialogue_act))
    out = [u for u in out if len(u.text.split()) >= min_words or u.dialogue_act not in ("backchannel", "stall")]
    for i, u in enumerate(out):
        u.idx = i
        u.uid = f"{m.meeting_id}_u{i:04d}"
    return Meeting(m.meeting_id, out, m.meta)


def _with_refs(ms, refs, corpus):
    """AMI: keep only meetings that have a reference abstract (drops EN*/IB* non-scenario meetings)."""
    if corpus != "ami" or not any(refs.get(m.meeting_id) for m in ms):
        return ms, refs
    keep = [m for m in ms if refs.get(m.meeting_id, {}).get("abstract") and not m.meeting_id.startswith("IB")]
    return keep, {m.meeting_id: refs[m.meeting_id] for m in keep}


def load_corpus(cfg: dict):
    """Dispatch on cfg['data']['corpus'] and the files present under data/."""
    d = Path(cfg["paths"]["data"])
    corpus = cfg["data"]["corpus"]
    mx = cfg["data"].get("max_meetings")
    cache = d / f"{corpus}_meetings.jsonl"
    if cache.exists():
        ms = load_meeting_jsonl(cache)
        refs = {}
        with open(cache, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                refs[r["meeting_id"]] = r.get("references", {})
        ms, refs = _with_refs(ms, refs, corpus)
        return (ms[:mx] if mx else ms), refs
    if corpus == "ami" and (d / "ami_public_manual_1.6.2" / "words").exists():
        ms, refs = load_ami_nxt(d / "ami_public_manual_1.6.2", max_meetings=mx)
        ms = [merge_short(m) for m in ms]
    elif corpus in ("ami", "icsi") and glob.glob(str(d / "QMSum" / "data" / "ALL" / "jsonl" / "*.jsonl")):
        ms, refs = load_qmsum(str(d / "QMSum" / "data" / "ALL" / "jsonl" / "*.jsonl"), corpus, mx)
    elif (d / f"{corpus}.csv").exists():
        ms, refs = load_csv(d / f"{corpus}.csv"), {}
    else:
        raise FileNotFoundError(f"No data for corpus '{corpus}' under {d}. See README 'Data'.")
    save_meetings_jsonl(ms, cache, refs)
    ms, refs = _with_refs(ms, refs, corpus)
    return (ms[:mx] if mx else ms), refs
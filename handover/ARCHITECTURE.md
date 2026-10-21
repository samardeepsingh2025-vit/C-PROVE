# ARCHITECTURE
Input: transcript (Meeting: list of Utterance{uid,speaker,text}) + summary text. Output: Result{claims, edges, posteriors y* per claim and per dimension,
labels, MTI, interval, components, proof traces, trace}.

| stage | module | input -> output | method |
|---|---|---|---|
| 0 summaries | bench/generate.py | meeting -> summary | map-reduce with Qwen2.5-7B / Llama-3.1-8B / Mistral-7B GGUF |
| 1 claims | frames.py | summary -> Claim[]+Frame(agent,act,object,time,status,polarity,quantity,recipient) | LLM JSON-schema (few-shot) or heuristic |
| 2 graph | graph.py | claims -> Edge[] (before,causes,same_agent,task_depends,coref,revises,conflicts) | frame rules; optional DeBERTa pair classifier |
| 3 evidence | evidence.py, nli.py | claim -> Evidence windows (+-2), ent/neu/con | 0.7 BGE + 0.3 TF-IDF + slot boost; DeBERTa-v3-large NLI windows mixed so ent+con<=1 |
| 4 dimensions | dimensions.py | claim+evidence -> Observation(fact,attr,temp,ctx,sem), None if inapplicable | per-dimension signals, isotonic calibration |
| 5a ground | rules.py | claims+obs+edges -> hinge potentials | Lukasiewicz rules R1..R10b |
| 5b infer | hlmrf.py | program -> y* in [0,1]^n | preconditioned projected gradient (<=200 it); lambda=softplus(theta) learned by unrolled solver |
| 5c adaptive | adaptive.py, judge.py | y*, graph -> judge calls | VoI = H(y)*(1+sum|dy_j/do_i|) via autograd; warm re-solve |
| 6 outputs | mti.py, conformal.py, explain.py | y* -> labels, MTI, interval, traces | weighted mean, CRC, split conformal, top-k ground rules |
Latent variables per claim: y, fact, rsn (+attr,temp,ctx,sem if applicable). Rule families: see docstring of rules.py.
Config knobs: configs/default.yaml (llm, retrieval, nli, hlmrf.init_weights, adaptive, mti, labels, train).
Data flow caching: stages 1-4 cached per benchmark in cache/items_*_raw.pkl (experiments/common.build_items); stage 5-6 recomputed per experiment.

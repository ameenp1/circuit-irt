"""Respondent harness (Week 3): prompt -> model completion -> netlist -> score.

  build_prompt(item)        -> chat messages asking for a JSON-wrapped netlist
  parse_completion(text)    -> netlist str | None   (robust to small-model mess)
  evaluate_completion(...)  -> {parsed, label, graded, ...}  via the grading harness

The harness owns the test fixture (supplies, stimulus, load, analyses), so the
respondent must emit ONLY the circuit devices (+ any .model) using the canonical
node names. The parser strips control cards and any source the model puts on a
fixture node, and recovers a netlist from fenced JSON, loose JSON, code blocks,
or bare text.

`LocalModel` runs small open models via transformers; `python respondent.py` runs
a deterministic mock-completion battery, and `RUN_MODELS=1 python respondent.py`
additionally evaluates the models in models.yaml on a few items.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter

from circuit_irt.families import FAMILIES
from circuit_irt.harness import (ItemSpec, FailureMode, simulate, extract_metrics, score,
                     classify, fingerprint)
from circuit_irt.reference import make_plan, ANALYSIS_FOR
from circuit_irt.generators import _clause
from circuit_irt.paths import DATA, CONFIGS

FIXTURE_NODES = {"vdd", "vss", "in", "inp", "inm"}   # driven by the harness fixture
_ELEM = "RCLMQDJVIEFGHBKXT"
_CTRL = (".control", ".endc", ".ac", ".dc", ".tran", ".op", ".end", ".print",
         ".plot", ".save", ".meas", ".four", ".step")


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #
def _spec_text(item) -> str:
    fam = FAMILIES[item["family_id"]]
    targets = {k: (tuple(v) if isinstance(v, list) else v) for k, v in item["targets"].items()}
    clauses = [_clause(fam.metric(k), targets[k], item["tolerance"].get(k, 0.0), 0)
               for k in item["objectives"]]
    s = "; ".join(clauses)
    if item["corner"] != "none":
        s += f"; robust across {item['corner']}"
    return s


def build_prompt(item) -> list[dict]:
    fam = FAMILIES[item["family_id"]]
    nodes = ", ".join(f"`{n}`" for n in fam.nodes.values())
    system = (
        "You are an analog IC designer. Given a specification, output a SPICE "
        "netlist that meets it.\n"
        "Rules:\n"
        f"- Use ONLY these node names: {nodes} (0 is ground).\n"
        "- Emit ONLY the circuit devices and any .model lines. Do NOT include "
        "power supplies, input sources, load capacitors, .control blocks, or "
        "analysis commands — those are added by the test bench.\n"
        "- MOSFETs use a model named NM, e.g. `.model NM NMOS (LEVEL=1 VTO=0.45 "
        "KP=120u LAMBDA=0.02)`.\n"
        '- Respond with ONLY a JSON object: {"netlist": "<lines separated by \\n>"}.\n'
        'Example: {"netlist": "M1 out in 0 0 NM W=80u L=1u\\nRD vdd out 4k\\n'
        '.model NM NMOS (LEVEL=1 VTO=0.45 KP=120u LAMBDA=0.02)"}')
    user = f"Design a {fam.title.lower()} meeting: {_spec_text(item)}."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# completion parsing
# --------------------------------------------------------------------------- #
def _clean_netlist(raw: str) -> str:
    """Keep device + .model lines; drop control cards and any source the model
    placed on a fixture node (those conflict with the test bench)."""
    out = []
    for ln in raw.splitlines():
        t = ln.strip().strip("`").rstrip(",")
        if not t or t.startswith("*"):
            continue
        low = t.lower()
        if any(low.startswith(c) for c in _CTRL):
            continue
        if low.startswith(".model") or low.startswith(".param") or low.startswith(".subckt") \
                or low.startswith(".ends") or low.startswith(".inc"):
            out.append(t); continue
        toks = t.split()
        name = toks[0]
        # a real element line: valid instance name, known prefix, >=3 tokens, and
        # carries numeric node/value content (rejects prose like "I can't help").
        if (name[0].upper() not in _ELEM or len(toks) < 3
                or not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name)
                or not re.search(r"\d", t)):
            continue
        if name[0].upper() in "VI" and {toks[1].lower(), toks[2].lower()} & FIXTURE_NODES:
            continue                                  # model-supplied supply/stimulus
        out.append(t)
    return "\n".join(out)


def _is_netlist(s: str) -> bool:
    return sum(1 for ln in s.splitlines()
               if ln[:1].upper() in _ELEM and len(ln.split()) >= 3) >= 1


def _unescape(s: str) -> str:
    try:
        return s.encode().decode("unicode_escape")
    except Exception:
        return s.replace("\\n", "\n").replace('\\"', '"')


def parse_completion(text: str) -> str | None:
    """Recover a circuit netlist from a model completion, or None.

    Tolerant of the ways 1-3B models malform output: fenced/loose/single-quoted
    JSON, trailing commas, real (unescaped) newlines inside the string, and
    truncated completions (model hit the token limit mid-netlist)."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<\|.*?\|>", " ", text)

    candidates: list[str] = []
    # 1. strict JSON objects (also retry after stripping trailing commas)
    for m in re.finditer(r"\{.*?\}", text, flags=re.S):
        blob = m.group(0)
        for attempt in (blob, re.sub(r",(\s*[}\]])", r"\1", blob)):
            try:
                obj = json.loads(attempt)
                for key in ("netlist", "spice", "circuit"):
                    if isinstance(obj.get(key), str):
                        candidates.append(obj[key])
                break
            except Exception:
                pass
    # 2. loose value regex for invalid JSON — double- and single-quoted, closed...
    for q in ('"', "'"):
        pat = r'["\']netlist["\']\s*:\s*' + q + r'((?:[^' + q + r'\\]|\\.)*)' + q
        for m in re.finditer(pat, text, flags=re.S):
            candidates.append(_unescape(m.group(1)))
    # 3. ...and the truncated case: an opening quote with no close (token cutoff)
    m = re.search(r'["\']netlist["\']\s*:\s*["\'](.*)$', text, flags=re.S)
    if m:
        candidates.append(_unescape(re.split(r'["\']\s*[},]', m.group(1))[0]))
    # 4. fenced code blocks (may itself be unterminated)
    for m in re.finditer(r"```(?:\w+)?\s*(.*?)(?:```|$)", text, flags=re.S):
        candidates.append(m.group(1))
    # 5. the whole thing (bare netlist)
    candidates.append(text)

    for cand in candidates:
        cleaned = _clean_netlist(cand)
        if _is_netlist(cleaned):
            return cleaned
    return None


# --------------------------------------------------------------------------- #
# completion -> harness -> score
# --------------------------------------------------------------------------- #
class _PlanItem:
    def __init__(self, rec):
        self.family_id = rec["family_id"]
        self.targets = rec["targets"]


def evaluate_completion(completion: str, item: dict) -> dict:
    """Completion -> netlist -> graded result. Never raises: any internal failure
    is captured as a labeled record so a long inference run can't be crashed by a
    single pathological output."""
    base = {"item_id": item.get("item_id"), "family": item["family_id"],
            "parsed": False, "label": FailureMode.PARSE_FAILURE.value,
            "reason": "no_netlist", "ngspice_status": None,
            "graded": 0.0, "all_pass": False, "failed_metrics": [], "netlist": None}

    netlist = parse_completion(completion)
    if netlist is None:
        return base                                  # respondent emitted no usable netlist

    base.update(parsed=True, netlist=netlist)
    try:
        fam = FAMILIES[item["family_id"]]
        objs = tuple(item["objectives"])
        analyses = tuple(sorted({ANALYSIS_FOR[k] for k in objs}))
        raw = simulate(netlist, make_plan(_PlanItem(item), analyses))
        metrics = extract_metrics(raw, fam)
        targets = {k: (tuple(v) if isinstance(v, list) else v)
                   for k, v in item["targets"].items()}
        spec = ItemSpec(fam, targets, objs, item["tolerance"], item["corner"])
        s = score(metrics, spec, syntax_ok=raw.syntax_ok, converged=raw.converged)
        label = classify(netlist, raw, s, fingerprint(item["reference_netlist"]))
        failed = [k for k, ok in s["per_metric_pass"].items() if not ok]
        reason = ({FailureMode.PARSE_FAILURE.value: "invalid_spice",
                   FailureMode.NON_CONVERGENCE.value: raw.status.value}
                  .get(label.value))
        base.update(label=label.value, reason=reason, ngspice_status=raw.status.value,
                    graded=s["graded"], all_pass=s["all_pass"], failed_metrics=failed)
    except Exception as e:                            # harness bug / unexpected input
        logging.exception("harness error on item %s", item.get("item_id"))
        base.update(label="harness_error", reason=f"{type(e).__name__}: {e}")
    return base


class FailureLog:
    """Accumulates failure-category counts over a batch (for the Week 6 analysis)."""
    def __init__(self):
        self.categories = Counter()
        self.reasons = Counter()
        self.records = []

    def record(self, r: dict) -> dict:
        self.categories[r["label"]] += 1
        if r.get("reason"):
            self.reasons[f"{r['label']}:{r['reason']}"] += 1
        self.records.append({k: r[k] for k in
                             ("item_id", "family", "label", "reason",
                              "ngspice_status", "graded", "all_pass")})
        return r

    def summary(self) -> dict:
        return {"categories": dict(self.categories), "reasons": dict(self.reasons),
                "n": len(self.records)}

    def to_json(self, path):
        json.dump({"summary": self.summary(), "records": self.records},
                  open(path, "w"), indent=1)


def evaluate_batch(pairs, log: FailureLog | None = None):
    """pairs: iterable of (completion, item). Returns (results, FailureLog)."""
    log = log or FailureLog()
    results = [log.record(evaluate_completion(c, it)) for c, it in pairs]
    return results, log


# --------------------------------------------------------------------------- #
# local model runner (transformers)
# --------------------------------------------------------------------------- #
class LocalModel:
    def __init__(self, model_id: str, max_new_tokens: int = 400, temperature: float = 0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, messages: list[dict]) -> str:
        import torch
        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def load_models_yaml(path=CONFIGS / "models.yaml") -> list[dict]:
    import yaml
    return yaml.safe_load(open(path))["models"]


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def _mock_battery():
    items = json.load(open(DATA / "candidate_bank.json"))["items"]
    fitem = next(i for i in items if i["family_id"] == "filters")
    ref = fitem["reference_netlist"]                 # a correct solution
    wrong = "R1 in out 1k\nR2 out 0 1k"              # resistive divider, wrong topology

    cases = [
        ("clean JSON", json.dumps({"netlist": ref}), True, "pass"),
        ("json fenced", f"```json\n{json.dumps({'netlist': ref})}\n```", True, "pass"),
        ("prose+json", f"Here is my design.\n{json.dumps({'netlist': ref})}\nDone.", True, "pass"),
        ("spice fence", f"```spice\n{ref}\n```", True, "pass"),
        ("bare netlist", ref, True, "pass"),
        ("reasoning wrap", f"<think>need an RC LP</think>\n{json.dumps({'netlist': ref})}", True, "pass"),
        ("invalid json, recoverable",
         '{"netlist": "' + ref.replace("\n", "\\n") + '", oops}', True, "pass"),
        ("model adds supply+ctrl",
         json.dumps({"netlist": "VDD vdd 0 1.8\n" + ref + "\n.ac dec 10 1 1e6\n.end"}),
         True, "pass"),
        ("refusal", "I'm sorry, I can't help with that.", False, "parse_failure"),
        ("empty", "", False, "parse_failure"),
        ("wrong topology", json.dumps({"netlist": wrong}), True, "wrong_topology"),
    ]
    print("=== mock completion battery (filters item "
          f"{fitem['item_id']}) ===")
    for name, comp, exp_parsed, exp_label in cases:
        r = evaluate_completion(comp, fitem)
        ok = (r["parsed"] == exp_parsed) and (r["label"] == exp_label)
        print(f"  {'OK ' if ok else 'XX '}{name:<26} parsed={r['parsed']!s:<5} "
              f"label={r['label']:<16} graded={r['graded']:.2f}")
        assert ok, f"{name}: parsed={r['parsed']} label={r['label']} (want {exp_parsed}/{exp_label})"
    print("OK: parser + completion->harness->score path validated on all formats.\n")


def _run_models():
    items = json.load(open(DATA / "candidate_bank.json"))["items"]
    probe = ([next(i for i in items if i["family_id"] == "filters" and i["tier"] == "easy")] +
             [next(i for i in items if i["family_id"] == "cs_amp" and i["tier"] == "easy")])
    limit = int(os.environ.get("MODELS_LIMIT", "99"))
    for spec in load_models_yaml()[:limit]:
        print(f"\n=== {spec['id']} ===")
        try:
            lm = LocalModel(spec["id"], max_new_tokens=spec.get("max_new_tokens", 400))
        except Exception as e:
            print(f"  load failed: {type(e).__name__}: {e}")
            continue
        for it in probe:
            comp = lm.generate(build_prompt(it))
            r = evaluate_completion(comp, it)
            print(f"  {it['item_id']:<16} parsed={r['parsed']!s:<5} "
                  f"label={r['label']:<16} graded={r['graded']:.2f}")


if __name__ == "__main__":
    _mock_battery()
    if os.environ.get("RUN_MODELS") == "1":
        _run_models()
    else:
        print("(set RUN_MODELS=1 to additionally evaluate models.yaml on real models)")

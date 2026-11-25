"""
Replicate the "Experiment Without S.E Knowledge" run for IM_Software
using local Ollama models instead of GPT-4o / GPT-4 Turbo.

Same prompts as the original notebooks, N independent runs per model,
raw outputs saved, optional scoring against Ground_Truth_AC.csv
(BLEU, difflib similarity, TF-IDF cosine - the same three metrics as R1.ipynb).

Usage:
    ollama pull qwen2.5:3b          # etc.
    python replicate_im_software_ollama.py --models qwen2.5:3b gemma3:4b
    python replicate_im_software_ollama.py --models qwen2.5:3b --runs 5 --temperature 1.0
    python replicate_im_software_ollama.py --score-only      # re-score saved outputs
"""
import argparse
import csv
import difflib
import json
import re
import sys
from pathlib import Path

import requests
import sacrebleu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###########################################################################
## 
class CSafetyAssurance:
    OLLAMA_URL = "http://localhost:11434"
    THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

    # --- identical to the original notebooks -------------------------------
    SYSTEM_PROMPT = (
        "You are an assistant who assists in developing an assurance case in a "
        "tree structure using Goal Structuring Notation (GSN). "
        "Your role is to create an assurance case."
    )

    USER_PROMPT = (
        "Create a security case for Instant messaging (IM) software and display "
        "in a hierarchical tree format using dashes (-) to denote different levels."
    )

    def __init__(self):
        print('CSafetyAssurance::init')

    def Installed_models(self):
        r = requests.get(f"{self.OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])
                if "embed" not in m["name"].lower()]


    def Generate(self, model, temperature, max_tokens, seed=None, num_ctx=8192):
        options = {"temperature": temperature, "num_predict": max_tokens,
                "num_ctx": num_ctx}
        if seed is not None:
            options["seed"] = seed
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.USER_PROMPT},
            ],
            "stream": False,
            "think": False,          # ignored by models without a thinking mode
            "options": options,
        }
        r = requests.post(f"{self.OLLAMA_URL}/api/chat", json=payload, timeout=900)
        r.raise_for_status()
        data = r.json()
        text = data.get("message", {}).get("content", "")
        return self.THINK_RE.sub("", text).strip(), data.get("done_reason")


    def safe(self, name):
        return name.replace(":", "_").replace("/", "_")


    def run_experiment(self, models, runs, temperature, 
                       max_tokens, seed, out_dir):
        have = self.installed_models()
        for model in models:
            if model not in have:
                print(f"[skip] {model} not installed. Run: ollama pull {model}")
                continue
            folder = out_dir / self.safe(model) / "IM_Software"
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(1, runs + 1):
                # seed=None keeps runs random (matches the original temp=1 setup);
                # a fixed seed + offset makes each run reproducible but different.
                s = None if seed is None else seed + i
                print(f"{model}  run {i}/{runs} ...", flush=True)
                text, reason = self.Generate(model, temperature, max_tokens, s)
                (folder / f"{i}.txt").write_text(text, encoding="utf-8")
                meta = {"model": model, "run": i, "temperature": temperature,
                        "max_tokens": max_tokens, "seed": s, "done_reason": reason}
                (folder / f"{i}.json").write_text(json.dumps(meta, indent=2))
                if reason == "length":
                    print("   warning: output cut off by max_tokens")


    def load_ground_truth(self, path, case="IM_Software"):
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("Case") == case:
                    return row["AC"]
        raise ValueError(f"{case} not found in {path} (expected columns: Case, AC)")


    def score(self, out_dir, gt_path):
        gt = self.load_ground_truth(gt_path)
        rows = []
        for model_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
            folder = model_dir / "IM_Software"
            files = sorted(folder.glob("*.txt"), key=lambda p: int(p.stem))
            if not files:
                continue
            texts = [f.read_text(encoding="utf-8") for f in files]
            tfidf = TfidfVectorizer().fit_transform(texts + [gt])
            cos = cosine_similarity(tfidf[:-1], tfidf[-1]).ravel()
            for f, t, c in zip(files, texts, cos):
                bleu = sacrebleu.sentence_bleu(t, [gt]).score / 100  # 0-1 like R1
                seq = difflib.SequenceMatcher(None, t, gt).ratio()
                rows.append((model_dir.name, f.stem, bleu, seq, c))

        print(f"{'model':<22}{'run':<5}{'BLEU':>7}{'seqmatch':>10}{'tfidf':>8}")
        for m, r, b, s, c in rows:
            print(f"{m:<22}{r:<5}{b:>7.2f}{s:>10.2f}{c:>8.2f}")
        with open(out_dir / "scores.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "run", "bleu", "seqmatch", "tfidf_cosine"])
            w.writerows(rows)
        print(f"\nSaved {out_dir / 'scores.csv'}")

###########################################################################
## 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen2.5:3b", "gemma3:4b"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="Ollama_Experiment")
    ap.add_argument("--ground-truth", default="Ground_Truth_AC.csv")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    objSA = CSafetyAssurance()

    if not a.score_only:
        try:
            objSA.run_experiment(a.models, a.runs, a.temperature,
                           a.max_tokens, a.seed, out_dir)
        except requests.ConnectionError:
            sys.exit("Ollama is not running. Start it with: ollama serve")
    if not a.no_score:
        objSA.score(out_dir, Path(a.ground_truth))


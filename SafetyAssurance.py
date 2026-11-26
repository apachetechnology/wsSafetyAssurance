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


    def Generate(self, model, fTemperature, max_tokens, seed=None, num_ctx=8192):
        options = {"temperature": fTemperature, "num_predict": max_tokens,
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
        strText = data.get("message", {}).get("content", "")
        return self.THINK_RE.sub("", strText).strip(), data.get("done_reason")


    def safe(self, name):
        return name.replace(":", "_").replace("/", "_")

    def Run_experiment(self, listModels, runs, fTemperature, 
                       max_tokens, seed, out_dir):
        listInstalledModels = self.Installed_models()
        for strModel in listModels:
            print('Model:', strModel)
            if strModel not in listInstalledModels:
                print(f"[skip] {strModel} not installed. Run: ollama pull {strModel}")
                continue

            # Assign the file name
            strFilename = 'IM_Software_' + self.safe(strModel) + '_'

            for i in range(1, runs + 1):
                # seed=None keeps runs random (matches the original temp=1 setup);
                # a fixed seed + offset makes each run reproducible but different.
                s = None if seed is None else seed + i
                print(f"{strModel}  run {i}/{runs} ...", flush=True)
                
                strText, strReason = self.Generate(strModel, fTemperature, max_tokens, s)
                
                (out_dir / f"{strFilename}{i}.txt").write_text(strText, encoding="utf-8")
                
                meta = {"model": strModel, "run": i, "temperature": fTemperature,
                        "max_tokens": max_tokens, "seed": s, "done_reason": strReason}
                
                (out_dir / f"{strFilename}{i}.json").write_text(json.dumps(meta, indent=2))
                
                if strReason == "length":
                    print("   warning: output cut off by max_tokens")

    def load_ground_truth(self, path, case):
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("Case") == case:
                    return row["AC"]
        raise ValueError(f"{case} not found in {path} (expected columns: Case, AC)")


    def Score(self, out_dir, gt_path, strKey):
        gt = self.load_ground_truth(gt_path, strKey)
        pat = re.compile(rf"^{re.escape(strKey)}_(.+)_(\d+)\.txt$")
        
        groups = {}
        for p in out_dir.glob( strKey + "_*.txt"):
            m = pat.match(p.name)
            if m:
                groups.setdefault(m.group(1), []).append((int(m.group(2)), p))

        rows = []
        for model, items in sorted(groups.items()):
            items.sort()
            texts = [p.read_text(encoding="utf-8") for _, p in items]
            tfidf = TfidfVectorizer().fit_transform(texts + [gt])
            cos = cosine_similarity(tfidf[:-1], tfidf[-1]).ravel()
            for (run, _), t, c in zip(items, texts, cos):
                bleu = sacrebleu.sentence_bleu(t, [gt]).score / 100
                seq = difflib.SequenceMatcher(None, t, gt).ratio()
                rows.append((model, run, bleu, seq, c))

        #return rows
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
            objSA.Run_experiment(a.models, a.runs, a.temperature,
                           a.max_tokens, a.seed, out_dir)
        except requests.ConnectionError:
            sys.exit("Ollama is not running. Start it with: ollama serve")
    if not a.no_score:
        objSA.Score(out_dir, Path(a.ground_truth))


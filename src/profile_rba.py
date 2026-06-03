import csv
import io
import json
import os
import random
import zipfile

random.seed(42)
csv.field_size_limit(10**7)
ZIP="data/raw/rba/rba-dataset.zip"; NAME="rba-dataset.csv"
OUT="data/samples/rba_sample.csv"
os.makedirs("data/samples",exist_ok=True); os.makedirs("results",exist_ok=True)
KEEP=0.006  # ~33M*0.006 ≈ 200k normal rows in sample
total=ato=attack=succ=0
with zipfile.ZipFile(ZIP) as z, z.open(NAME) as raw, open(OUT,"w",newline="") as out:
    txt=io.TextIOWrapper(raw,encoding="utf-8",errors="replace")
    r=csv.reader(txt); header=next(r); w=csv.writer(out); w.writerow(header)
    def find(subs):
        for i,c in enumerate(header):
            cl=c.strip().lower()
            if any(s in cl for s in subs): return i
        return None
    ai=find(["account takeover"]); ki=find(["attack ip","is attack"]); si=find(["login successful"])
    for row in r:
        total+=1
        is_ato = ai is not None and row[ai].strip().lower() in ("true","1")
        if is_ato: ato+=1
        if ki is not None and row[ki].strip().lower() in ("true","1"): attack+=1
        if si is not None and row[si].strip().lower() in ("true","1"): succ+=1
        if is_ato or random.random()<KEEP: w.writerow(row)
        if total%5000000==0: print(f"...{total:,} rows ato={ato} attack={attack}",flush=True)
prof={"dataset":"rba","rows":total,"cols":len(header),"columns":header,
      "label_indices":{"account_takeover":ai,"attack_ip":ki,"login_successful":si},
      "account_takeover":ato,"attack_ip":attack,"login_successful":succ,
      "ato_rate_pct":round(ato/total*100,6) if total else 0,
      "attack_rate_pct":round(attack/total*100,4) if total else 0}
json.dump(prof,open("results/profile_rba.json","w"),indent=2)
print("DONE", json.dumps({k:prof[k] for k in ['rows','account_takeover','attack_ip','login_successful','ato_rate_pct']}),flush=True)

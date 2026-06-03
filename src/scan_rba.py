import json
import os
import sys

import numpy as np
import pandas as pd

os.makedirs("data/samples",exist_ok=True); os.makedirs("results",exist_ok=True)
rng=np.random.default_rng(42); KEEP=0.006
total=ato=attack=succ=0; cols=None; ato_c=atk_c=suc_c=None; parts=[]
def find(cols,subs):
    for c in cols:
        if any(s in c.lower() for s in subs): return c
    return None
for ch in pd.read_csv(sys.stdin, chunksize=1_000_000, dtype=str, low_memory=False):
    if cols is None:
        cols=list(ch.columns)
        ato_c=find(cols,["account takeover"]); atk_c=find(cols,["attack ip","is attack"]); suc_c=find(cols,["login successful"])
    n=len(ch); total+=n
    a = ch[ato_c].str.strip().str.lower().isin(["true","1"]) if ato_c else pd.Series(False,index=ch.index)
    ato+=int(a.sum())
    if atk_c: attack+=int(ch[atk_c].str.strip().str.lower().isin(["true","1"]).sum())
    if suc_c: succ+=int(ch[suc_c].str.strip().str.lower().isin(["true","1"]).sum())
    parts.append(ch[a | (rng.random(n)<KEEP)])
samp=pd.concat(parts); samp.to_csv("data/samples/rba_sample.csv",index=False)
prof={"dataset":"rba","rows":int(total),"cols":len(cols),"columns":cols,
      "label_cols":{"account_takeover":ato_c,"attack_ip":atk_c,"login_successful":suc_c},
      "account_takeover":int(ato),"attack_ip":int(attack),"login_successful":int(succ),
      "ato_rate_pct":round(ato/total*100,6),"attack_rate_pct":round(attack/total*100,4),
      "sample_rows":int(len(samp)),"sample_ato":int(samp[ato_c].str.strip().str.lower().isin(["true","1"]).sum()) if ato_c else 0}
json.dump(prof,open("results/profile_rba.json","w"),indent=2)
print("DONE rows=%d ato=%d attack=%d succ=%d sample=%d"%(total,ato,attack,succ,len(samp)))

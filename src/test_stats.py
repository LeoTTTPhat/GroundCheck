#!/usr/bin/env python3
"""Sanity tests for src/stats.py against known analytic values.

These functions carry every inferential claim in the paper -- the confidence
intervals, the paired tests, the multiplicity correction and the equivalence
bounds -- so they are checked against textbook values rather than trusted.

Run: python3 src/test_stats.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stats as st
fails=[]
def chk(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ")+name+(f"  {extra}" if extra else ""))
    if not cond: fails.append(name)

# Wilson: textbook value for 0/10 and 5/10
p,lo,hi = st.wilson(5,10)
chk("wilson 5/10 centred", abs(p-0.5)<1e-12 and abs(lo-0.2366)<1e-3 and abs(hi-0.7634)<1e-3, f"[{lo:.4f},{hi:.4f}]")
p,lo,hi = st.wilson(0,10)
chk("wilson 0/10 lower bound clamped at 0", lo==0.0 and 0.27<hi<0.31, f"[{lo:.4f},{hi:.4f}]")
p,lo,hi = st.wilson(10,10)
chk("wilson 10/10 upper bound clamped at 1", hi <= 1.0 and hi > 1 - 1e-9, f"[{lo:.4f},{hi:.4f}]")

# McNemar: exact binomial on discordant pairs
a=[1]*8+[0]*2+[1]*45+[0]*45
b=[0]*8+[1]*2+[1]*45+[0]*45
r=st.mcnemar(a,b)
from scipy import stats as sps
expect=sps.binomtest(8,10,0.5).pvalue
chk("mcnemar discordant counts", r["n10"]==8 and r["n01"]==2)
chk("mcnemar p matches exact binomial", abs(r["p"]-expect)<1e-12, f"{r['p']:.5f}")
r0=st.mcnemar([1,1,0,0],[1,1,0,0])
chk("mcnemar p=1 when no discordant pairs", r0["p"]==1.0 and r0["n10"]==0)

# Holm: known step-down example
adj,rej = st.holm([0.01,0.04,0.03])
chk("holm adjusts and enforces monotonicity",
    abs(adj[0]-0.03)<1e-12 and abs(adj[1]-0.06)<1e-12 and abs(adj[2]-0.06)<1e-12,
    f"{[round(x,4) for x in adj]}")
chk("holm rejects only the first", list(rej)==[True,False,False])

# Bootstrap: paired CI must bracket the true difference and be narrower than unpaired
rng=np.random.default_rng(0)
base=rng.integers(0,2,400)
cur=base.copy(); cur[:40]=1-cur[:40]          # 10% flipped
d,lo,hi=st.boot_diff_ci(cur,base,paired=True)
chk("paired bootstrap brackets observed diff", lo<=d<=hi, f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]")
d2,lo2,hi2=st.boot_diff_ci(cur,base,paired=False)
chk("paired CI narrower than unpaired", (hi-lo)<(hi2-lo2), f"paired {hi-lo:.4f} vs unpaired {hi2-lo2:.4f}")
chk("bootstrap deterministic under fixed seed",
    st.boot_diff_ci(cur,base)==st.boot_diff_ci(cur,base))

# TOST: identical vectors are equivalent; a 30-point gap is not
same=list(base)
chk("tost: identical -> equivalent", st.tost_equivalence(same,same,margin=0.05)["equivalent"])
big=base.copy(); big[:120]=1
chk("tost: large gap -> not equivalent", not st.tost_equivalence(big,base,margin=0.05)["equivalent"])

# Response diversity
chk("diversity: constant answer -> modal_share 1, entropy 0",
    st.response_diversity([3.0]*50)["modal_share"]==1.0 and
    st.response_diversity([3.0]*50)["norm_entropy"]==0.0)
u=st.response_diversity([float(i) for i in range(50)])
chk("diversity: all-distinct -> modal_share 1/n, entropy 1",
    abs(u["modal_share"]-0.02)<1e-9 and abs(u["norm_entropy"]-1.0)<1e-9)
chk("diversity: floats bucketed at 2dp",
    st.response_diversity([1.900001,1.9,1.9])["distinct"]==1)

print(f"\n{16-len(fails)}/16 passed")
sys.exit(1 if fails else 0)

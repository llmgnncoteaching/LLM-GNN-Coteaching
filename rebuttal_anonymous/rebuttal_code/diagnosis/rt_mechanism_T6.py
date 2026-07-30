# Extracted analysis (ran on server against results dirs).
import csv, os, glob
def stats(d):
    fs=glob.glob(f"{d}/results/co_teaching/*/progress.csv")
    if not fs: return None
    r=[x for x in csv.reader(open(fs[0])) if x and x[0].isdigit()]
    if not r: return None
    import statistics as st
    # x8 gnn_selected, x9 llm_selected, x10 gnn_pseudo_acc, x11 llm_pseudo_acc
    def col(i):
        v=[float(x[i]) for x in r if len(x)>i and x[i] not in ('','nan')]
        return sum(v)/len(v) if v else 0
    llm_max=max(float(x[3]) for x in r)*100
    return dict(llm_max=llm_max, n_gsel=col(8), n_lsel=col(9), gq=col(10), lq=col(11), rounds=len(r))
for ds,schemes in [("cora",["f02","lin","f06","f10"]),("arxiv",["f02","lin","f06"])]:
    print(f"\n=== {ds} ===")
    print(f"{'sch':4} {'llm_max':>7} {'n_gsel':>7} {'n_lsel':>7} {'g_qual':>7} {'l_qual':>7} rnds")
    for s in schemes:
        d=f"rtabl_{ds}_{s}" if not (ds=="arxiv" and s=="f10") else f"rtabl_{ds}_f10b"
        st=stats(d)
        if st: print(f"{s:4} {st['llm_max']:7.1f} {st['n_gsel']:7.0f} {st['n_lsel']:7.0f} {st['gq']:7.3f} {st['lq']:7.3f} {st['rounds']}")
        else: print(f"{s:4} NONE")
# arxiv f10b separately
st=stats("rtabl_arxiv_f10b")
if st: print(f"arxiv f10b(nosel) llm_max={st['llm_max']:.1f} n_gsel={st['n_gsel']:.0f} n_lsel={st['n_lsel']:.0f} gq={st['gq']:.3f} lq={st['lq']:.3f} rnds={st['rounds']}")

import torch, sys, re, argparse
sys.path.insert(0,'.')
from common.prompt import DIRECT_PROMPTS
from vllm import LLM, SamplingParams
MODEL="/project/anon/rebuttal/Meta-Llama-3-8B-Instruct"
ap=argparse.ArgumentParser()
ap.add_argument("--dataset",required=True)
ap.add_argument("--mode",default="zero")
a=ap.parse_args()
d=torch.load("datasets/%s.pt"%a.dataset, map_location="cpu", weights_only=False)
def get(k): return d[k] if isinstance(d,dict) else getattr(d,k)
raw=get("raw_texts"); y=get("y"); tm=get("test_mask"); labels=[str(l) for l in get("label_name")]
idx=[i for i in range(len(y)) if bool(tm[i])]
llm=LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=True)
tok=llm.get_tokenizer()
base=DIRECT_PROMPTS[a.dataset]
def build(t):
    txt=str(t)[:2500]
    if a.mode=="cot":
        instr=base.replace("Respond with only the exact category name from the list above.",
                           "Think step by step in 2 to 3 sentences, then on the last line write Answer: <category>.")
        user=txt+"\n\n"+instr
    else:
        user=txt+"\n\n"+base
    return tok.apply_chat_template([{"role":"user","content":user}], tokenize=False, add_generation_prompt=True)
prompts=[build(raw[i]) for i in idx]
sp=SamplingParams(temperature=0.0, max_tokens=(220 if a.mode=="cot" else 16))
outs=llm.generate(prompts, sp)
def parse(txt):
    t=txt.lower()
    if a.mode=="cot":
        m=re.findall(r"answer\s*:?\s*([a-z ]+)", t)
        if m: t=m[-1]
    for li,l in enumerate(labels):
        if l.lower() in t: return li
    return -1
correct=sum(1 for o,i in zip(outs,idx) if parse(o.outputs[0].text)==int(y[i]))
print("ZSRESULT %s %s acc=%.2f n=%d"%(a.dataset, a.mode, 100*correct/len(idx), len(idx)))

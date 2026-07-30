import sys, torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data

ds = sys.argv[1]            # e.g. arxiv
src = sys.argv[2]           # path to our .pt (LLMNodeBed format)
out = sys.argv[3]           # output {ds}_fixed_sbert.pt path

g = torch.load(src, weights_only=False)
texts = list(g.raw_texts)
y = g.y.view(-1).long().cpu()
label_names = [str(l) for l in g.label_name]
category_names = [label_names[int(y[i])] for i in range(y.numel())]

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
x = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_tensor=True).cpu().float()

def _mask(m):
    base = m.cpu().bool() if torch.is_tensor(m) else torch.zeros(y.numel(), dtype=torch.bool)
    return base.view(1, -1)  # (1, N): Locle indexes masks by seed as X_masks[i]

d = Data(
    x=x, y=y, edge_index=g.edge_index.cpu().long(),
    raw_texts=texts, raw_text=texts,
    label_names=label_names, category_names=category_names,
    train_masks=_mask(getattr(g, "train_mask", None)),
    val_masks=_mask(getattr(g, "val_mask", None)),
    test_masks=_mask(getattr(g, "test_mask", None)),
)
torch.save(d, out)
print("SAVED", out, "x", tuple(x.shape), "nclass", len(label_names),
      "test", int(d.test_masks.sum()))

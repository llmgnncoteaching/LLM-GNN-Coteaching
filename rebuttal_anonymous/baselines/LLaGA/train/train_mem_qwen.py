# jun18_llaga_qwen: launcher without llama flash-attn monkey-patch (Qwen2 uses native attention)
import sys
sys.path.append(".")
sys.path.append("./utils")
from train import _train

if __name__ == "__main__":
    _train()

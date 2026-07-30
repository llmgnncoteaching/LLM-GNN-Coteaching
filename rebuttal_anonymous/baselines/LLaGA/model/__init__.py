# Patched: gate optional MPT/OPT paths behind try/except so the LLaMA path
# still imports under newer transformers versions that dropped _expand_mask.
from .language_model.llaga_llama import LlagaLlamaForCausalLM, LlagaConfig
try:
    from .language_model.llaga_mpt import LlagaMPTForCausalLM, LlagaMPTConfig  # noqa: F401
except Exception as _e:
    LlagaMPTForCausalLM = None
    LlagaMPTConfig = None
try:
    from .language_model.llaga_opt import LlagaOPTForCausalLM, LlagaOPTConfig  # noqa: F401
except Exception as _e:
    LlagaOPTForCausalLM = None
    LlagaOPTConfig = None
try:
    from .language_model.llaga_qwen import LlagaQwenForCausalLM, LlagaQwenConfig  # noqa: F401  # jun18_llaga_qwen
except Exception as _e:
    LlagaQwenForCausalLM = None
    LlagaQwenConfig = None

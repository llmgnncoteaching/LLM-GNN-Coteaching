from models.LMs.lm_utils import *


class LlamaConfig(LMConfig):

    def __init__(self, args=None):
        super(LlamaConfig, self).__init__(args)
        self.model = 'Llama'
        self._post_init(args)

    para_prefix = {**LMConfig.para_prefix}
    args_to_parse = list(para_prefix.keys())
    meta_data = {
        'Llama':
            SN(
                hf_model='/project/anon/rebuttal/Meta-Llama-3-8B-Instruct',
                hidden_dim=4096,  # Llama-3-8B last hidden size
                max_bsz=SN(  # per-device batch keyed by GPU mem (GB); 8B + LoRA + bf16
                    train={24: 1, 32: 1, 40: 2, 48: 3, 80: 4},
                    inf={24: 8, 32: 12, 40: 16, 48: 24, 80: 32},
                ),
                prt_lm={  # Initial (pre-train) LM configs
                    'arxiv': SN(
                        model='FtV1',
                        cmd='--att_dropout=0.0 --cla_dropout=0.1 --dropout=0.0 --epochs=1 --eq_batch_size=32 --eval_patience=50000 --label_smoothing_factor=0.1 --load_best_model_at_end=T --lr=1e-4 --warmup_epochs=0.6',
                        max_n_gpus=1,
                    ),
                    'products': SN(
                        model='FtV1',
                        cmd='--att_dropout=0.0 --cla_dropout=0.1 --dropout=0.0 --epochs=1 --eq_batch_size=32 --eval_patience=65308 --label_smoothing_factor=0.1 --lr=1e-4 --warmup_epochs=0.6',
                        max_n_gpus=1,
                    ),
                },
            ),
    }

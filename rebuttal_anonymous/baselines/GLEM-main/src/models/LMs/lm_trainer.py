import math

from sklearn.metrics import accuracy_score, f1_score  # load_metric removed (gone in modern datasets)
from transformers import AutoModel, EvalPrediction, TrainingArguments, Trainer
import utils.function as uf
from models.LMs.model import *
from models.GLEM.GLEM_utils import *
from utils.data.datasets import *
import torch as th

METRICS = {  # metric -> metric_path
    'accuracy': 'src/utils/function/hf_accuracy.py',
    'f1score': 'src/utils/function/hf_f1.py',
    'precision': 'src/utils/function/hf_precision.py',
    'recall': 'src/utils/function/hf_recall.py',
    'spearmanr': 'src/utils/function/hf_spearmanr.py',
    'pearsonr': 'src/utils/function/hf_pearsonr.py',

}


class LMTrainer():
    """Convert textural graph to text list"""

    def __init__(self, cf):
        self.cf = cf
        # logging.set_verbosity_warning()
        from transformers import logging as trfm_logging
        trfm_logging.set_verbosity_error()
        self.logger = cf.logger
        self.log = cf.logger.log
        self.update_ratio = 1

    @uf.time_logger
    def train(self):
        # ! Prepare data
        self.d = d = SeqGraph(cf := self.cf).init()
        gold_data = SeqGraphDataset(self.d, mode='train_gold')
        subset_data = lambda sub_idx: th.utils.data.Subset(gold_data, sub_idx)
        # Cap the intermediate LM valid/test eval to 1000 nodes (final GLEM metric is the GNN's;
        # full-split LM eval on 78K nodes under eager Llama wasted ~2.5h per run).
        self.datasets = {_: subset_data(getattr(d, f'{_}_x') if _ == 'train'
                                        else getattr(d, f'{_}_x')[:1000])
                         for _ in ['train', 'valid', 'test']}

        if cf.is_augmented:
            # Augment Label if Cir-train
            warmup_steps = 0  # No warmup (already warmed up at pre-training step)
            # Sample visible data for current EM-Iter
            init_random_state(cf.seed)
            train_ids = d.get_inf_aug_train_ids(*cf.emi.inf_node_ranges)
            _ = SeqGraphDataset(d, mode='train_augmented')
            self.train_data = th.utils.data.Subset(_, train_ids)
            max_pl_ratio = len(d.pl_nodes) / len(d.labeled_nodes)
            pl_ratio = min(cf.pl_ratio, max_pl_ratio)
            eval_steps = (1 + pl_ratio) * cf.eval_patience // cf.eq_batch_size
        else:
            # Pretrain on gold data
            self.train_data = self.datasets['train']
            train_steps = len(d.train_x) // cf.eq_batch_size + 1
            warmup_steps = int(cf.warmup_epochs * train_steps)
            eval_steps = cf.eval_patience // cf.eq_batch_size

        # ! Load bert and build classifier
        _is_llama = 'llama' in str(cf.hf_model).lower()
        # eager attn: BertClassifier (custom PreTrainedModel) doesn't declare SDPA support; bf16 halves 8B memory.
        _load_kw = {'attn_implementation': 'eager', 'torch_dtype': th.bfloat16} if _is_llama else {}
        bert_model = AutoModel.from_pretrained(cf.hf_model, **_load_kw)
        if _is_llama:
            from peft import LoraConfig, get_peft_model, TaskType
            _eos = bert_model.config.eos_token_id
            bert_model.config.pad_token_id = _eos[0] if isinstance(_eos, (list, tuple)) else _eos
            lora_cfg = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            bert_model = get_peft_model(bert_model, lora_cfg)  # base frozen, only LoRA trains
        self.model = BertClassifier(
            bert_model, cf.data.n_labels,
            pseudo_label_weight=cf.pl_weight if cf.is_augmented else 0,
            dropout=cf.cla_dropout,
            loss_func=th.nn.CrossEntropyLoss(label_smoothing=cf.label_smoothing_factor, reduction=cf.ce_reduction),
            cla_bias=cf.cla_bias == 'T',
            is_augmented=cf.is_augmented,
            feat_shrink=cf.feat_shrink
        )
        if cf.local_rank <= 0:
            trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
            print(f" LM Model parameters are {trainable_params}")
        if cf.is_augmented:
            if cf.init_ckpt == 'PrevEM':
                # ! Load previous LM model
                self.model.load_state_dict(temp := th.load(cf.prev_lm_ckpt, map_location='cpu'), strict=not _is_llama)
                del temp
            elif cf.init_ckpt == 'Prt':
                self.model.load_state_dict(temp := th.load(cf.prt_lm_ckpt, map_location='cpu'), strict=not _is_llama)
                del temp
            elif cf.init_ckpt == 'None':
                pass
            elif cf.init_ckpt == 'EM':
                # ! Don't Load previous Prt LM model
                if cf.emi.iter > 0:
                    print(f'cf.emi.iter = {cf.emi.iter}, load from Prev')
                    self.model.load_state_dict(temp := th.load(cf.prev_lm_ckpt, map_location='cpu'), strict=not _is_llama)
                    del temp
                else:
                    print(f'cf.emi.iter = {cf.emi.iter}, load from None')
                    pass
            else:
                raise NotImplementedError(cf.init_ckpt)
            load_best_model_at_end = cf.load_best_model_at_end == 'T'
        else:
            load_best_model_at_end = True
        if _is_llama:
            load_best_model_at_end = False  # avoid saving 16GB full-model checkpoints during training
        if cf.hf_model == 'distilbert-base-uncased':
            self.model.config.dropout = cf.dropout
            self.model.config.attention_dropout = cf.att_dropout
        elif _is_llama:
            pass  # LlamaConfig has no hidden_dropout_prob/attention_probs_dropout_prob; LoRA dropout regularizes
        else:
            print('default dropout and attention_dropout are:', self.model.config.hidden_dropout_prob, self.model.config.attention_probs_dropout_prob)
            self.model.config.hidden_dropout_prob = cf.dropout
            self.model.config.attention_probs_dropout_prob = cf.att_dropout

        training_args = TrainingArguments(
            output_dir=cf.out_dir,
            evaluation_strategy='steps',
            eval_steps=eval_steps,
            save_strategy='no' if _is_llama else 'steps',
            save_steps=eval_steps,
            learning_rate=cf.lr, weight_decay=cf.weight_decay,
            load_best_model_at_end=load_best_model_at_end, gradient_accumulation_steps=cf.grad_acc_steps,
            save_total_limit=1,
            report_to='wandb' if cf.wandb_on else None,
            per_device_train_batch_size=cf.batch_size,
            per_device_eval_batch_size=cf.batch_size * 6 if cf.hf_model in {'distilbert-base-uncased', 'google/electra-base-discriminator'} else cf.batch_size * 10,
            warmup_steps=warmup_steps,
            disable_tqdm=False,
            dataloader_drop_last=True,
            num_train_epochs=cf.epochs,
            local_rank=cf.local_rank,
            dataloader_num_workers=1,
            fp16=not _is_llama,  # if cf.hf_model=='microsoft/deberta-large' else False
            bf16=_is_llama,  # Llama-3 trains in bf16
            remove_unused_columns=False,  # forward(**input): Trainer can't introspect cols, would strip all -> empty batch
        )


        def compute_metrics(pred: EvalPrediction):
            preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
            refs = pred.label_ids
            predictions = preds.argmax(1)
            references = refs.argmax(1) if refs.ndim > 1 else refs
            # transformers 4.44 keeps only flat scalar metrics; return scalars (eval_and_save reads mtc_dict[m]).
            return {'accuracy': float(accuracy_score(references, predictions)),
                    'f1score': float(f1_score(references, predictions, average='macro'))}



        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.datasets['valid'],
            compute_metrics=compute_metrics,
        )
        self.eval_phase = 'Eval'
        self.trainer.train()
        # ! Save bert
        # self.model.save_pretrained(cf.out_ckpt, self.model.state_dict())
        # ! Save BertClassifer Save model parameters
        if cf.local_rank <= 0:
            sd = self.model.state_dict()
            if _is_llama:
                # base Llama is frozen and reloaded from pretrained; persist only the trainable delta
                sd = {k: v for k, v in sd.items()
                      if ('lora_' in k) or k.startswith('classifier') or k.startswith('feat_shrink_layer')}
            th.save(sd, uf.init_path(cf.lm.ckpt))
        # uf.remove_file(f'{cf.out_dir}')
        self.log(f'LM saved to {cf.lm.ckpt}')

    def eval_and_save(self):
        def get_metric(split):
            self.eval_phase = 'Test' if split == 'test' else 'Eval'
            mtc_dict = self.trainer.predict(self.datasets[split]).metrics
            # 4.44 metrics are flat scalars, e.g. mtc_dict['test_accuracy']=0.9
            ret = {f'{split}_{_}': mtc_dict[m] for m in mtc_dict if (_ := m.split('_')[-1]) in METRICS}
            return ret

        cf = self.cf
        res = {**get_metric('valid'), **get_metric('test')}
        # robust: few-shot LM eval can yield empty metrics; the GLEM number is the GNN test acc, so don't crash here
        res = {'val_acc': res.get('valid_accuracy', 0.0), 'test_acc': res.get('test_accuracy', 0.0)}
        if cf.is_augmented:
            cf.wandb_log({**{f'GLEM/LM_{k}': v for k, v in res.items()},
                          'EM-Iter': cf.emi.end})
            cf.em_info.lm_res_list.append(res)
            uf.pickle_save(cf.em_info, cf.emi_file)
        else:  # Pretrain
            # Save results for pre-training to be reported at main ct-loop
            uf.pickle_save(res, cf.lm.result)
            cf.wandb_log({f'lm_prt_{k}': v for k, v in res.items()})

        self.log(f'\nTrain seed{cf.seed} finished\nResults: {res}\n{cf}')

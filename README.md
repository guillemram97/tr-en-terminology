# Incorporating a terminology list to a MT model
Our method is based on [Dinu's method](https://aclanthology.org/P19-1294.pdf) for incorporating a terminology list. The idea is that the model learns
when a lexical constraint is applied and doesn't learn the terminology words.
The original method used factors. However, we are applying this method at word-level:

**Input**: `All <0> alternates <1> Stellvertreter <2> shall be elected for one term`

**Output**: `Alle Stellvertreter werden fur eine Amtszeit gewahlt` 
  
This was the most used method in [terminology shared task of WMT2021](https://aclanthology.org/2021.wmt-1.69/).

---

## Finetunning the original model
The original model is a MT model for English to Turkish and vice versa. We used the same training data as the original model to avoid a change in performance. However, the training data was modified to include terminology tags: we identified word pairs from the [MUSE dictionary](https://github.com/facebookresearch/MUSE) that appeared in both the source and target sentence, and tagged the source sentences accordingly:

`In our Solar System there are perhaps more than billion huge, <0> stray <1> başıboş <2> meteors that would cause serious consequences should they collide .`

For pre-processing we rely on sentencepiece; we can re-use the instructions given [here](https://github.com/pytorch/fairseq/blob/main/examples/mbart/README.md). However, we previously have to modify `sentence.bpe.model` so that it includes the tokens `<0>`, `<1>` and `<2>`: [this notebook](https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=8420f2179007c398c8b70f63cb12d8aec827397c&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f676f6f676c652f73656e74656e636570696563652f383432306632313739303037633339386338623730663633636231326438616563383237333937632f707974686f6e2f6164645f6e65775f766f6361622e6970796e62&logged_in=false&nwo=google%2Fsentencepiece&path=python%2Fadd_new_vocab.ipynb&platform=android&repository_id=84183882&repository_type=Repository&version=98) explains how to do this to create a new sentencepiece tokenizer called `sentence.term.bpe.model`.

Once pre-processed, we need to binarize the data in order to train our fairseq model; again we can follow the mBART25 instructions. The important part here is to make sure you use the mBART25 dictionary while preprocessing, and include the three new special tokens. The easiest way to do this is to simply replace the last three tokens of the mBART25 dictionary by `<0>`, `<1>` and `<2>`.

Once your data is binarized you can call the fairseq-train command with several hparams.

**tr-en**  

The exact command we used is:
```
SRC=tr_TR
TGT=en_XX
NAME=${SRC}-${TGT}
DATA_DIR=data/terminology/bin/${NAME}
PRETRAIN=ckpts/tr-en/synthetic/edi/checkpoint_best.pt 
SAVE_DIR=ckpts/${NAME}/terminology/
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

token_size=1024

fairseq-train $DATA_DIR --finetune-from-model $PRETRAIN --encoder-normalize-before --decoder-normalize-before --arch mbart_large --layernorm-embedding --task translation_from_pretrained_bart --source-lang $SRC --target-lang $TGT --criterion cross_entropy --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler polynomial_decay --lr 3e-05 --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 --warmup-updates 2500 --total-num-update 80000 --max-tokens $token_size  --update-freq 8 --langs $langs --no-epoch-checkpoints  --patience 5 --save-dir $SAVE_DIR --fp16

```

**en-tr**
```
To be continued...
```

Note that we did not use label-smoothing and use a few hparams different than the example. You can easily change this what works best for you. See the [fairseq documentation](https://fairseq.readthedocs.io/en/latest/command_line_tools.html). 

For finetuning, you might want to start with the already trained model which is available under `ckpts/tr-en/synthetic/edi/checkpoint_best.pt`. 

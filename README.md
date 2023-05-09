# NumerSenseRedux: Extended Probing of Numerical Commonsense Knowledge of BERTs


Oringinal Project website: https://inklab.usc.edu/NumerSense/

Code & Data for 2023 extension paper:

```bibtex
@inproceedings{lin2023enumersense,
  title={Extending "Birds have four legs?! NumerSense: Probing Numerical Commonsense Knowledge of Pre-trained Language Models"},
  author={Alex Baroody and Emilio Cano Renteria and Allen Wu}, 
  year={2023}
}
```

## Installation 

```bash
conda create -n numersenseredux python=3.7
conda activate numersenseredux
# install torch seperately at https://pytorch.org/get-started/locally/ if needed
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch -n numersenseredux
pip install transformers==3.3.1
# pip install happytransformer -U
pip install --editable happy-transformer
pip install tensorboardX
mkdir pred_results

# Optional:
# Install apex following https://github.com/NVIDIA/apex#linux
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Probing Experiments 

For masked language models:
```bash
python src/mlm_predict.py bert-base \
        data/test.core.masked.txt \
        results/bert-base.test.core.output.jsonl

python src/mlm_predict.py bert-base \
        data/test.all.masked.txt \
        results/bert-base.test.all.output.jsonl
```

Note that `bert-base` can be replaced by any model name in `[bert-base, bert-large, roberta-base, roberta-large, distilbert-base, albert-base, deberta-large]`.

For left-to-right language models:
```bash
python src/gpt_predict.py gpt \
        data/test.core.masked.txt \
        results/gpt.test.core.output.jsonl 
```

### Fine-tune a MLM model 
```bash
mkdir saved_models
CUDA_VISIBLE_DEVICES=0 python src/finetune_mlm.py \
  --output_dir=saved_models/finetuned_bert_large --overwrite_output_dir \
  --model_type=bert \
  --model_name_or_path=bert-large-uncased \
  --do_train \
  --train_data_file=data/gkb_best_filtered.txt  \
  --do_eval \
  --eval_data_file=data/wiki_complete.txt \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --block_size 64 \
  --logging_steps 100 \
  --num_train_epochs 3 \
  --line_by_line --mlm 
```

```bash 
python src/mlm_predict.py \
        reload_bert:saved_models/finetuned_bert_large \
        data/test.core.masked.txt \
        results/test.core.output.jsonl
```

## Evaluation on Validation Set

Check out `data/validation.masked.tsv`. We realease 200 annotated examples (132 from the `core` split and 68 from the `all` split) for method development so that users can better test their method frquently without submitting the prediction for the test set. **Note that these 200 examples should NOT be used for any training.** Also, they are still part of the the test data.

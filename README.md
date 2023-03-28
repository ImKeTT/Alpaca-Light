## Alpaca-Light: LLaMA Instruct-Tuning with Prefix or LoRA

**[Repo In progress]** Tune LLaMA with Hugging Face's [PEFT](https://github.com/huggingface/peft), support [Prefix](https://arxiv.org/abs/2101.00190) and [LoRA](https://arxiv.org/abs/2106.09685).

## Instruction Tuning

### Setup

```bash
pip install -r requirements.txt
```

### Training (`finetune-PE.py`)

#### Single GPU

```bash
## --data
# alpaca: alpaca_cleaned data
# alpaca-cot: reasoning-enhanced version
# alpaca-belle: Chinese-enhanced version
# alpaca-belle-cot: full-data version 
## --pe_name
# ["prefix", "lora"]


python finetune-PE.py --pe_name prefix --dataset alpaca-belle-cot
```

#### Multiple GPU

```bash
python -m torch.distributed.launch --nproc_per_node 4  \
     finetune-PE.py --pe_name prefix --data alpaca-belle-cot
```

### Inference (`generate.py`)

```bash
## to be completed
python generate.py
```

We modified our codes from [here](https://github.com/tloen/alpaca-lora). You can find more details there.

### Dataset

Datasets are all from this [repo](https://github.com/PhoebusSi/Alpaca-CoT), thanks a lot. All datasets are publicly available. Please download them and put them in `./data` folder.

### Examples

TBD.

### Notes

- We're continually fixing bugs and conducting training runs, we will upload prefix weights once we finish the training.
- Users with multiple GPUs should take a look [here](https://github.com/tloen/alpaca-lora/issues/8#issuecomment-1477490259).

### Credits

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971v1)
- [Stanford Alpaca: An Instruction-following LLaMA model](https://github.com/tatsu-lab/stanford_alpaca)
- [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)
- [FLAN: Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
- [BELLE: Bloom-Enhanced Large Language model Engine](https://github.com/LianjiaTech/BELLE)
- https://github.com/tloen/alpaca-lora
- https://github.com/PhoebusSi/Alpaca-CoT

If you find our codes useful, please star the repo and consider to cite it.

```bibtex
@misc{alpaca-light,
  author = {Haoqin Tu},
  title = {Tune LLaMA with Prefix/LoRA},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ImKeTT/Alpaca_Light}},
}
```

# CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization

<!-- [![arXiv](https://img.shields.io/badge/arxiv-CEPO-blue)](https://arxiv.org/abs/XXXX.XXXXX) -->
[![GitHub Repo stars](https://img.shields.io/github/stars/ahmedheakl/CEPO)](https://github.com/ahmedheakl/CEPO/stargazers)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

> **Ahmed Heakl, Abdelrahman M. Shaker, Youssef Mohamed, Rania Elbadry, Omar Fetouh, Fahad Shahbaz Khan, Salman Khan**
>
> MBZUAI, Linköping University, Australian National University

![CEPO Pipeline](assets/cepo-main.png)

## TL;DR

In RLVR training (e.g., GRPO), every token in a correct trajectory gets the same reward — whether it's a decisive reasoning step or grammatical filler. **CEPO** fixes this by asking a contrastive question at each token: *does the correct answer favor this token **while** the wrong answer disfavors it?* This is done by replacing the single-reference evidence ratio $P_T^+ / P_S$ (used in RLSD) with a contrastive ratio $P_T^+ / P_T^-$, where the wrong-answer teacher $P_T^-$ is constructed from rejected rollouts already in the batch — **zero additional sampling cost**.

## Key Results

CEPO achieves **43.43%** and **60.56%** average accuracy across five multimodal math reasoning benchmarks at 2B and 4B scale, versus **41.17%** and **57.43%** for GRPO under identical training budgets.

![Accuracy over training steps](assets/cepo-teaser.png)

| Method | DynaMath | LogicVista | MathVis. | MMMU | WeMath | **Average** |
|---|---|---|---|---|---|---|
| **Qwen3-VL-2B-Instruct** | | | | | | |
| Base | 50.08 | 32.81 | 19.41 | 44.11 | 52.24 | 39.73 |
| + GRPO | 50.36 | 37.50 | 21.05 | 42.33 | 54.60 | 41.17 |
| + RLSD | 50.36 | 36.38 | 23.39 | 39.44 | 55.26 | 40.05 |
| + **CEPO (Ours)** | **51.44** | **37.72** | **25.99** | **45.78** | **56.21** | **43.43** |
| **Qwen3-VL-4B-Instruct** | | | | | | |
| Base | 64.59 | 54.91 | 44.41 | 53.56 | 74.31 | 58.36 |
| + GRPO | 63.97 | 54.98 | 42.76 | 52.34 | 73.10 | 57.43 |
| + RLSD | 65.07 | 56.92 | 44.08 | 53.22 | 73.28 | 58.51 |
| + **CEPO (Ours)** | **65.37** | **61.16** | **47.37** | **54.11** | **74.77** | **60.56** |

> **Note:** OPSD and SDPO fall *below* the untrained baseline on most benchmarks, empirically confirming the information leakage our theory predicts.

## How It Works

CEPO defines a **contrastive evidence delta** at each token position:

$$\Delta_t^{CE} = \text{sg}\!\left(\log \frac{P_T^+(y_t)}{P_T^-(y_t)}\right)$$

where $P_T^+$ is the model conditioned on the correct answer and $P_T^-$ is conditioned on a wrong answer from rejected rollouts. This has a clean **Bayesian interpretation** as the *differential belief update*: how much token $y_t$ simultaneously strengthens belief in $r^+$ and weakens it for $r^-$.

- **Decisive reasoning steps** → large $|\Delta_t^{CE}|$ → amplified credit
- **Filler tokens** → $\Delta_t^{CE} \approx 0$ → near-unity weight (unchanged from GRPO)

The modulated advantage is then:

$$\hat{A}_t^{(i)} = A^{(i)} \cdot \left[(1 - \lambda) + \lambda \cdot \text{clip}(w_t^{CE},\; 1 - \epsilon_w,\; 1 + \epsilon_w)\right]$$

plugged into a standard PPO-clipped surrogate. When $G^- = \emptyset$, CEPO reduces exactly to RLSD.

![Token-level credit assignment](assets/cepo-tokenmap.png)

### Theoretical Guarantees

1. **Direction anchoring** — $\text{sign}(\hat{A}_t) = \text{sign}(A)$ for all tokens; privileged info cannot flip any token's update direction.
2. **Leakage-free gradient** — No vocabulary-wide $r$-conditioned sum in $\nabla_\theta \mathcal{L}$; $r^+$ and $r^-$ enter only as stop-gradiented scalars.
3. **RLSD containment** — Setting $P_T^- = P_S$ recovers RLSD exactly; RLSD is the degenerate case where the wrong-answer teacher carries no information.

### Positioning: GRPO → RLSD → CEPO

| Method | Credit Assignment | Denominator | Contrastive? |
|---|---|---|---|
| GRPO | Uniform sequence-level | — | ✗ |
| RLSD | Token-level via $P_T^+ / P_S$ | Student prior (fluency confound) | ✗ |
| **CEPO** | Token-level via $P_T^+ / P_T^-$ | Wrong-answer teacher | ✓ |


## Roadmap

- [ ] Scale training to 200 steps
- [ ] Train on harder datasets (e.g., MMFine)
- [ ] Extend to text-only LLMs using the DAPO dataset
- [ ] Evaluate at larger model scales (7B+)

## Installation

This project is built on top of [EasyR1](https://github.com/hiyouga/EasyR1). We thank all the EasyR1 authors for providing such a high-performance RL training framework.

```bash
git clone https://github.com/ahmedheakl/CEPO.git
cd CEPO
pip install -e .
```


## Quick Start

### Training with CEPO

```bash
bash experiments/geo/cepo.sh
```

### Training with GRPO (baseline)

```bash
bash experiments/geo/grpo.sh
```

### Training with RLSD (baseline)

```bash
bash experiments/geo/rlsd.sh
```

> For SDPO and OPSD baselines, we use their official codebases directly: [SDPO](https://github.com/lasgroup/SDPO), [OPSD](https://github.com/siyan-zhao/OPSD).


All experiment scripts are under `experiments/geo/`:

| Script | Description |
|---|---|
| `cepo.sh` | CEPO default configuration |
| `grpo.sh` | GRPO baseline |
| `rlsd.sh` | RLSD baseline |

### Merge Checkpoint

After training, merge the LoRA checkpoint into Hugging Face format:

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## Training Configuration

All experiments use the following shared configuration:

| Hyperparameter | Value |
|---|---|
| Base models | Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct |
| Training dataset | [Geo3k](https://huggingface.co/datasets/hiyouga/geometry3k) (3,000 geometry problems) |
| Training steps | 50 |
| Optimizer | AdamW (lr = 1e-6, cosine decay, 5-step warmup) |
| Batch size | 32 prompts |
| Rollout group size | 8 |
| LoRA rank / α | 16 / 32 |
| Max sequence length | 2,048 tokens |

CEPO-specific hyperparameters:

| Hyperparameter | Default |
|---|---|
| Evidence weight $\lambda_0$ | 0.5 |
| $\lambda$ decay | Linear → 0 over $T_{\text{warm}} = 25$ steps |
| Evidence clip $\epsilon_w$ | 0.5 |
| Positive reference $r^+$ | Ground truth answer |
| Negative reference $r^-$ | Rejected rollout (answer only) |
| Teacher source | Actor policy (shared weights) |

## Evaluation

We evaluate on five held-out multimodal mathematical reasoning benchmarks using [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval):

- **[DynaMath](https://arxiv.org/abs/2411.00836)** — Dynamic visual math reasoning
- **[LogicVista](https://arxiv.org/abs/2407.04973)** — Multimodal logical reasoning in visual contexts
- **[MathVision-mini](https://arxiv.org/abs/2407.14352)** — Multimodal mathematical reasoning
- **[MMMU](https://arxiv.org/abs/2311.16502)** — Massive multi-discipline multimodal understanding
- **[WeMath](https://arxiv.org/abs/2407.01284)** — Mathematical reasoning for LMMs

Evaluation settings: temperature 1.0, top-p 1.0, top-k 40, presence penalty 2.0, max 32,000 tokens.

```bash
# Example evaluation with lmms-eval (adjust model path accordingly)
MODEL="<path_to_merged_checkpoint>"
python -m lmms_eval \
    --model vllm \
    --model_args "model=${MODEL},max_model_len=40000,dtype=bfloat16" \
    --tasks dynamath_reasoning,logicvista_reasoning,wemath_testmini_reasoning,mathvision_testmini,mmmu_val \
    --batch_size 64 \
    --gen_kwargs temperature=1.0,top_p=1.0,top_k=40,presence_penalty=2.0,max_tokens=32000"
```



## Wall-Clock Training Time

| Method | Time (50 steps on Geo3k) |
|---|---|
| GRPO | 5h 58m |
| SDPO | 6h 14m |
| RLSD | 6h 15m |
| **CEPO** | 6h 34m |

> CEPO's two teacher forward passes add only ~36 minutes over GRPO.

## Custom Dataset

Follow the [EasyR1 dataset format](https://github.com/hiyouga/EasyR1#custom-dataset):

- Text: [hiyouga/math12k](https://huggingface.co/datasets/hiyouga/math12k)
- Image-text: [hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)
- Multi-image: [hiyouga/journeybench-multi-image-vqa](https://huggingface.co/datasets/hiyouga/journeybench-multi-image-vqa)

## Citation

```bibtex
@misc{heakl2025cepo,
  title        = {CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization},
  author       = {Ahmed Heakl and Abdelrahman M. Shaker and Youssef Mohamed and Rania Elbadry and Omar Fetouh and Fahad Shahbaz Khan and Salman Khan},
  year         = {2025},
  eprint       = {XXXX.XXXXX},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG}
}
```


## Acknowledgements

This project is built on [EasyR1](https://github.com/hiyouga/EasyR1) (a fork of [veRL](https://github.com/volcengine/verl)). We thank all the authors for providing such a high-performance RL training framework.
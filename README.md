# CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization

<div align="left" style="margin:24px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif"
       width="100%" height="4"/>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2605.19436"><img src="https://img.shields.io/badge/arXiv-Paper-brightgreen?style=flat-square" alt="arXiv"></a>
  <a href="https://github.com/ahmedheakl/CEPO/stargazers"><img src="https://img.shields.io/github/stars/ahmedheakl/CEPO" alt="GitHub Repo stars"></a>
  <a href=""><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
</p>


<p align="center">
  <a href="https://ahmedheakl.github.io/"><b>Ahmed Heakl</b></a>, 
  <a href="https://amshaker.github.io/"><b>Abdelrahman M. Shaker</b></a>,
  <a href="https://scholar.google.com/citations?user=MCV_U08AAAAJ&hl=en"><b>Youssef Mohamed</b></a>, 
  <a href="https://scholar.google.com/citations?user=ic1jai8AAAAJ&hl=en"><b>Rania Elbadry</b></a><br>
  <a href="https://ae.linkedin.com/in/omar-fetouh1"><b>Omar Fetouh</b></a>,
  <a href="https://sites.google.com/view/fahadkhans"><b>Fahad Shahbaz Khan</b></a>,
  <a href="https://salman-h-khan.github.io/"><b>Salman Khan</b></a>
</p>

<p align="center"><b>MBZUAI</b> · <b>Australia National University</b> . <b>Linköping University</b></p>


---

## 🆕 Latest Updates
- 📢 **May 2025**: Training code is released.

## Table of Contents
- [💡 TL;DR](#tldr)
- [📊 Key Results](#key-results)
- [🧠 How It Works](#how-it-works)
- [📦 Installation](#installation)
- [🚀 Quick Start](#quick-start)
- [⚙️ Training Configuration](#training-configuration)
- [📈 Evaluation](#evaluation)
- [📚 Citation](#citation)

<p align="center">
<img src="assets/cepo-main.png" alt="Accuracy over training steps" width=600/>
</p>


## TL;DR

In RLVR training (e.g., GRPO), every token in a correct trajectory gets the same reward, whether it's a decisive reasoning step or grammatical filler. **CEPO** fixes this by asking a contrastive question at each token: *does the correct answer favor this token **while** the wrong answer disfavors it?* This is done by replacing the single-reference evidence ratio $P_T^+ / P_S$ (used in RLSD) with a contrastive ratio $P_T^+ / P_T^-$, where the wrong-answer teacher $P_T^-$ is constructed from rejected rollouts already in the batch — **zero additional sampling cost**.

## Key Results

CEPO achieves **43.43%** and **60.56%** average accuracy across five multimodal math reasoning benchmarks at 2B and 4B scale, versus **41.17%** and **57.43%** for GRPO under identical training budgets.

<p align="center">
<img src="assets/cepo-teaser.png" alt="Accuracy over training steps" width=500/>
</p>

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
| Optimizer | AdamW (lr = 5e-6, cosine decay, 5-step warmup) |
| Evidence weight $\lambda_0$ | 0.5 |
| $\lambda$ decay | Linear → 0 over $T_{\text{warm}} = 25$ steps |
| Evidence clip $\epsilon_w$ | 0.5 |
| Positive reference $r^+$ | Ground truth answer |
| Negative reference $r^-$ | Rejected rollout (answer only) |
| Teacher source | Actor policy (shared weights) |

## Evaluation

We evaluate on five held-out multimodal mathematical reasoning benchmarks using [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval):

- **[DynaMath](https://arxiv.org/abs/2411.00836)**: Dynamic visual math reasoning
- **[LogicVista](https://arxiv.org/abs/2407.04973)**: Multimodal logical reasoning in visual contexts
- **[MathVision-mini](https://arxiv.org/abs/2407.14352)**: Multimodal mathematical reasoning
- **[MMMU](https://arxiv.org/abs/2311.16502)**: Massive multi-discipline multimodal understanding
- **[WeMath](https://arxiv.org/abs/2407.01284)**: Mathematical reasoning for LMMs

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
@article{heakl2026cepo,
  title={CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization},
  author={Heakl, Ahmed and Shaker, Abdelrahman M. and Mohamed, Youssef and Elbadry, Rania and Fetouh, Omar and Khan, Fahad Shahbaz and Khan, Salman},
  journal={arXiv preprint arXiv:2605.19436},
  year={2026}
}
```


## Acknowledgements

This project is built on [EasyR1](https://github.com/hiyouga/EasyR1) (a fork of [veRL](https://github.com/volcengine/verl)). We thank all the authors for providing such a high-performance RL training framework.
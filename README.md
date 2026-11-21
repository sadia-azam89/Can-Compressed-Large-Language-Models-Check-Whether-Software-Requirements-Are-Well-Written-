# Can Compressed Large Language Models Check Whether Software Requirements Are Well Written?

Code and notebooks for the paper **"Can Compressed Large Language Models Check Whether Software Requirements Are Well Written?"**, submitted to ICIDST 2026.

**Authors:** Sadia Azam¹, Akhtar Jamil¹, Seemab Zafar Janjua², Imran Javed²
¹ FAST-NUCES Islamabad, Pakistan
² NUML Rawalpindi Campus, Pakistan

---

## Overview

This work builds on **NICE** (Rejithkumar & Anish, ICSE-SEIP 2025), which fine-tunes small language models to **identify** whether a software requirement is functional and/or non-functional, **classify** its non-functional requirement (NFR) categories, and **explain** the decision in natural language. We treat this identify–classify–explain task as a *requirements quality check* and ask whether **compressed** models can perform it without a meaningful loss in quality.

The repository contains two notebooks:

1. **A reproduction of NICE** on the public PROMISE (relabeled-NICE) dataset.
2. **The main experiments of the paper**: knowledge distillation, INT8 quantization, classical and encoder baselines, project-wise evaluation with confidence intervals, a data-leakage study, and explanation-quality analysis.

Every model reads a requirement and generates one structured output:

```
functional: yes
quality: yes
categories: Performance; Security
explanation: ...
```

## Repository structure

```
.
├── 01_NICE_reproduction.ipynb              # Reproduction of the original NICE pipeline (Flan-T5-base)
├── 02_compressed_models_experiments.ipynb  # Main experiments reported in the paper
├── data/
│   └── README.md                           # How to obtain the PROMISE (relabeled-NICE) dataset
├── results/
│   └── README.md                           # Where to place the exported result tables
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Dataset

We use the **PROMISE (relabeled-NICE)** dataset released by the NICE authors (Zenodo, DOI [10.5281/zenodo.14590935](https://doi.org/10.5281/zenodo.14590935), CC-BY-4.0). It contains 622 requirements from 15 projects, each labelled as functional and/or quality, with multi-label NFR categories. The dataset is **not redistributed here**; see [`data/README.md`](data/README.md) for download instructions.

## Notebooks

### `01_NICE_reproduction.ipynb`: NICE reproduction

Reproduces the NICE pipeline: fine-tuning `google/flan-t5-base` on an 80/20 stratified split (497 train / 125 test). Optional QLoRA fine-tuning of Phi-3-mini and Llama 3.1 8B is included but switched off by default. Explanation targets can be generated with Gemini (the original paper used GPT-4o).

**Run on:** Google Colab with a GPU runtime. Optionally store a `GEMINI_API_KEY` in Colab Secrets.

### `02_compressed_models_experiments.ipynb`: main experiments

| Component | Setting |
|---|---|
| Reference model | `google/flan-t5-base` (fine-tuned) |
| Teacher | `google/flan-t5-large` |
| Student (distilled) | `google/flan-t5-small` (KD α = 0.5, T = 2.0), plus a no-KD ablation |
| Compression | Post-training dynamic INT8 quantization |
| External baselines | TF-IDF + logistic regression; `bert-base-uncased` multi-label classifier |
| Explanation targets | Written by `openai/gpt-oss-120b` via Groq from the gold labels (used only as training targets, never as model input) |
| Evaluation | Project-wise outer folds (no project appears in both training and test), validation-only model selection, cluster bootstrap 95% CIs (resampling projects, B = 1000), Holm-corrected paired tests |
| Model selection | Smallest model that is non-inferior to the reference on validation data (pre-specified margin δ = 0.03) |
| Additional analyses | Random vs project-wise split leakage study, per-class and per-project results, CPU/GPU latency, ROUGE-L / BERTScore / label-consistency for explanations |

**Composite score** = mean of FR-identification F1, NFR-identification F1, micro-F1 and macro-F1 over NFR categories.

**Run on:** Kaggle (or Colab) with a GPU (tested on an NVIDIA Tesla T4) and Internet enabled.

1. Upload `PROMISE-relabeled-NICE.csv` as a Kaggle input (Add Input → Upload).
2. Store your Groq API key in Kaggle Secrets (or Colab Secrets) under the name `GROQ_API_KEY`. A free key is available at [console.groq.com](https://console.groq.com). Never paste the key into the notebook.
3. Run all cells from top to bottom. One fold with one seed takes about 45 minutes on a T4.
4. Result tables are written to `nice_conference_efficient/export/`.

## Main results

Held-out test results pooled over 5 project-wise folds (617 requirements, 15 projects, training seed 42). Values in brackets are 95% cluster-bootstrap confidence intervals.

| Model | Size (MB) | Composite score | Micro-F1 | Δ score vs reference |
|---|---:|---|---|---|
| Teacher: Flan-T5-large (fp32) | 3132.8 | 0.748 [0.713, 0.772] | 0.625 [0.575, 0.675] | +0.074 [+0.033, +0.110] |
| BERT-base classifier | 438.0 | 0.699 [0.664, 0.732] | 0.581 [0.543, 0.631] | +0.025 [−0.012, +0.062] |
| **Reference: Flan-T5-base (fp32)** | 990.4 | 0.674 [0.649, 0.698] | 0.536 [0.505, 0.573] | — |
| **Flan-T5-base (INT8)** | **509.3** | **0.676 [0.649, 0.697]** | 0.545 [0.516, 0.576] | **+0.001 [−0.010, +0.010]** |
| Flan-T5-small, no KD (fp32) | 307.9 | 0.561 [0.509, 0.607] | 0.431 [0.387, 0.474] | −0.113 [−0.172, −0.057] |
| Flan-T5-small, KD (fp32) | 307.9 | 0.547 [0.499, 0.598] | 0.414 [0.354, 0.475] | −0.127 [−0.178, −0.072] |
| Flan-T5-small, KD (INT8) | 201.1 | 0.535 [0.478, 0.580] | 0.362 [0.286, 0.420] | −0.140 [−0.196, −0.084] |
| TF-IDF + logistic regression | 0.5 | 0.529 [0.494, 0.562] | 0.412 [0.340, 0.493] | −0.146 [−0.185, −0.114] |

Key findings:

- **INT8 quantization of Flan-T5-base halves the model size (990 MB → 509 MB) with no measurable loss** (Δ = +0.001, 95% CI [−0.010, +0.010]); it was selected as the deployment model in every fold under the pre-specified non-inferiority rule.
- The distilled Flan-T5-small student is clearly worse than the reference, and knowledge distillation did not significantly improve it over training without KD (Δ = −0.014, 95% CI [−0.075, +0.059]).
- Evaluating with a **random split instead of a project-wise split inflates the composite score by +0.167** (95% CI [+0.132, +0.204]) for the TF-IDF baseline, showing why project-wise evaluation matters.

See the paper for the full results, per-class tables and discussion.

## Notes and limitations

- **Reproduction notebook explanations.** The saved run of `01_NICE_reproduction.ipynb` was executed without a Gemini key, so it used simple template explanations. Its classification results are valid, but its explanation ROUGE-L score is not meaningful.
- **Seeds.** The reported run of `02_compressed_models_experiments.ipynb` uses a single training seed (42), so seed-to-seed variability is not estimated.
- **Explanations are post-hoc rationales.** The explanation targets were written by an LLM from the gold labels. ROUGE-L and BERTScore measure how closely a model imitates these references, not whether its reasoning is correct.
- **Single dataset.** All experiments use PROMISE; generalisation to other datasets is not tested here.
- Trained model weights are not included because of their size.

## Citation

If you use this code, please cite our paper (citation details will be updated after publication):

```bibtex
@inproceedings{azam2026compressed,
  title     = {Can Compressed Large Language Models Check Whether Software Requirements Are Well Written?},
  author    = {Azam, Sadia and Jamil, Akhtar and Janjua, Seemab Zafar and Javed, Imran},
  booktitle = {ICIDST 2026},
  year      = {2026}
}
```

Please also cite the original NICE paper and dataset:

```bibtex
@inproceedings{rejithkumar2025nice,
  title     = {{NICE}: Non-Functional Requirements Identification, Classification, and Explanation Using Small Language Models},
  author    = {Rejithkumar, G. and Anish, P. R.},
  booktitle = {Proceedings of the IEEE/ACM 47th International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP)},
  year      = {2025}
}

@dataset{rejithkumar2025nicedata,
  author    = {Rejithkumar, G. and Anish, P. R.},
  title     = {{NICE}: Non-Functional Requirements Identification, Classification, and Explanation Using Small Language Models [Dataset]},
  publisher = {Zenodo},
  year      = {2025},
  doi       = {10.5281/zenodo.14590935}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). The PROMISE (relabeled-NICE) dataset is distributed by its authors under CC-BY-4.0.

## Contact

Sadia Azam, FAST-NUCES Islamabad. ORCID: [0009-0001-3348-1424](https://orcid.org/0009-0001-3348-1424)

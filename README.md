# The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2604.02178)

This repository contains the code for the paper _"The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level"_.

<details><summary>Abstract</summary>

> Mixture-of-Experts (MoE) architectures have become the dominant choice for scaling Large Language Models (LLMs), activating only a subset of parameters per token. While MoE architectures are primarily adopted for computational efficiency, it remains an open question whether their sparsity makes them inherently easier to interpret than dense feed-forward networks (FFNs). We compare MoE experts and dense FFNs using `k`-sparse probing and find that expert neurons are consistently less polysemantic, with the gap widening as routing becomes sparser. This suggests that sparsity pressures both individual neurons and entire experts toward monosemanticity. Leveraging this finding, we _zoom out_ from the neuron to the expert level as a more effective unit of analysis. We validate this approach by automatically interpreting hundreds of experts. This analysis allows us to resolve the debate on specialization: experts are neither broad domain specialists (e.g., biology) nor simple token-level processors. Instead, they function as fine-grained task experts, specializing in linguistic operations or semantic tasks (e.g., closing brackets in LaTeX). Our findings suggest that MoEs are inherently interpretable at the expert level, providing a clearer path toward large-scale model interpretability.
</details>

---

## TL;DR

This repo provides scripts to analyze interpretability in Mixture-of-Experts (MoE) language models.

- MoE experts are more **monosemantic** than dense FFNs.
- We provide **k-sparse probing**, **automatic expert labeling**, and **specialization analysis** scripts.
- Includes scripts to reproduce all results from the paper.

## Requirements

- To run any scripts you need to have **uv** installed. If you don't want to install it you can also try to install dependencies manually using **pip**.
- Python `>= 3.13`
- Most experiments run on 2 consumer GPUs (e.g., RTX 3080 Ti) using smaller MoE models.
- The LLM pipeline (`auto.py`) uses the Gemini API and requires an API key.

## Getting Started

- Install dependencies using `uv sync`
- Dependencies are pinned; other versions may also work but are untested."
- Check out the options that can be passed to a certain script: `uv run script_name.py --help`.
- The scripts for reproducing the experiments can be found at the top level.
- The `data/` folder contains pre-computed experimental results and does not include model weights or datasets.
- See [Reproduce.md](Reproduce.md) for step-by-step instructions to reproduce each figure in the paper.
- Questions? Open an issue or contact Jeremy Herbst at <jeremy.herbst111@gmail.com>.

## Data

The `data` folder contains:

- The prompts, natural language labels and scores of the Automatic Interpretability experiment from Section 5 (JSON files).
- The generated test cases  from Section 5.3 (JSON files).
- The probing results from Section 4 for all 12 models (CSV files).
- The clustering results x from Section 6.2 as a mapping from cluster id to tokens (JSON files).
- The figures used in our paper (PDF files).

## License

- MIT

## Citation

```tex
 @misc{herbst2026expertstrikesbackinterpreting,
      title={The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level},
      author={Jeremy Herbst and Jae Hee Lee and Stefan Wermter},
      year={2026},
      eprint={2604.02178},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={<https://arxiv.org/abs/2604.02178}>,
}
```

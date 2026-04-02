# Reproducing Results

This guide provides commands to reproduce all experiments and figures from the paper:

> "The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level"

Each section corresponds to a section in the paper and produces the data used for specific figures. The `--help` flag can be helpful for additional options.

⚠️ Note: Results may slightly vary due to randomness, hardware differences, and GPU nondeterminism.

## Compute Requirements

- Probing: ~2–12 hours per model (depends mostly on the GPUs)
- Interpretability (LLM-based): ~30 Minutes per model (depends mostly on the GPUs)
- Clustering: ~30 minutes (depends mostly on the CPU)

## Workflow Overview

To reproduce results:

1. Run experiments (e.g., probing, automatic interpretability, specialization)
2. Save output files (CSV / JSON / NPZ)
3. Use `plot.py` to generate figures (use `-f` to specify Figure number)

Each section below follows this pattern.

## Probing MoE Experts and Dense FFNs

Reproduction of probing results from Section 4. For example, to run this on _allenai/OLMoE-1B-7B-0125_:

````sh
uv run probe.py -m allenai/OLMoE-1B-7B-0125 -c all
````

Note that this will produce a single _CSV_ file which contains the probing metrics for all layers.

### Models

In our paper we ran the probing experiment for these models:

- allenai/OLMoE-1B-7B-0125 (MoE)
- allenai/OLMo-1B-0724-hf (Dense)
- allenai/OLMo-7B-0724-hf (MoE)
- baidu/ERNIE-4.5-21B-A3B-Base-PT (MoE)
- meta-llama/Llama-3.2-3B (Dense)
- Qwen/Qwen3-30B-A3B (MoE)
- openai/gpt-oss-20b (MoE)
- Qwen/Qwen3-4B-Base (Dense)
- mistralai/Ministral-3-3B-Base-2512 (Dense)
- zai-org/GLM-4.7-Flash (MoE)
- mistralai/Mixtral-8x7B-v0.1 (MoE)

### Plots

 Plotting the results (Figure 1, 2, 3, 4, 8, 9). Figure 1, 2, 8 and 9 need pairs of models (MoE vs. Dense), we matched models based on active parameter count.

````sh
uv run plot.py -f 1 --pairs path1 path2 --pairs path3 path4 --pairs path5 path6 --pairs path7 path8
````

````sh
uv run plot.py -f 2 --pairs path1 path2 --pairs path3 path4 --pairs path5 path6 --pairs path7 path8
````

````sh
uv run plot.py -f 3 --files path1 path2 path3
````

````sh
uv run plot.py -f 4 --files path1 path2 ...
````

````sh
uv run plot.py -f 8 --pairs path1 path2 --pairs path3 path4
````

````sh
uv run plot.py -f 9 --pairs path1 path2 --pairs path3 path4
````

## Automatic Interpretability of MoE Experts

⚠️ Requires Gemini API key.

Reproduction of Automatic Interpretability labels and scores from Section 5.1. For example, to run this on _allenai/OLMoE-1B-7B-0125_ (Layer 7):

````sh
uv run auto.py -m allenai/OLMoE-1B-7B-0125 -l 7 -n 40
````

Note that this will produce a single _JSON_ file which contains full prompts, labels, binary scores and metrics (e.g., F1 score) for all experts in layer 7.

### Models

In our paper we ran this experiment for:

- allenai/OLMoE-1B-7B-0125 (Layers 1, 4, 7, 9, 11, 13, 14, 15)
- baidu/ERNIE-4.5-21B-A3B-Base-PT (Layers 4, 15, 25)
- Qwen/Qwen3-30B-A3B (Layers 4, 24, 44)

### Plots

Plotting the results (Figure 5):

````sh
uv run plot.py -f 5 --files path1 path2 ...
````

## Trigger-Target Experiment (Causal attribution)

Reproduction of causal attribution experiment in Section 5.3. Use the test cases in the `data` folder (or use your own). For example, to run this on _allenai/OLMoE-1B-7B-0125_ (Layer 4):

````sh
uv run causal.py -m allenai/OLMoE-1B-7B-0125 -l 4 -e 0 11 12 25 30 46 48 53 55 62
````

This will produce a _npz_ file containing the scores and ranks for each expert as numpy arrays.

### Models

We ran this experiment for _allenai/OLMoE-1B-7B-0125_:

- Layer 4 (Experts 0, 11, 12, 25, 30, 46, 48, 53, 55, 62)
- Layer 9 (Experts 5, 14, 15, 17, 19, 23, 33, 37, 56, 60)
- Layer 14 (Experts 0, 1, 2, 6, 21, 37, 38, 55, 58, 59)

### Plots

````sh
uv run plot.py -f 6 --files path1 path2 ...
````

## Expert Specialization

Reproduction of the expert specialization experiment in Section 6.2.
For example, to run this on _allenai/OLMoE-1B-7B-0125_:

````sh
uv run cluster.py -m allenai/OLMoE-1B-7B-0125 
````

which will produce a _npz_ file containing a mapping from cluster ID to vocabulary ID and a JSON file for each `k` (cluster granularity of _k_-means) which contains the mapping from cluster ID to token strings.

Then run

````sh
uv run spec.py -m allenai/OLMoE-1B-7B-0125 
````

which will save a _npz_ file containing the specialization scores.

### Plots

````sh
uv run plot.py -f 7 --files path1
````

## Additional Plots

From the Appendix (requires Probing data):

````sh
uv run plot.py -f 10 --files path1 path2 ...
````

# Legal Clause Simplifier using RLHF

This repository hosts the SageMaker-friendly implementation of the **Legal Clause
Simplifier** project. The end goal is to fine-tune a language model that rewrites
legal clauses into plain English while preserving critical entities. The
workflow follows a staged approach:

1. Prepare the CUAD dataset and bootstrap simplified references.
2. Train a supervised baseline model.
3. Optimise the model further using PPO-based RLHF.
4. Evaluate quality improvements and surface a Streamlit demo.

The current commit establishes the project scaffold so each stage can be
implemented incrementally.

## Repository layout

```
legal-simplifier-rlhf/
├── README.md
├── requirements.txt
├── app.py
├── configs/
│   ├── ppo.yaml
│   └── sft.yaml
├── data/
│   ├── examples/
│   ├── processed/
│   └── raw/
└── scripts/
    ├── data_prep.py
    ├── evaluate.py
    ├── ppo_train.py
    ├── reward_fn.py
    ├── sft_train.py
    └── utils.py
```

Each Python script currently exposes a minimal CLI entry-point to simplify
SageMaker job integration. Functional logic will be populated in subsequent
steps.

## Next steps

* Populate `configs/sft.yaml` and `configs/ppo.yaml` with concrete hyperparameters.
* Implement CUAD ingestion and simplification generation in `scripts/data_prep.py`.
* Fill in training and evaluation logic.
* Replace the placeholder Streamlit message with live inference against the PPO
  checkpoint.

> ⚠️ **Important:** No legal advice is provided. The forthcoming model is a
> research prototype intended for educational purposes only.

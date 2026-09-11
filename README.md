# LLM Fine-Tuning Lab

## Objective

Develop practical judgment in supervised fine-tuning (SFT), preference optimization, reinforcement learning, evaluation, alignment, and monitoring. Compare interventions against a prompt-only baseline, investigate failures, and use evidence to decide what to try next.

## Current result — September 11, 2026

- Setup notebook: [01_stack_setup.ipynb](01_stack_setup.ipynb).
- Model: `google/gemma-3-1b-it` (Gemma 3 1B Instruct).
- Hardware observed: NVIDIA Tesla T4, approximately 14.6 GiB GPU memory.
- Hugging Face authentication, model download, and first inference completed successfully.
- The model classified the account-access feedback example as **Negative**, with an explanation. This is a single setup smoke test, not an accuracy evaluation.
- Experiment configuration saved in Google Drive at `MyDrive/llm-finetuning-lab/experiments/exp-001-smoke-test.json`. The configuration file has not yet been committed to this repository.
- Dataset version label: `feedback-v1`. This is a planned label in the configuration; it does not establish that a dataset has been created or validated.
- No fine-tuning run or held-out evaluation has been completed yet.

## Setup in Google Colab

[Open the setup notebook in Colab](https://colab.research.google.com/github/ArielSmoliar/llm-finetuning-lab/blob/main/01_stack_setup.ipynb)

Run cells individually. The saved notebook includes troubleshooting cells from the first session, so use the order below rather than **Run all**.

1. Sign in to Google and Hugging Face. Visit the [Gemma model page](https://huggingface.co/google/gemma-3-1b-it) and review and accept its access agreement if required.
2. In Colab, select **Runtime → Change runtime type → GPU**. Check that `torch.cuda.is_available()` returns `True`.
3. Install the lab packages, including the compatibility constraints used to resolve the first session's dependency conflicts:

```python
%pip install -U "transformers>=5.10.1" datasets accelerate evaluate bitsandbytes trl "peft>=0.19.0" huggingface_hub sentencepiece "tensorboard~=2.20.0" "protobuf>=5.29.1,<6"
```

These ranges incorporate the observed fix, but do not lock every dependency. Check installation output for new conflicts; package upgrades can change compatibility. Do not rerun the older unconstrained install cell after applying the fix.

4. Select **Runtime → Restart session**, then run the notebook's import/version verification cell. A restart clears Python variables and loaded models, even when Drive remains mounted.
5. Mount Drive and recreate the project variable after the restart:

```python
from google.colab import drive
from pathlib import Path
import json

drive.mount('/content/drive')
PROJECT_DIR = Path('/content/drive/MyDrive/llm-finetuning-lab')
for folder in ('data', 'experiments', 'outputs', 'checkpoints'):
    (PROJECT_DIR / folder).mkdir(parents=True, exist_ok=True)
print('Project folder:', PROJECT_DIR)
```

Use the same Drive path in later sessions. An “already mounted” message is normal; no forced remount is needed.

6. Create a Hugging Face **read** token in [Access Tokens](https://huggingface.co/settings/tokens). Store it in Colab's **Secrets** panel with the name `HF_TOKEN`, and enable **Notebook access**. Run the notebook authentication cell. Never put the token value in code or outputs.
7. Load Gemma and run the inference cell. Use the following precision selection in that cell:

```python
import torch

dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported(including_emulation=False)
    else torch.float16
)
```

The T4 uses FP16 on this path. The default BF16 check can include emulation and report `True` on a T4; the native-support check above returned `False` in the completed session.

8. Run the configuration cell after `PROJECT_DIR` has been defined. Saving a configuration does not start training.
9. Save the notebook to Drive, then use **File → Save a copy in GitHub** to save a version to `ArielSmoliar/llm-finetuning-lab`, branch `main`, path `01_stack_setup.ipynb`. Future Colab changes do not automatically sync to GitHub. Saving the notebook does not upload separate configuration files or checkpoints.

### Versions observed in the successful setup

| Package | Version |
|---|---|
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.17.0 |
| Datasets | 5.0.1 |
| PEFT | 0.20.0 |
| TRL | 1.13.0 |
| Accelerate | 1.15.0 |
| TensorBoard | 2.20.0 |
| Protobuf | 5.29.6 |

Imports and GPU checks passed, and Gemma generated a response. This does not establish compatibility of all later training APIs. Record installed versions again for each training experiment.

## Planned experiment configuration

| Setting | Value |
|---|---|
| Experiment ID | `exp-001-smoke-test` |
| Model ID | `google/gemma-3-1b-it` |
| Dataset version label | `feedback-v1` |
| Seed | 42 |
| Maximum sequence length | 512 |
| LoRA rank / alpha | 16 / 16 |
| Learning rate | 0.0002 |
| Epochs | 1 |

## What belongs in GitHub

Track notebooks, lightweight experiment configurations, data cards, evaluation scripts, and small result tables. Suggested locations as those artifacts are created:

- `experiments/`: lightweight JSON configurations.
- `data/`: data cards and small synthetic or approved public samples; document source, labels, splits, limitations, and version.
- `evals/`: evaluation scripts.
- `results/`: small metrics tables and selected non-sensitive failure examples.
- `reports/`: experiment reports using **Question → Prediction → Experiment → Results → Failure analysis → Decision → Next experiment**.

Keep Hugging Face tokens, Colab credentials, raw private customer data, downloaded model weights, and large checkpoints out of GitHub. Keep checkpoints in Drive. Review notebook source and outputs before saving publicly.

The `.gitignore` excludes common local secret and artifact paths. It cannot remove secrets embedded inside a tracked notebook, stop a direct GitHub upload, or untrack files already committed.

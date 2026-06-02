# Model Capability Alignment Notes

`model_capabilities.csv` is keyed by the experiment `family` values in
`deception_benchmark/config.py`.

Direct matches:

| family | source row |
|--------|------------|
| `kimik2` | supplied `kimi-k2` row |
| `gptoss20b` | supplied `gpt-oss-20b` row |
| `mistralsmall32` | supplied `mistral-small-3.2-24b` row |

Proxy matches:

| family | proxy basis |
|--------|-------------|
| `glm46` | supplied `glm-4.5` GCI, with GLM-4.6 external metrics noted |
| `gptoss120b` | official OpenAI model-card MMLU scaling from supplied `gpt-oss-20b` GCI |
| `qwen32b` | Qwen3 Technical Report MMLU-Redux values |
| `deepseekr1qwen8b` | official DeepSeek model-card reasoning benchmarks |
| `qwen8b` | Qwen3 Technical Report MMLU-Redux values |

These scores are suitable as robustness covariates.  The primary model should
still include attacker/defender family fixed effects so that proxy capability
scores are not carrying the identification argument.

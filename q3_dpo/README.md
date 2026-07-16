# Part 1.3 


## Setup and commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python run_experiment.py --data data/toy_preferences.jsonl --beta 0.5
```

## Test result
```

........                                                                 [100%]
8 passed in 0.08s
```

## Toy experiment result

```
Loaded 12 preference pairs
beta: 0.5
mean DPO loss: 0.6371
preference accuracy from DPO logits: 0.583

Per-example diagnostics:
ex01: logit= 0.350, loss= 0.533, prompt=Explain why RLHF can be unstable.
ex02: logit= 0.250, loss= 0.576, prompt=What is Direct Preference Optimization?
ex03: logit=-0.050, loss= 0.718, prompt=Summarize the main advantage of pairwise preference training.
ex04: logit= 0.400, loss= 0.513, prompt=Why keep a reference policy in DPO?
ex05: logit= 0.300, loss= 0.554, prompt=What does beta control in DPO?
ex06: logit=-0.075, loss= 0.731, prompt=Define a chosen response in preference data.
ex07: logit= 0.300, loss= 0.554, prompt=Explain why scalar reward RL can be difficult for language models.
ex08: logit=-0.150, loss= 0.771, prompt=What is a rejected response?
ex09: logit= 0.350, loss= 0.533, prompt=Why might DPO be easier to implement than PPO-based RLHF?
ex10: logit=-0.125, loss= 0.758, prompt=What is the intuition behind the DPO objective?
ex11: logit=-0.250, loss= 0.826, prompt=What does a negative DPO margin suggest?
ex12: logit= 0.250, loss= 0.576, prompt=How does DPO use preference pairs?

```

## Interpretation

A positive DPO logit means that, relative to the reference policy, the current policy favors the chosen answer more strongly than the rejected answer. The reference policy acts as an anchor: the objective rewards preference improvement beyond the baseline instead of merely maximizing the chosen answer's raw probability.

"beta" scales the policy-reference difference. Increasing it makes the sigmoid loss react more strongly to the same relative preference gap, while a smaller value softens that distinction.

"ex04" has the lowest loss (0.513) because its logit is the most positive (0.400), so the current policy already improves strongly on the reference policy's chosen-versus-rejected preference. Example ex11 has the highest loss (0.826) because its logit is the most negative (-0.250), meaning the current policy favors the chosen response less strongly than the reference does and would need the strongest corrective update.

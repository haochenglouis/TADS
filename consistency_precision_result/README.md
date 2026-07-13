# Annotation Quality Evaluation Results

This directory contains the raw evaluation records for the LLM annotation quality study
(paper Appendix D.2, "Examining the LLM Annotation Quality", Table 8), following the
evaluation framework of Lu et al. (2023, #INSTAG).

We assess the generated proxy labels along two axes:

- **Precision** — whether a label correctly describes its corresponding instruction.
- **Consistency** — whether the same label is applied to semantically similar instructions.

## Files

| File | Records | Description |
|---|---|---|
| `precision_gpt_eval.xlsx` | 500 | Randomly sampled instruction–label pairs; GPT-4 judges whether the label adequately describes the instruction (`message`, `task`, `gpt score`). |
| `precision_human_eval.xlsx` | 50 | The first 50 pairs from the GPT-4 precision sample, independently rated by three human annotators. |
| `consistency_gpt_eval.xlsx` | 500 | Randomly sampled labels, each associated with at least two distinct instructions; GPT-4 judges whether the instructions under a given label are semantically consistent (`message1`, `message2`, `task`, `gpt score`). |
| `consistency_human_eval.xlsx` | 50 | The first 50 label sets from the GPT-4 consistency sample, independently rated by three human annotators. |

Scores are binary (1 = precise/consistent, 0 = not). Aggregated results and
human–GPT agreement (Cohen's Kappa) / inter-annotator agreement (Fleiss' Kappa)
are reported in Table 8 of the paper.

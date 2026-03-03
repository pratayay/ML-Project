# Unified Dataset Data Spec

This document defines how source moderation datasets map into a common ontology and risk target for AdaptiveGuard training.

## Unified schema

Each normalized training row must include:

- `text`: original content text used for modeling.
- `category`: shared ontology category.
- `risk_target`: normalized float in `[0, 1]`.
- `dataset_name`: source dataset identifier.
- `original_label`: source label before mapping.
- `mapping_version`: semantic version for the mapping rules.

## Shared ontology

The shared category space is:

- `safe`
- `abuse_harassment`
- `self_harm`
- `violence_threat`
- `hate_identity`

## Source dataset mappings

### `civil_comments`

- Source label field: `toxicity_label`
- Source text field: `comment_text`

| Source label | Unified category |
|---|---|
| `non_toxic` | `safe` |
| `toxic` | `abuse_harassment` |
| `threat` | `violence_threat` |
| `identity_attack` | `hate_identity` |

### `moderation_v2`

- Source label field: `moderation_tag`
- Source text field: `text`

| Source label | Unified category |
|---|---|
| `ok` | `safe` |
| `harassment` | `abuse_harassment` |
| `self-harm` | `self_harm` |
| `violence` | `violence_threat` |
| `hate` | `hate_identity` |

### `incident_reports`

- Source label field: `incident_type`
- Source text field: `narrative`

| Source label | Unified category |
|---|---|
| `benign` | `safe` |
| `abusive_language` | `abuse_harassment` |
| `suicide_instruction` | `self_harm` |
| `violent_planning` | `violence_threat` |
| `protected_group_slur` | `hate_identity` |

## Risk-target computation policy

`risk_target` is computed per row using observed dataset signals (not fixed constants):

1. Use `annotator_agreement` when available (`[0,1]`).
2. Use `empirical_harm_rate` when available (`[0,1]`).
3. Blend available observed signals with category-level severity priors.
4. If no observed signal exists, use severity prior for that category.

Current severity priors:

- `safe`: `0.02`
- `abuse_harassment`: `0.58`
- `self_harm`: `0.86`
- `violence_threat`: `0.91`
- `hate_identity`: `0.82`

## Validation gate before training

Before training, run inter-dataset validation over unified rows:

- Compare each dataset mean `risk_target` against global mean.
- Compare each dataset category distribution against global distribution using total variation distance (TVD).
- Flag extreme mismatch if:
  - `mean_gap > 0.20`, or
  - `tvd > 0.35`.

The validation script exits non-zero on mismatch so training can be blocked until data quality issues are addressed.

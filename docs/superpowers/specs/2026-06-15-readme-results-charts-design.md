# README Results Charts Design

## Goal

Expand the existing `Results (Phase 1, Synthetic)` section in `README.md` so all 14 generated evaluation charts are available with concise explanations of what they show and why each result matters.

## Placement

Keep the current summary metrics table, model comparison table, inference-latency statement, and synthetic-data warning. Insert the chart narrative after the model comparison table and before the final synthetic-data warning so readers move from headline metrics to supporting evidence and then to the limitation statement.

## Presentation

Use repository-relative Markdown image paths rooted at `data/figures/charts-images/`. Keep the principal evaluation charts visible inline. Put each wide feature-family histogram in a separate HTML `<details>` block with a descriptive `<summary>` so the README remains easy to scan.

Each chart or chart group will have a short explanation covering:

- what is being measured or visualized;
- the main result visible in the chart;
- why that result is important for model quality, sensor design, or deployment confidence.

## Chart Organization

### Dataset Quality

1. `class_distribution.png`: show that the 26 static letter classes are balanced at 70 samples each, reducing class-frequency bias during training and evaluation.
2. `tsne_all_classes.png`: explain that most classes form distinct clusters in the learned feature space, supporting the separability of the 26-feature representation while highlighting limited overlap and outliers.
3. `confused_pairs.png`: discuss focused t-SNE views for M/N, D/K, and P/Q; note that M/N and D/K are well separated while P/Q is closer and therefore deserves additional monitoring.

### Static Classifier Performance

4. `confusion_matrix_xgb.png`: present the 97.4% XGBoost result and explain that the strong diagonal indicates consistently correct predictions, with residual errors concentrated in a small number of visually similar signs.
5. `per_class_f1_xgb.png`: show that precision, recall, and F1 remain above the 0.80 threshold for every letter, demonstrating that aggregate accuracy is not hiding a failed class.

### Dynamic Classifier Performance

6. `dynamic_confusion_matrix.png`: explain the perfect diagonal on the synthetic dynamic test set for J, Z, and auxiliary motion classes, while clearly retaining the broader synthetic-data caveat.

### Interpretability And Sensor Contribution

7. `shap_importance_xgb.png`: identify finger flexion, pitch, FSR, touch, and derived openness features among the strongest contributors, showing that the classifier uses multiple sensor modalities rather than one isolated channel.
8. `ablation_chart.png`: explain that removing flex or touch causes the largest drops, all reduced models remain above the 70% minimum, and derived features can be removed with little loss because they largely summarize raw channels.

### Robustness

9. `noise_curve.png`: show that accuracy stays above the 85% target across the tested additional-noise range, supporting tolerance to moderate synthetic perturbation while avoiding claims about untested real-world noise.

### Feature Distribution Appendix

Add five independent collapsible sections:

10. `hist_flex.png`: finger-bend distributions reveal strong class-dependent hand-shape information.
11. `hist_fsr.png`: pressure distributions capture fingertip and pad contact patterns that help separate otherwise similar signs.
12. `hist_touch.png`: mostly discrete touch states encode finger-contact relationships and provide high-value categorical distinctions.
13. `hist_orientation.png`: roll, pitch, and yaw distributions capture wrist and hand pose; overlap shows why orientation is most useful in combination with other modalities.
14. `hist_derived.png`: openness and inter-hand deltas summarize coordinated two-hand geometry and offer interpretable aggregate features.

Each `<details>` block will contain a one-paragraph interpretation followed by its full-width image.

## Constraints

- Do not change reported metric values or imply that synthetic results prove real-world performance.
- Do not move, rename, regenerate, or alter the supplied PNG files.
- Do not rewrite unrelated README sections.
- Use GitHub-compatible Markdown and HTML only.
- Preserve the README's existing style and heading hierarchy.

## Verification

After editing:

1. Confirm every supplied chart filename appears exactly once in `README.md`.
2. Confirm all referenced image paths resolve to existing files.
3. Confirm there are exactly five `<details>` and five closing `</details>` tags.
4. Review the resulting diff for unrelated changes and accidental metric modifications.
5. Inspect the rendered README structure or, if direct rendering is unavailable, validate Markdown ordering and HTML tag balance from the source.

# Critical Review: Sycophancy Flip Prediction Project

---

## Overview Assessment

This is a well-constructed project that asks a genuinely novel question — which linguistic properties of a user's rebuttal predict whether a model will sycophantically flip? The experimental pipeline is methodologically careful in several respects: the grouped cross-validation prevents question-level leakage, the robust/brittle filter is principled, and the SHAP triangulation across three classifier paradigms is the right instinct for an interpretability-first project. But the project has a structural problem that no amount of feature engineering will fix, and several design choices that are suppressing the signal you're looking for.

---

## 1. Is the problem formulation right?

Partially. Predicting FLIP/HOLD from rebuttal-side features only is a clean experimental design, but it may be *too* clean — you've controlled away a major source of variance that the literature says matters.

Xie et al. (2024) showed that LLMs are "adaptive chameleons" when external evidence is coherent and convincing, but "stubborn sloths" when the external evidence contains information *consistent* with parametric memory. This means the interaction between the rebuttal and the model's internal confidence is likely a first-order predictor. A rebuttal that introduces a plausible-sounding alternative fact should have very different flip-inducing power depending on whether Haiku's parametric knowledge about that question is strong or weak. By excluding target-response features entirely, you're treating the model's confidence as noise rather than signal.

Concretely: Haiku's initial response almost certainly contains hedging cues ("I believe," "this is likely"), response length, and structural confidence markers that are predictive. The five "rebuttal-vs-initial diff" features you already compute (hedging, certainty, modal uncertainty, sentiment, length differences) partially address this, but they're a narrow slice. You should include the full set of initial-response linguistic features as a separate feature block, and then test whether they contribute incremental AUC in an ablation. This is not a design compromise — it's testing a more complete model of the interaction. The question is not "do rebuttal features predict flipping?" in isolation, but "given a (rebuttal, response) pair, which properties of the rebuttal matter?" That's still your research question; you're just conditioning on the other side of the interaction.

**Recommendation:** Add a full initial-response feature block. Report the ablation (rebuttal-only → response-only → combined). If response features dominate, that's itself a finding: it would mean sycophancy is primarily a function of model confidence, not persuasion quality.

---

## 2. What's a realistic AUC ceiling?

Your AUC of 0.665 is roughly in line with what comparable behavioral-prediction tasks achieve, but the framing matters.

Laban et al. (2024) found that the FlipFlop effect — models flipping after a simple "Are you sure?" — varies enormously by task domain and model, with flip rates between 20% and 80%. Critically, they found that even the *domain* of the classification task (e.g., sentiment vs. NLI) and the presence of an "authoritative persona" in the challenger systematically shifted flip rates. But they did not attempt to predict per-instance flips from rebuttal features; they documented aggregate effects. Similarly, Fanous et al. (2025) in SycEval found sycophancy rates of ~58% overall, with significant variation by rebuttal type (simple vs. citation-based), but again at the aggregate level, not per-instance prediction.

No published work I've found attempts your exact task — training a classifier to predict per-instance flip from rebuttal linguistic features — so there is no direct comparison point. The closest analogue is predicting individual LLM outputs from input features, which is a notoriously noisy task because LLM behavior is stochastic (temperature, sampling) and context-sensitive in ways that surface features don't capture.

Here is the hard truth: your AUC ceiling is probably fundamental, not a bug. You are trying to predict, from ~120 surface-level features of a short text, the behavior of a neural network with billions of parameters that is processing the entire conversation context including the question, its own answer, and the rebuttal jointly. The irreducible noise floor comes from: (a) the model's internal representations, which encode question difficulty, knowledge confidence, and instruction-following tendencies that no rebuttal feature can observe; (b) generation stochasticity; and (c) the question-level random effect, which your filter reduces but doesn't eliminate. An AUC of 0.65–0.70 from surface features alone is plausible as a ceiling. You should frame this honestly: the contribution is the SHAP analysis showing *which* features matter most, not the raw predictive performance.

**Realistic target:** 0.68–0.72 with the additions I suggest. Above 0.75 would require internal model signals (logits, attention patterns), which are outside your scope.

---

## 3. The embedding-only collapse

The AUC = 0.599 for embedding-only is *not* a reliable indicator that sentence-transformer embeddings lack relevant signal. There are at least two confounds.

First, the PCA compression is aggressive. Retaining only 60 of 384 components (65% variance) is discarding a third of the information in a space where the discriminative signal may live in the later principal components. PCA maximizes variance, not discriminative power — the components that distinguish FLIP from HOLD may have low total variance but high class-conditional variance. This is a well-known limitation when PCA is used as a preprocessing step for classification.

Second, all-MiniLM-L6-v2 is a relatively weak sentence-transformer. It's fast and small (6 layers, 22M parameters), but its representational capacity is limited compared to all-mpnet-base-v2 (12 layers, 109M parameters), which Reimers & Gurevych (2019) and subsequent benchmarks consistently show to produce higher-quality embeddings. The embedding may simply lack the resolution to capture the pragmatic and rhetorical features that matter for this task.

**What you should do, in order of expected impact:**

1. Test full 384-dim embedding with no PCA. If AUC improves meaningfully over 0.599, the issue was PCA compression, not embedding quality. This takes 30 minutes and settles the question.
2. If full 384-dim helps, try a supervised dimensionality reduction (e.g., PLS or an MLP bottleneck) instead of PCA. This would project onto dimensions that discriminate FLIP/HOLD, not dimensions of maximum variance.
3. Upgrade to all-mpnet-base-v2. This is a stronger embedding and may capture more of the pragmatic signal. But do step 1 first — if MiniLM at full 384-dim doesn't help, mpnet probably won't either, and the issue is that generic sentence embeddings don't encode what you need.
4. Skip the MLP head idea for now. Adding a trainable head reintroduces the interpretability problem you moved away from, and with 4,800 samples you'll overfit unless the architecture is very shallow.

---

## 4. The "lifeless" SHAP rankings

This is the most important finding in your project, and I think you're misreading it. The absence of pragmatic/epistemic-stance features from the top SHAP rankings is probably a combination of two issues — one methodological and one empirical — and disentangling them is critical.

**The methodological issue: lexicon-based feature extraction is too crude.** Hedging, certainty, authority claims, and aggression are complex pragmatic phenomena. Detecting them with substring matching against small hardcoded phrase lists is a known limitation. Leidinger et al. (2023) showed that even carefully designed linguistic prompts for feature extraction have limited coverage. Your hedging detector probably catches "I think" and "perhaps" but misses hedging expressed through syntactic subordination ("It might be worth considering whether..."), evidential marking ("Some researchers have suggested..."), or prosodic cues encoded in punctuation patterns. The same applies to certainty and authority.

Consider replacing your lexicon-based detectors with LLM-based feature extraction. Use a small model (e.g., Claude Haiku or GPT-4o-mini) to rate each rebuttal on a 1–5 scale for hedging, certainty, aggression, authority, emotional pressure, logical coherence, and specificity. This is closer to how Sharma et al. (2024) generated text-level features for their preference model analysis. The cost is modest for ~5,000 rebuttals, and it would give you continuous, contextual measurements instead of binary lexicon matches.

**The empirical issue: generator diversity.** You use Gemini 3 Flash and Gemma 3 27B with four prompting templates. Both are Google-family models trained on overlapping data, and the diversity prompts may not be producing genuinely varied rhetorical strategies. If all your rebuttals cluster in a narrow band of "polite-but-firm academic disagreement," then hedging, aggression, and authority claims will have low variance in your data, and low-variance features can't be predictive regardless of whether they matter in principle.

You can test this directly: compute the coefficient of variation for each pragmatic feature. If hedging_count has CV < 0.3 across your dataset, the rebuttals simply aren't varying enough along that axis for your classifier to learn anything.

**The genuine empirical possibility.** It's also possible that structural/surface features (clause count, punctuation, length) genuinely matter more than rhetoric for triggering sycophancy. Sclar et al. (2024) showed that meaning-preserving formatting changes can shift LLM performance by up to 76 accuracy points. If formatting and structure influence behavior more than content, then finding that clause counts beat hedging would be consistent with this literature. But you need to rule out the methodological confounds first before claiming this as a finding.

**What Fanous et al. (2025) found is relevant here:** in SycEval, simple rebuttals maximized progressive sycophancy, while citation-based rebuttals triggered the highest regressive sycophancy. This suggests that the *type* of rhetorical strategy matters — but in a categorical way (simple vs. evidence-based), not necessarily through the continuous pragmatic features you're measuring. Consider adding a categorical "rebuttal strategy" feature (classified by an LLM) alongside or instead of the continuous pragmatic features.

---

## 5. Single-target generalization

Yes, you should test on a second target before publishing SHAP rankings as findings. This is not optional — it's the difference between "linguistic properties that predict Haiku 3.5's sycophancy" and "linguistic properties that predict sycophancy."

Sharma et al. (2024) showed that sycophancy is consistent across five state-of-the-art assistants, which suggests some generalization is plausible. But Laban et al. (2024) found substantial between-model variance in flip rates and sensitivity to challenger properties. Your SHAP rankings could reflect Haiku-specific quirks (e.g., Haiku may be unusually sensitive to clause complexity because of its training mix) rather than general mechanisms.

**Which model to test on:** Use an open-weights model where you can control for API-level nondeterminism. Llama 3 8B-Instruct is a good choice: it's a different architecture (Meta vs. Anthropic), different training pipeline, widely used, and cheap to run. Avoid testing on another Anthropic model (which would share training methodology) or a model that's too large to run practically. You don't need to regenerate rebuttals — just re-run the existing rebuttals against the new target and re-label with your judge pipeline.

**What to report:** Compare the top-20 SHAP feature sets between Haiku and the second model. If the rankings are similar (e.g., Spearman ρ > 0.5), that's evidence for generalization. If they diverge, that's still interesting — you'd be showing that different models are sycophantic for different reasons.

---

## 6. Robust/brittle filter

The filter is methodologically defensible, and a reviewer would accept the rationale, but the framing needs adjustment.

The filter removes questions where the outcome is almost entirely determined by the question itself (near-0% or near-100% flip rate), on the grounds that rebuttal features can't explain variance when there's no variance. This is standard practice in item-response theory: items with extreme difficulty are uninformative for discriminating among test-takers. The analogue holds here.

However, a reviewer will raise two points. First, the scope restriction: your findings apply to "questions where the model is persuadable but not trivially so." You need to state this explicitly and discuss what it means. Questions with near-0% flip rate are ones where Haiku has strong parametric knowledge; questions with near-100% flip rate are ones where Haiku is highly uncertain. Your model is only making claims about the middle ground, and that's fine, but it's a meaningful limitation.

Second, the threshold sensitivity: [0.10, 0.90] is somewhat arbitrary. Report a sensitivity analysis with [0.05, 0.95] and [0.20, 0.80] and show that your main SHAP rankings are robust to the threshold choice. If they aren't, that's a red flag. If they are, it's one paragraph and one table that preempts the reviewer's concern.

---

## 7. Ranking of proposed experiments

Here is my ranking by expected impact on the project, accounting for both predictive improvement and interpretability contribution.

**Tier 1 — Do these before submission:**

1. **Target-response features as classifier input.** This is your single highest-value experiment. You already have the data (Haiku's initial responses). Extract the same 61 linguistic features from the initial response, add them as a feature block, and run your pipeline. I predict this will push AUC from 0.665 to ~0.69–0.72 and will reveal that model-side hedging and uncertainty are among the top SHAP features, which would be a much more compelling finding than clause counts.

2. **LLM-based pragmatic feature extraction** (replacing lexicon-based detectors). Have an LLM rate each rebuttal on hedging, certainty, aggression, authority, emotional pressure, logical coherence, and evidence quality on continuous scales. This directly addresses the "lifeless SHAP" problem and could rescue the proposal's core hypothesis.

3. **Full 384-dim embedding test** (no PCA). Takes 30 minutes, settles the embedding question.

**Tier 2 — Do one of these for the submission; the other for a revision:**

4. **Second target model** (Llama 3 8B-Instruct). Essential for generalization claims. Run existing rebuttals against the new target. Compare SHAP rankings.

5. **Sbert upgrade to all-mpnet-base-v2.** Only if the full 384-dim MiniLM test (item 3) shows that embedding quality matters. Otherwise skip.

**Tier 3 — Future work, not needed for initial submission:**

6. **More PCA components (60 → 150).** Only relevant if full 384-dim helps and you want to find the minimal sufficient dimensionality. Low priority.

7. **Counterfactual rebuttal generation.** Intellectually interesting but methodologically complex and hard to validate. This is a separate paper.

8. **TCAV-style attribution on a fine-tuned transformer.** This is the right long-term direction for the project, but it's a substantial engineering effort. The idea would be to fine-tune DistilBERT on FLIP/HOLD, then define concept datasets for hedging, aggression, authority, etc. (collections of rebuttals that exemplify each concept), and use TCAV (Kim et al., 2018) to test whether the model's internal representations are sensitive to these concepts. This would bypass the feature-engineering bottleneck entirely. But it requires careful concept dataset curation and is probably a second paper.

---

## 8. Published methods we're missing

Several methods from the literature are directly relevant and not incorporated:

**Probing classifiers on the target model's internal representations.** Chen et al. (2024) in the pinpoint tuning work used path patching (Wang et al., 2022) to identify specific attention heads responsible for sycophancy. While you don't have access to Haiku's internals, for your open-weights second target model (e.g., Llama), you could extract intermediate-layer representations and train probing classifiers to predict FLIP/HOLD. This would tell you whether the model's internal state at the point where it processes the rebuttal is predictive — and if so, which layers encode the "decision to flip." This is more informative than surface features and directly connects to the mechanistic interpretability literature.

**Rebuttal typology from Fanous et al. (2025).** SycEval distinguishes between simple, logical, citation-based, and authoritative rebuttals and finds that these categories predict sycophancy differentially. You should add a categorical rebuttal-type feature (classified automatically by an LLM judge) and test whether it's predictive above your continuous features.

**Confirmation bias from Xie et al. (2024).** Their finding that LLMs show strong confirmation bias — being more receptive to external evidence that partially aligns with their parametric memory — suggests a feature you haven't considered: the degree of partial overlap between the rebuttal's claims and Haiku's initial response. A rebuttal that acknowledges part of Haiku's answer while redirecting the conclusion may be more flip-inducing than one that flatly contradicts everything. You could operationalize this as the cosine similarity between the rebuttal and the *correct-answer portion* of Haiku's response (as opposed to just sim_to_wrong and sim_to_correct).

**Multi-turn persistence from SycEval.** Fanous et al. found 78.5% persistence of sycophantic behavior once triggered. You're treating each (question, rebuttal) pair as independent, but if you're running multiple rebuttals per question sequentially against the same conversation, ordering effects could confound your labels. Confirm that each rebuttal is tested in a fresh conversation context.

---

## 9. Methodological holes a reviewer would flag

**The LLM-as-a-judge reliability.** You label FLIP/HOLD/AMBIGUOUS using Claude Sonnet 4.6 as a judge. Zheng et al. (2023) showed that LLM judges have systematic biases including position bias and verbosity bias. But the more specific concern here is that your judge is an Anthropic model labeling the behavior of another Anthropic model. A reviewer will ask: (a) what's the inter-annotator agreement between the LLM judge and human annotators on a random subset? and (b) is there a systematic bias where Sonnet is more or less likely to label certain response types as FLIP because of shared training? You need to report human agreement on at least 200 randomly sampled instances. If you haven't done this, do it before submission.

**Synthetic rebuttal authenticity.** All your rebuttals are generated by LLMs (Gemini, Gemma), not by humans. A reviewer will ask whether LLM-generated rebuttals cover the same distribution of rhetorical strategies that real users employ. The concern is ecological validity: real users might use emotional appeals, personal anecdotes, appeal to consequences, or simply repeat their claim more loudly — strategies that LLM generators may underrepresent due to their own alignment training. Consider including a small set of human-written rebuttals (even 100–200) as a calibration check. If the classifier trained on synthetic rebuttals transfers to human rebuttals with comparable AUC, that's a strong validity argument.

**Multiple comparisons in SHAP.** You're interpreting the top-20 SHAP features from 121 candidates across three classifier paradigms. A reviewer will ask whether the "interpretive triangulation" is principled or post-hoc. Report which features are in the top-20 for *all three* classifiers, not just LogReg. If the three classifiers disagree substantially on which features matter, the SHAP rankings are classifier-dependent and less interpretable. Quantify the agreement (e.g., Jaccard similarity of top-20 sets across the three paradigms).

**Class imbalance handling is inconsistent.** LogReg and SVM use class_weight="balanced," XGBoost uses scale_pos_weight, and MLP uses neither (relying on threshold tuning). This makes cross-classifier comparisons difficult — you're comparing models that are solving slightly different objectives. Standardize: use cost-sensitive weighting for all classifiers and threshold tuning for all classifiers, then compare.

**No calibration analysis.** You report F1 and AUC but not calibration (Brier score, reliability diagrams). A model with AUC 0.665 could be well-calibrated (predicted probabilities match observed frequencies) or poorly calibrated (overly confident on wrong predictions). Calibration matters if anyone wants to use these predictions operationally.

**The 6-teammate merge.** The data was distributed across 6 teammates and merged. Were the generation parameters, prompting templates, and labeling criteria exactly identical across all slices? Any variation introduces batch effects that could confound your features. Report whether teammate identity predicts FLIP/HOLD after controlling for question — if it does, you have a batch effect.

---

## 10. Should you publish or keep iterating?

Keep iterating, but not for long — you're one round of experiments away from a publishable paper.

What you have now is a carefully engineered pipeline with a modest but honest negative result: surface-level linguistic features predict sycophantic flipping only weakly (AUC ~0.66), and the features that do predict are structural rather than rhetorical. That's a contribution, but it's a disappointing one relative to the proposal's ambition, and a reviewer will ask whether the ceiling is real or whether you under-measured the interesting variables.

The minimum additional work for a defensible submission is: (1) add target-response features and report the ablation, (2) replace lexicon-based pragmatic detectors with LLM-rated features and re-run the pipeline, (3) test full 384-dim embeddings, (4) validate your LLM judge against human annotations on a subset, and (5) test on one additional target model. Items 1–3 are engineering work that can be done in a week. Item 4 requires some human annotation effort. Item 5 requires re-running the rebuttal pipeline, which is mostly compute time.

If the target-response features push AUC above 0.70 and the LLM-rated pragmatic features show up in the SHAP rankings, the headline finding becomes: "Sycophantic flipping is primarily predicted by the interaction between the model's initial confidence and the structural complexity of the rebuttal, while rhetorical strategy (authority claims, aggression, hedging) plays a secondary but measurable role." That's a solid contribution to the sycophancy literature that would be appropriate for EMNLP Findings, ACL Findings, or a relevant workshop.

If nothing changes — the new features don't help, AUC stays at 0.66, and the SHAP rankings remain dominated by clause counts — then the honest headline is: "Surface-level linguistic features are weak predictors of per-instance sycophantic flipping, suggesting that the decision to flip is primarily determined by internal model states (question-level confidence, parametric knowledge strength) rather than by the persuasive properties of the rebuttal." That's still publishable as a negative result with an important implication for sycophancy mitigation: if the *rebuttal's rhetoric* doesn't much matter, then sycophancy is less about the user being persuasive and more about the model being uncertain. That reframes the problem in a way that matters for alignment work.

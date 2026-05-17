# The 60 Rebuttal-Only Linguistic Features

All 60 features are computed **from the rebuttal text alone**, with no access
to the question, the correct/wrong answer choices, or the target model's
initial response. This is the strict feature set used for the
"predict sycophancy from how the attacker writes" framing.

Two feature groups present in [FEATURES.md](FEATURES.md) are intentionally
**excluded** here:

- **Group A — Semantic similarity (2 features):** `sim_to_wrong`,
  `sim_margin_wrong_minus_correct`. These require the correct/wrong answer
  strings to compute cosine similarity against the rebuttal embedding.
- **Group B — Rebuttal-vs-initial diff (5 features):** `diff_n_hedging`,
  `diff_n_certainty`, `diff_modal_uncertainty`, `diff_vader_compound`,
  `diff_len_words`. These require the target model's `initial_answer`
  text to compute style deltas.

The remaining 60 features depend only on the rebuttal string.

Tools used: plain Python + regex, `textstat` (readability), `vaderSentiment`
(sentiment), `spaCy` (POS tagging, dependency parse, morphology, NER).
No sentence-embedding components.

## 1. Orthographic — 9
Surface measurements of the rebuttal as written text.

| # | Feature | What it counts |
|---|---|---|
| 1 | `len_words` | Number of whitespace-separated tokens |
| 2 | `n_sentences` | Sentence count (spaCy `doc.sents`) |
| 3 | `avg_word_len` | Mean characters per word (whitespace excluded) |
| 4 | `n_question_marks` | Count of `?` |
| 5 | `n_exclamations` | Count of `!` |
| 6 | `n_commas` | Count of `,` |
| 7 | `caps_ratio` | Uppercase chars / total chars |
| 8 | `starts_lower` | 1 if first character is lowercase, else 0 |
| 9 | `has_digit` | 1 if rebuttal contains a digit, else 0 |

## 2. Typography / emphasis — 4
Stylistic emphasis markers.

| # | Feature | What it counts |
|---|---|---|
| 10 | `n_all_caps_words` | Words of length ≥ 2 entirely uppercase |
| 11 | `n_repeated_punctuation` | Runs of 2+ consecutive `!`, `?`, or `.` (e.g. `!!`, `???`) |
| 12 | `has_emoji` | 1 if any Unicode emoji codepoint present |
| 13 | `n_quoted_strings` | Substrings inside straight or curly quotes |

## 3. Lexical (readability + diversity) — 2

| # | Feature | What it measures |
|---|---|---|
| 14 | `flesch_reading_ease` | Flesch Reading Ease score (higher = easier) |
| 15 | `type_token_ratio` | Unique words / total words (lexical diversity) |

## 4. Pragmatic — lexicon counts — 8
Substring count of phrases from a small hardcoded set per category.

| # | Feature | Examples of the lexicon |
|---|---|---|
| 16 | `n_hedging` | "maybe", "perhaps", "might", "I think", "I'm not sure" |
| 17 | `n_certainty` | "definitely", "absolutely", "100%", "certainly", "obviously" |
| 18 | `n_politeness` | "please", "could you", "thank you", "kindly" |
| 19 | `n_aggression` | "wrong", "incorrect", "you're wrong", "ridiculous", "lol", "smh" |
| 20 | `n_sources` | "study", "research", "scientists", "Wikipedia", "according to" |
| 21 | `n_anecdote` | "my teacher", "my mom", "when I was", "I learned" |
| 22 | `n_authority` | "experts say", "scientists agree", "everyone knows" |
| 23 | `n_command` | "check your facts", "look it up", "verify", "Google it" |

## 5. Pragmatic — discourse & person reference — 6

| # | Feature | What it measures |
|---|---|---|
| 24 | `discourse_contrast_markers` | Count of "actually", "but", "however", "in fact", "nevertheless"… |
| 25 | `n_intensifiers` | Count of "very", "really", "extremely", "totally", "super"… |
| 26 | `first_person_ratio` | Count of {I, my, me, I'm, …} / total words |
| 27 | `second_person_ratio` | Count of {you, your, you're, …} / total words |
| 28 | `third_person_ratio` | Count of {he, she, it, they, his, her, …} / total words |
| 29 | `is_question_only` | 1 if rebuttal ends with `?` and contains no `.` or `!` |

## 6. Pragmatic — discourse moves (hypothesis-driven) — 4
Patterns added based on a priori sycophancy hypotheses.

| # | Feature | What it captures |
|---|---|---|
| 30 | `n_concede_markers` | "yes but", "you're right but", "fair point", "I agree but" — concede-then-redirect |
| 31 | `has_initial_reframe` | 1 if rebuttal starts with "actually", "but", "however", "wait", "no" |
| 32 | `n_rhetorical_question_markers` | "have you considered", "isn't it", "don't you think", "what if" |
| 33 | `n_numerical_expressions` | Count of numeric tokens (digits, percentages, years, units) — specificity/evidence proxy |

## 7. Pragmatic — data-grounded curated — 2
Patterns observed during a qualitative scan of FLIP rebuttals, retained
after empirical filtering (3 others tested and dropped after mean-|SHAP| < 0.005).

| # | Feature | What it captures |
|---|---|---|
| 34 | `n_ellipsis` | Count of `…` and `...` — hesitant/musing tone |
| 35 | `n_asterisk_emphasis` | Count of `*word*` — Markdown-style stress |

## 8. Semantic — sentiment — 3
VADER polarity scores for the rebuttal.

| # | Feature | Range |
|---|---|---|
| 36 | `vader_pos` | Positive sentiment intensity, [0, 1] |
| 37 | `vader_neg` | Negative sentiment intensity, [0, 1] |
| 38 | `vader_compound` | Overall polarity, [-1, 1] |

## 9. Morphological — 10
Properties derived from spaCy's POS tagger and morphological features
on each token.

| # | Feature | What it measures |
|---|---|---|
| 39 | `n_modals_certain` | Modal verbs in {must, will, shall, would} |
| 40 | `n_modals_uncertain` | Modal verbs in {may, might, could, can, should} |
| 41 | `modal_uncertainty_ratio` | `n_modals_uncertain / max(1, n_modals_all)` |
| 42 | `modal_density` | Modal verbs per word |
| 43 | `n_past_tense` | Past-tense verb count |
| 44 | `n_present_tense` | Present-tense verb count |
| 45 | `past_pres_ratio` | Past-tense share of tensed verbs |
| 46 | `n_nominalisations` | Nouns ending in -tion, -ment, -ness, -ity, -ance, -ence, -sion, -ship |
| 47 | `nominalisation_rate` | Nominalisations / total nouns |
| 48 | `n_negations` | Tokens with dep label `neg` |

## 10. Syntactic — 11
Properties derived from spaCy's dependency parse.

| # | Feature | What it measures |
|---|---|---|
| 49 | `n_clauses` | Finite-verb count (tag VBD/VBP/VBZ/MD) |
| 50 | `clauses_per_sentence` | `n_clauses / n_sentences` |
| 51 | `mean_parse_depth` | Mean depth of each token in the parse tree |
| 52 | `n_passive_aux` | Tokens with dep `auxpass`/`aux:pass` |
| 53 | `n_passive_subj` | Tokens with dep `nsubjpass`/`csubjpass`/`*:pass` |
| 54 | `is_passive` | 1 if any passive marker present |
| 55 | `n_subord_clauses` | Tokens with dep `ccomp`, `xcomp`, `advcl`, `acl`, `acl:relcl` |
| 56 | `subord_rate` | `n_subord_clauses / n_sentences` |
| 57 | `n_conjunctions` | Tokens with dep `cc` or `conj` |
| 58 | `mean_sentence_len` | Mean tokens per sentence |
| 59 | `sentence_len_var` | Variance of per-sentence token counts |

## 11. Named entities — 1

| # | Feature | What it measures |
|---|---|---|
| 60 | `n_named_entities` | Count of `doc.ents` from spaCy's NER (specificity proxy) |

---

## Summary by proposal linguistic level

| Proposal level | Features | Count |
|---|---|---|
| Orthographic | #1–13 | 13 |
| Lexical | #14–15 | 2 |
| Pragmatic (lexicons + discourse + curated) | #16–35 | 20 |
| Semantic (sentiment only) | #36–38 | 3 |
| Morphological | #39–48 | 10 |
| Syntactic | #49–59 | 11 |
| Named entities | #60 | 1 |
| **Total** | | **60** |

## Excluded features (vs the 67-feature set)

| Group | # | Feature | Why excluded |
|---|---|---|---|
| A | 61 | `sim_to_wrong` | Requires `wrong_answer` text |
| A | 62 | `sim_margin_wrong_minus_correct` | Requires `correct_answer` and `wrong_answer` text |
| B | 63 | `diff_n_hedging` | Requires `initial_answer` text |
| B | 64 | `diff_n_certainty` | Requires `initial_answer` text |
| B | 65 | `diff_modal_uncertainty` | Requires `initial_answer` text |
| B | 66 | `diff_vader_compound` | Requires `initial_answer` text |
| B | 67 | `diff_len_words` | Requires `initial_answer` text |

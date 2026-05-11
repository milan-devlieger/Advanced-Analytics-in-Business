Final academic notebook: assignment1_clv.ipynb
Final academic submission: ../data/processed/submission.csv
Best public leaderboard candidate: ../data/processed/leaderboard_candidates/submission_scale_1030.csv
Best public score so far: 61.717
Baseline public score: about 61.719
Failed candidates:
- quantile + zero threshold: 61.781
- quantile alpha 0.50 no zero threshold: 61.821
- zero-lift: no candidate written because OOF did not improve
- scale 1.040 and 1.050: no improvement
Note: future leaderboard experiments must be created in one new scratch notebook only and must not modify assignment1_clv.ipynb.


Failed later scratch experiments:
- model diversity: no useful blend; written candidates were only anchor scale variants.
- targeted feature groups: all engineered feature groups were worse than reduced baseline on OOF.
- weighted LightGBM: positive/high-value weighting worsened OOF MAE.
- Current best public candidate remains data/processed/leaderboard_candidates/submission_scale_1030.csv with score 61.717.
Target-encoding/isotonic scratch experiment rejected: scratch baseline did not reproduce the final OOF baseline and target encoding/isotonic degraded OOF strongly.
Current best public candidate = submission_scale_1030.csv, score 61.717
Rejected candidate = submission_exact_feature_repair_rank01.csv, score 61.762
Rejected reason = OOF feature gain did not transfer to public leaderboard.

Final selected submission:
- file: data/processed/submission.csv
- equivalent preserved candidate: data/processed/leaderboard_candidates/submission_scale_1030.csv
- public score: 61.717
- selected model path: reduced features + LightGBM raw MAE + final scale 1.030

Rejected experiment families:
- quantile objective / no-zero candidate: worsened public score
- zero-lift candidates: did not improve OOF enough / no reliable candidate
- broad targeted feature groups: OOF/public did not transfer sufficiently
- weighted LightGBM: worsened OOF
- model diversity / blends: did not beat anchor meaningfully
- target encoding / isotonic calibration attempt: rejected due to unstable or broken validation behavior
- exact baseline feature repair: OOF gain did not transfer to public leaderboard
- further public scale probing around 1.0275 and 1.0325 did not beat 1.030


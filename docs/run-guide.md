# Running the research pipelines

## Extended pipeline

Install dependencies from the repository root, prepare the cleaned data, then change into `strategy02`. Run these commands in order using the root virtual environment:

```powershell
cd strategy02
..\.venv\Scripts\python p1_data_costs.py
..\.venv\Scripts\python p2_regimes_stochastic.py
..\.venv\Scripts\python p3_features.py
..\.venv\Scripts\python p4_ml_models.py
..\.venv\Scripts\python p5_deep_learning.py
..\.venv\Scripts\python p6_portfolio_backtest.py
..\.venv\Scripts\python p7_robustness_report.py
..\.venv\Scripts\python p8_train_test_split.py
..\.venv\Scripts\python p9_daily_ml_overlay.py
..\.venv\Scripts\python p10_research_improvements.py
..\.venv\Scripts\python p11_edge_builder.py
```

Later phases consume earlier phase outputs. The common parser supports `--quick` for reduced experiments; it is not a substitute for full validation. Check each script's help for phase-specific options, such as model trials or epochs. No execution-time estimates are promised.

| Phase | Focus |
|---|---|
| 1 | Data, costs, baselines and shared utilities |
| 2 | Market regimes and stochastic models |
| 3 | Features and labels |
| 4 | Machine-learning challengers |
| 5 | Deep-learning challengers |
| 6 | Portfolio simulation |
| 7 | Robustness and reports |
| 8 | Train/test split evaluation |
| 9 | Daily exposure and trade-quality overlays |
| 10 | Rebalancing frequency, residual and consensus momentum |
| 11 | Portfolio-rule ablations |

## Baseline pipeline

From the repository root, change into `strategy` and run its phase 1 through phase 8 files in numerical order using the same virtual environment. Do not mix its generated files with those from `strategy02`: the pipelines share filenames but use different implementations.

Phase 6 of the extended pipeline optionally reads `strategy/phase6_portfolio_backtest/daily_V3.csv` to draw a baseline MIS overlay comparison. That output is not bundled; run the baseline if you want that comparison.

## Outputs and verification

Phases generate local `phase*/` directories containing CSV, JSON, Markdown and charts. These outputs are excluded from version control. Start with phase reports and train/test comparisons, checking costs, dates and benchmarks before interpreting metrics. Repeated runs may replace existing output files.

The publication checks parsed all included Python source and screened the selected text for credential patterns and known local credential values. They did not execute these pipelines, install the dependencies or validate financial results.

# Symbolic Recurrence Quantification with Macro‑Conditioned Thresholds

Applies recurrence quantification analysis (RQA) to ETF returns, with the recurrence threshold dynamically adjusted by macro variables (VIX, DXY, yields). The per‑ETF score is the recurrence rate of the last state – a measure of structured predictability under current macro conditions.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- State space reconstruction (embedding dimension E=3, τ=1)
- Macro‑conditioned threshold: base quantile × (1 + γ × macro_factor)
- Macro factor computed from ridge regression weights on all available macro variables
- Score = recurrence rate (proportion of recurrent points)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-symbolic-rqa-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast, O(n²) per ETF)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High recurrence rate → ETF dynamics are highly structured/predictable given today's macro.
- Low recurrence rate → chaotic, regime‑sensitive.

## Requirements

See `requirements.txt`.

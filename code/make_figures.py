"""Figures for README: fig1 regime drivers (ΣA, σ_w, V_w), fig2 predicted Δ vs realised drift by year."""
import polars as pl, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
R = Path('research/tx_etf_rebalance/out'); F = Path('repos/tx-etf-rebalance/figures')
d = pl.read_parquet(R / 'daily5.parquet').sort('date')
d = d.with_columns(sig20=pl.col('db_15').rolling_std(20).shift(1), V20=pl.col('V_w').rolling_mean(20).shift(1)).with_columns(Delta=0.5 * pl.col('sig20') * (pl.col('N') / pl.col('V20')).sqrt())
x = np.array([np.datetime64(s) for s in d['date']])
BLUE, AMBER, GREEN, GRAY = '#2563EB', '#D97706', '#059669', '#9CA3AF'
fig, axs = plt.subplots(3, 1, figsize=(8, 7), dpi=150, sharex=True)
axs[0].plot(x, d['A_sum'], lw=1.5, color=BLUE); axs[0].set_ylabel('ΣA  (億 TWD)'); axs[0].set_title('Regime drivers: leveraged/inverse ETF AUM, window volatility, window volume')
axs[1].plot(x, d['sig20'], lw=1.5, color=AMBER); axs[1].set_ylabel('σ_w 20d  (points)')
axs[2].plot(x, d['V20'], lw=1.5, color=GREEN); axs[2].set_ylabel('V_w 20d  (contracts)')
for a in axs:
    a.grid(alpha=0.2); [a.spines[s].set_visible(False) for s in ['top', 'right']]
plt.tight_layout(); plt.savefig(F / 'fig1_regime.png'); plt.close()
# fig2: yearly predicted Δ (median) vs realised gross drift in sign(r) direction
yr = d.with_columns(y=pl.col('date').str.slice(0, 4)).group_by('y').agg(pred=pl.col('Delta').median(), real=pl.col('pnl').mean(), real_med=pl.col('pnl').median()).sort('y')
fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
i = np.arange(len(yr)); w = 0.38
ax.bar(i - w / 2, yr['pred'], w, color=BLUE, label='model: median Δ_t = 0.5·σ̂_w·√(N_t/V̂_w)', edgecolor='white')
ax.bar(i + w / 2, yr['real'], w, color=AMBER, label='realised: mean sign(r)·ΔF(13:30→13:45)', edgecolor='white')
ax.axhline(3.35, color=GRAY, lw=1, ls='--'); ax.text(len(yr) - 0.5, 3.6, 'cost 3.35', color=GRAY, fontsize=8, ha='right')
ax.set_xticks(i); ax.set_xticklabels(yr['y']); ax.set_ylabel('points'); ax.set_title('Predicted impact vs realised drift, by year'); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.2, axis='y')
[ax.spines[s].set_visible(False) for s in ['top', 'right']]
plt.tight_layout(); plt.savefig(F / 'fig2_pred_vs_real.png'); plt.close()
print(yr)

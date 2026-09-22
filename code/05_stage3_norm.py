"""M1 in the model's own units: ΔF/σ̂_w on sign(F)√(N/V̂_w); σ̂, V̂ = prior 20-day rolling. Plus M2 night-open detail."""
import polars as pl, numpy as np
import statsmodels.api as sm
from pathlib import Path
R = Path('research/tx_etf_rebalance/out')
d = pl.read_parquet(R / 'daily4.parquet').sort('date')
d = d.with_columns(sig15=pl.col('db_15').rolling_std(20).shift(1), V15=pl.col('V_w').rolling_mean(20).shift(1),
                   sig5=pl.col('db_5').rolling_std(20).shift(1), V5b=pl.col('V_5').rolling_mean(20).shift(1), s_dim=pl.col('r').sign() * pl.col('r').abs().sqrt() * 10)
d = d.with_columns(y15=pl.col('db_15') / pl.col('sig15'), x15=pl.col('r').sign() * (pl.col('N') / pl.col('V15')).sqrt(),
                   y5=pl.col('db_5') / pl.col('sig5'), x5=pl.col('r').sign() * (pl.col('N') / pl.col('V5b')).sqrt())
def hac(df, ycol, xcols, lag=5):
    df = df.select([ycol] + xcols).drop_nulls().filter(pl.all_horizontal(pl.all().is_finite()))
    X = sm.add_constant(df.select(xcols).to_numpy()); y = df[ycol].to_numpy()
    return sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lag}), len(df)
samples = {'全': d, '排': d.filter(~pl.col('excl')), '2021-23': d.filter(pl.col('date') < '2024-01-01'), '2024': d.filter(pl.col('date').str.starts_with('2024')), '2025': d.filter(pl.col('date').str.starts_with('2025')), '2026-01~04': d.filter((pl.col('date') >= '2026-01-01') & (pl.col('date') < '2026-05-01')), '2026-05+': d.filter(pl.col('date') >= '2026-05-01'), '2026-05+ 排': d.filter((pl.col('date') >= '2026-05-01') & ~pl.col('excl'))}
L = []; P = L.append
P('# M1 正規化版：ΔF/σ̂_w = α + Y_eff·sign(F)√(N/V̂_w) [+ sign(r)√|r| 控制]\n')
P('Y_eff 直接對應 Stage 1 的 Y（大單事件版 0.5）。若 ETF 流量的衝擊符合平方根律，Y_eff ≈ 0.5 且跨制度穩定。\n')
P('| 出場 | 樣本 | Y_eff (無控制) | t | Y_eff (控制) | t | 控制項 t | n | x 中位 √(N/V) |'); P('|---|---|---|---|---|---|---|---|---|')
for ex, yc, xc in [('13:45', 'y15', 'x15'), ('13:35', 'y5', 'x5')]:
    for sl, sub in samples.items():
        m1, n = hac(sub, yc, [xc]); m2, _ = hac(sub, yc, [xc, 's_dim'])
        P(f'| {ex} | {sl} | {m1.params[1]:.3f} | {m1.params[1]/m1.bse[1]:.2f} | {m2.params[1]:.3f} | {m2.params[1]/m2.bse[1]:.2f} | {m2.params[2]/m2.bse[2]:.2f} | {n} | {float(sub[xc].abs().median()):.3f} |')
P('')
P('## M2 補充：2026-05+ 13:45→夜盤 15:00 開盤，依 N 三分位符號調整平均（點）\n')
sub = samples['2026-05+'].with_columns(sg=pl.col('r').sign())
q1, q2 = float(sub['N'].quantile(1/3)), float(sub['N'].quantile(2/3))
P('| N 分位 | n | 13:30→13:45 | 13:45→15:00 開 | 13:45→15:30 | 13:45→次開 |'); P('|---|---|---|---|---|---|')
for nm, s_ in [('低', sub.filter(pl.col('N') < q1)), ('中', sub.filter((pl.col('N') >= q1) & (pl.col('N') < q2))), ('高', sub.filter(pl.col('N') >= q2))]:
    v = [float((s_['sg'] * s_[c]).mean()) for c in ['db_15', 'm2_gap', 'm2_night', 'm2_open']]; md = [float((s_['sg'] * s_[c]).median()) for c in ['db_15', 'm2_gap', 'm2_night', 'm2_open']]
    P(f'| {nm} | {len(s_)} | {v[0]:.1f} (中位 {md[0]:.1f}) | {v[1]:.1f} (中位 {md[1]:.1f}) | {v[2]:.1f} (中位 {md[2]:.1f}) | {v[3]:.1f} (中位 {md[3]:.1f}) |')
for sl in ['全', '排', '2025', '2026-05+']:
    m, n = hac(samples[sl], 'm2_gap', ['F_sgn']); P(f'- {sl} 13:45→15:00 開 對 sign(F)√N：β {m.params[1]:.3f} (t {m.params[1]/m.bse[1]:.2f}, n {n})')
open(R / 'stage3_norm.md', 'w').write('\n'.join(L)); print('\n'.join(L))

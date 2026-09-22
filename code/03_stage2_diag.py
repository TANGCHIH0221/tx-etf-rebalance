"""Stage 2 diagnostics: identify flow vs return-momentum via A_{t-1} time variation; placebo windows; r-hat sanity."""
import polars as pl, numpy as np, glob
import statsmodels.api as sm
from pathlib import Path
R = Path('research/tx_etf_rebalance/out'); M = 200
d = pl.read_parquet(R / 'daily2.parquet').sort('date')
tx = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob('data/gold/tx_mr/tx_1s/year=*/data.parquet'))]).filter(pl.col('session') == 'D')
mc = pl.read_parquet('data/gold/tx_mr/main_contract.parquet').select('date', 'contract')
tx = tx.join(mc, left_on='session_date', right_on='date').filter(pl.col('contract') == pl.col('contract_right')).with_columns(t=pl.col('ts').dt.time())
def vol(t0, t1, name): return tx.filter((pl.col('t') >= t0) & (pl.col('t') < t1)).group_by('session_date').agg(pl.col('vol').sum().alias(name))
for a, b, nm in [((13, 10), (13, 25), 'V_pl15'), ((13, 20), (13, 25), 'V_pl5'), ((12, 0), (12, 15), 'V_noon15')]:
    d = d.join(vol(pl.time(*a), pl.time(*b), nm), left_on='date', right_on='session_date', how='left')
tw = pl.read_parquet('data/raw/finmind_daily/taiex.parquet').select(pl.col('date').cast(pl.Utf8), I_open=pl.col('open'), I_close=pl.col('close')).sort('date')
tw = tw.with_columns(gap_next=pl.col('I_open').shift(-1) - pl.col('I_close'))
d = d.join(tw.select('date', 'gap_next'), on='date', how='left')
d = d.with_columns(s_r=pl.col('r').sign() * (pl.col('r').abs() * pl.col('I')).sqrt(),   # sign(r)·√|r| in point-scale
                   A_rel=pl.col('A_prev') / pl.col('A_prev').mean(), r_pts=pl.col('r') * pl.col('I'), abs_r_pts=(pl.col('r') * pl.col('I')).abs())
d = d.with_columns(s_rA=pl.col('s_r') * (pl.col('A_rel')).sqrt())   # = sign(r)√(|r|·A_rel·I) ∝ sign(F)√N up to θ and I

def hac(df, ycol, xcols, lag=5):
    df = df.select([ycol] + xcols).drop_nulls().filter(pl.all_horizontal(pl.all().is_finite()))
    X = sm.add_constant(df.select(xcols).to_numpy()); y = df[ycol].to_numpy()
    return sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lag}), len(df)
def fmt(m, i): return f'{m.params[i]:.3f} ({m.params[i]/m.bse[i]:.2f})'
L = []; P = L.append
P('# Stage 2 診斷\n')
# ---- P1 with r control ----
P('## P1 加 r 控制：y = β·2A_{t−1}r + γ·r_pts + δ·申贖\n')
P('| 設定 | β (t) | γ r_pts (t) | n | R² |'); P('|---|---|---|---|---|')
for lab, sub in [('全', d), ('排', d.filter(~pl.col('excl'))), ('2025+', d.filter(pl.col('date') >= '2025-01-01')), ('2026', d.filter(pl.col('date') >= '2026-01-01'))]:
    m, n = hac(sub, 'y_L', ['x_L', 'r_pts', 'inflow_L']); P(f'| 正2 {lab} | {fmt(m,1)} | {fmt(m,2)} | {n} | {m.rsquared:.3f} |')
    m, n = hac(sub, 'y_S', ['x_R', 'r_pts', 'inflow_R']); P(f'| 反1 {lab} | {fmt(m,1)} | {fmt(m,2)} | {n} | {m.rsquared:.3f} |')
P('')
# ---- P2 with |r| control and placebo windows ----
P('## P2 加 |r| 控制 + placebo 窗口：V = α + β₁N + β₂N²/1000 + γσ_t + δ|r_pts|\n')
P('| 窗口 | β₁ (t) | δ |r| (t) | n | R² |'); P('|---|---|---|---|---|')
for vc, nm in [('V_w', '13:30–13:45 (目標)'), ('V_pl15', '13:10–13:25 (placebo 15m)'), ('V_noon15', '12:00–12:15 (placebo 15m)'), ('V_5', '13:30–13:35 (目標)'), ('V_pl5', '13:20–13:25 (placebo 5m)'), ('V_pre', '13:25–13:30')]:
    m, n = hac(d, vc, ['N', 'N2', 'sig_t', 'abs_r_pts']); P(f'| {nm} | {fmt(m,1)} | {fmt(m,4)} | {n} | {m.rsquared:.3f} |')
P('\n差分版：V_w − V_pl15 對 N（同日相減，日內共同因子消掉）\n')
dd = d.with_columns(dV15=pl.col('V_w') - pl.col('V_pl15'), dV5=pl.col('V_5') - pl.col('V_pl5'))
for vc, nm in [('dV15', '15 分鐘差'), ('dV5', '5 分鐘差')]:
    for lab, sub in [('全', dd), ('排', dd.filter(~pl.col('excl')))]:
        m, n = hac(sub, vc, ['N', 'N2', 'sig_t', 'abs_r_pts']); P(f'- {nm} {lab}: β₁ = {fmt(m,1)}, β₂ = {fmt(m,2)}, n={n}, R²={m.rsquared:.3f}')
P('')
# ---- r-hat sanity ----
s = d.drop_nulls(['rhat'])
P('## r̂ = I_close − I_13:25 的性質\n')
P(f'n={len(s)}，r̂ std {s["rhat"].std():.1f} 點，ΔF(13:25→13:29:59) std {(s["p1329"]-s["p1325"]).std():.1f} 點')
m, n = hac(s.with_columns(dF=pl.col('p1329') - pl.col('p1325')), 'dF', ['rhat']); P(f'- ΔF_pre 對 r̂：β = {fmt(m,1)}, R² {m.rsquared:.3f}')
m, n = hac(s, 'gap_next', ['rhat']); P(f'- 次日開盤跳空 (I_open,t+1 − I_close) 對 r̂：β = {fmt(m,1)}, R² {m.rsquared:.3f}（負 = 收盤 surprise 隔天回吐）')
m, n = hac(s, 'db_15', ['rhat']); P(f'- ΔF(13:30→13:45) 對 r̂：β = {fmt(m,1)}')
P('')
# ---- P3/M1 identification: flow vs momentum via A variation ----
P('## 流量 vs 動能識別：ΔF = a + b₁·sign(r)√|r_pts| + b₂·sign(r)√(|r_pts|·A_rel)\n')
P('b₂ 只靠 A_{t−1} 的時序變異識別（A_rel = A_{t−1}/樣本均值）。若 13:30 後的漂移是 ETF 流量，b₂>0 且 b₁ 不需為正；若只是動能，b₁ 吸收、b₂≈0。\n')
P('| 被解釋 | 版本 | b₁ (t) | b₂ (t) | n | R² |'); P('|---|---|---|---|---|---|')
for yc, nm in [('db_pre', '13:25→13:30 basis'), ('jump_1330', '13:29:59→13:30:00'), ('db_5', '13:30→13:35'), ('db_15', '13:30→13:45')]:
    for lab, sub in [('全', d), ('排', d.filter(~pl.col('excl')))]:
        m, n = hac(sub, yc, ['s_r', 's_rA']); P(f'| {nm} | {lab} | {fmt(m,1)} | {fmt(m,2)} | {n} | {m.rsquared:.3f} |')
P('\n### 分 A 制度：ΔF 對 sign(r)√|r_pts| 單變數，係數應隨 √A 放大才是流量\n')
P('| 期間 | A_prev 均值(億) | 13:30→13:35 b (t) | 13:30→13:45 b (t) | 13:25→13:30 ΔF b (t) | n |'); P('|---|---|---|---|---|---|')
for lab, sub in [('2021', d.filter(pl.col('date').str.starts_with('2021'))), ('2022-23', d.filter((pl.col('date') >= '2022-01-01') & (pl.col('date') < '2024-01-01'))),
                 ('2024', d.filter(pl.col('date').str.starts_with('2024'))), ('2025', d.filter(pl.col('date').str.starts_with('2025'))), ('2026H1', d.filter((pl.col('date') >= '2026-01-01') & (pl.col('date') < '2026-05-01'))), ('2026-05+', d.filter(pl.col('date') >= '2026-05-01'))]:
    sub = sub.with_columns(dFpre=pl.col('p1329') - pl.col('p1325'))
    m5, n = hac(sub, 'db_5', ['s_r']); m15, _ = hac(sub, 'db_15', ['s_r']); mp, _ = hac(sub, 'dFpre', ['s_r'])
    P(f'| {lab} | {sub["A_prev"].mean():.0f} | {fmt(m5,1)} | {fmt(m15,1)} | {fmt(mp,1)} | {n} |')
d.write_parquet(R / 'daily3.parquet')
open(R / 'stage2_diag.md', 'w').write('\n'.join(L)); print('\n'.join(L))

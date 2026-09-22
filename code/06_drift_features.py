"""Which intraday paths (TAIEX via TX, TSMC) predict the 13:30->13:45 drift? Momentum framing."""
import polars as pl, numpy as np, glob
import statsmodels.api as sm
from pathlib import Path
R = Path('research/tx_etf_rebalance/out')
d = pl.read_parquet(R / 'daily4.parquet').sort('date')
tx = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob('data/gold/tx_mr/tx_1s/year=*/data.parquet'))]).filter(pl.col('session') == 'D')
mc = pl.read_parquet('data/gold/tx_mr/main_contract.parquet').select('date', 'contract')
tx = tx.join(mc, left_on='session_date', right_on='date').filter(pl.col('contract') == pl.col('contract_right')).with_columns(t=pl.col('ts').dt.time())
def pb(t0, nm): return tx.filter(pl.col('t') < t0).group_by('session_date').agg(pl.col('last').last().alias(nm))
f = tx.group_by('session_date').agg(p_open=pl.col('last').first())
for t0, nm in [(pl.time(12, 25), 'p1225'), (pl.time(12, 55), 'p1255'), (pl.time(13, 10), 'p1310'), (pl.time(13, 25), 'p1325b')]:
    f = f.join(pb(t0, nm), on='session_date')
rng = tx.filter(pl.col('t') < pl.time(13, 30)).group_by('session_date').agg(hi=pl.col('last').max(), lo=pl.col('last').min())
f = f.join(rng, on='session_date')
d = d.join(f, left_on='date', right_on='session_date', how='left')
# TSMC tick features
rows = []
for fp in sorted(glob.glob('data/raw/finmind_intraday/stock_tick/2330/*.parquet')):
    s = pl.read_parquet(fp)
    if len(s) == 0: continue
    s = s.sort('Time'); pre = s.filter(pl.col('Time') < '13:25:00')
    if len(pre) == 0: continue
    p1300 = s.filter(pl.col('Time') < '13:00:00')['deal_price']
    rows.append(dict(date=s['date'][0], ts_open=float(s['deal_price'][0]), ts_1300=float(p1300[-1]) if len(p1300) else None, ts_1325=float(pre['deal_price'][-1]), ts_close=float(s['deal_price'][-1]),
                     ts_hi=float(pre['deal_price'].max()), ts_lo=float(pre['deal_price'].min())))
ts = pl.DataFrame(rows).sort('date').with_columns(ts_prev=pl.col('ts_close').shift(1))
d = d.join(ts, on='date', how='left')
I = pl.col('I')
d = d.with_columns(
    r_day=pl.col('r') * 100,                                             # % TAIEX close/prev close
    mom_open=(pl.col('p1329') / pl.col('p_open') - 1) * 100,             # TX 08:45→13:29
    mom_60=(pl.col('p1329') / pl.col('p1225') - 1) * 100,
    mom_30=(pl.col('p1329') / pl.col('p1255') - 1) * 100,
    mom_15=(pl.col('p1329') / pl.col('p1310') - 1) * 100,
    mom_5=(pl.col('p1329') / pl.col('p1325b') - 1) * 100,
    pos_range=(pl.col('p1329') - pl.col('lo')) / (pl.col('hi') - pl.col('lo')),
    gap_open=(pl.col('p_open') / (pl.col('I') / (1 + pl.col('r'))) - 1) * 100,   # TX open vs prev TAIEX close
    rhat_pct=pl.col('rhat') / pl.col('I') * 100,
    tsmc_day=(pl.col('ts_close') / pl.col('ts_prev') - 1) * 100,
    tsmc_auc=(pl.col('ts_close') / pl.col('ts_1325') - 1) * 100,        # 2330 closing auction surprise
    tsmc_30=(pl.col('ts_1325') / pl.col('ts_1300') - 1) * 100,
    tsmc_pos=(pl.col('ts_1325') - pl.col('ts_lo')) / (pl.col('ts_hi') - pl.col('ts_lo')),
    y15=pl.col('db_15'), y14=pl.col('p14') - pl.col('p1330'), y_pl=pl.col('p1325b') - pl.col('p1310'))
d = d.with_columns(tsmc_div=pl.col('tsmc_day') - pl.col('r_day'), pos_c=pl.col('pos_range') - 0.5, tsmc_pos_c=pl.col('tsmc_pos') - 0.5)
feats = ['r_day', 'gap_open', 'mom_open', 'mom_60', 'mom_30', 'mom_15', 'mom_5', 'pos_c', 'rhat_pct', 'tsmc_day', 'tsmc_auc', 'tsmc_30', 'tsmc_pos_c', 'tsmc_div']
def hac(df, ycol, xcols, lag=5):
    df = df.select([ycol] + xcols).drop_nulls().filter(pl.all_horizontal(pl.all().is_finite()))
    X = sm.add_constant(df.select(xcols).to_numpy()); y = df[ycol].to_numpy()
    return sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lag}), len(df)
L = []; P = L.append
P('# 尾盤漂移的日內特徵（動能框架）\n')
P('目標 y15 = ΔF 13:30→13:45（點）；y14 = →13:44；placebo y_pl = 13:10→13:25。特徵皆 13:30:00 前可知，單位 %（pos 為區間位置 −0.5）。\n')
P('## 單變數：y = α + β·feature（HAC t）\n')
P('| 特徵 | β y15 | t | R² | β y14 | t | β placebo | t | n | 逐年 t 同號數 |'); P('|---|---|---|---|---|---|---|---|---|---|')
for fe in feats:
    m1, n = hac(d, 'y15', [fe]); m2, _ = hac(d, 'y14', [fe]); m3, _ = hac(d, 'y_pl', [fe])
    yr = []
    for y in range(2021, 2027):
        sub = d.filter(pl.col('date').str.starts_with(str(y)))
        try: my, ny = hac(sub, 'y15', [fe]); yr.append(np.sign(my.params[1]) == np.sign(m1.params[1]) and abs(my.params[1] / my.bse[1]) > 1)
        except Exception: pass
    P(f'| {fe} | {m1.params[1]:.2f} | {m1.params[1]/m1.bse[1]:.2f} | {m1.rsquared:.3f} | {m2.params[1]:.2f} | {m2.params[1]/m2.bse[1]:.2f} | {m3.params[1]:.2f} | {m3.params[1]/m3.bse[1]:.2f} | {n} | {sum(yr)}/{len(yr)} |')
P('')
P('## 多變數（全部特徵，需台積電 → 2022 起；rhat 另跑子樣本）\n')
fs_ = [x for x in feats if x != 'rhat_pct']; m, n = hac(d, 'y15', fs_)
P('| 特徵 | β | t |'); P('|---|---|---|')
for i, fe in enumerate(fs_): P(f'| {fe} | {m.params[i+1]:.2f} | {m.params[i+1]/m.bse[i+1]:.2f} |')
P(f'\nn={n}, R²={m.rsquared:.3f}\n')
m, n = hac(d, 'y15', ['r_day', 'mom_30', 'tsmc_auc', 'rhat_pct']); P(f'含 r̂ 子樣本 (n={n}, R² {m.rsquared:.3f})：r_day {m.params[1]:.2f} (t {m.params[1]/m.bse[1]:.1f}), mom_30 {m.params[2]:.2f} (t {m.params[2]/m.bse[2]:.1f}), tsmc_auc {m.params[3]:.2f} (t {m.params[3]/m.bse[3]:.1f}), rhat {m.params[4]:.2f} (t {m.params[4]/m.bse[4]:.1f})\n')
# ---- conditional sorts: sign(r_day) strategy split by feature terciles ----
P('## 條件分組：順 r_day 方向 13:30 進 13:45 出，毛損益（點），依特徵三分位\n')
d = d.with_columns(pnl=pl.col('r').sign() * pl.col('y15'), pnl14=pl.col('r').sign() * pl.col('y14'))
P(f'無條件：mean {d["pnl"].mean():.2f}、median {d["pnl"].median():.1f}、日 Sharpe×√252 (扣 3.35) {((d["pnl"]-3.35).mean()/d["pnl"].std()*np.sqrt(252)):.2f}、n {d["pnl"].drop_nulls().len()}\n')
P('| 特徵 | 低三分位 mean (t) | 中 | 高 | 說明 |'); P('|---|---|---|---|---|')
def tstat(s): s = s.drop_nulls(); return f'{s.mean():.1f} ({s.mean()/s.std()*np.sqrt(len(s)):.1f})'
for fe, desc in [('r_day', '|r| 用絕對值'), ('mom_30', '同號=順 r 的動能'), ('mom_5', '同號=13:25 後順 r'), ('pos_c', '同號=收在 r 那一側極端'), ('tsmc_auc', '同號=台積電收盤競價順 r'), ('tsmc_div', '同號=台積電比大盤更順 r'), ('rhat_pct', '同號=加權收盤 surprise 順 r')]:
    col = pl.col(fe).abs() if fe == 'r_day' else pl.col(fe) * pl.col('r').sign()
    s = d.with_columns(z=col).drop_nulls(['z', 'pnl']); q1, q2 = s['z'].quantile(1/3), s['z'].quantile(2/3)
    P(f'| {fe} | {tstat(s.filter(pl.col("z") < q1)["pnl"])} | {tstat(s.filter((pl.col("z") >= q1) & (pl.col("z") < q2))["pnl"])} | {tstat(s.filter(pl.col("z") >= q2)["pnl"])} | {desc} |')
P('')
P('## 逐年：順 r_day 無條件 vs |r|≥0.5% vs mom_30 同號\n')
P('| 年 | 無條件 mean (n) | |r|≥0.5% mean (n) | mom_30 同號 mean (n) | |r|≥0.5% & mom_30 同號 (n) | 前 10 天佔利潤 |'); P('|---|---|---|---|---|---|')
for y in range(2021, 2027):
    s = d.filter(pl.col('date').str.starts_with(str(y))).drop_nulls(['pnl'])
    a = s['pnl']; b = s.filter(pl.col('r').abs() >= 0.005)['pnl']; c = s.filter(pl.col('mom_30') * pl.col('r') > 0)['pnl']; e = s.filter((pl.col('r').abs() >= 0.005) & (pl.col('mom_30') * pl.col('r') > 0))['pnl']
    top = float(a.sort(descending=True).head(10).sum() / a.sum()) if a.sum() > 0 else float('nan')
    P(f'| {y} | {a.mean():.1f} ({len(a)}) | {b.mean():.1f} ({len(b)}) | {c.mean():.1f} ({len(c)}) | {e.mean():.1f} ({len(e)}) | {top:.0%} |')
d.write_parquet(R / 'daily5.parquet'); open(R / 'drift_features.md', 'w').write('\n'.join(L)); print('\n'.join(L))

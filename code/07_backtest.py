"""Gated + sized backtest of the 13:30 rebalance-impact trade. All inputs known at 13:29."""
import polars as pl, numpy as np
from pathlib import Path
R = Path('research/tx_etf_rebalance/out'); M = 200; Y = 0.5; C = 3.35
CAP = 1e7; MRATE = 0.06  # 保證金假設 = 6% 名目（TAIFEX 歷史約 5–8%）
d = pl.read_parquet(R / 'daily5.parquet').sort('date')
d = d.with_columns(sig20=pl.col('db_15').rolling_std(20).shift(1), V20=pl.col('V_w').rolling_mean(20).shift(1))
d = d.with_columns(Delta=Y * pl.col('sig20') * (pl.col('N') / pl.col('V20')).sqrt(), sg=pl.col('r').sign())
d = d.with_columns(g15=pl.col('sg') * (pl.col('p1345') - pl.col('p1330')), g14=pl.col('sg') * (pl.col('p14') - pl.col('p1330'))).drop_nulls(['Delta', 'g15', 'g14'])
L = []; P = L.append
P('# 門檻 + 部位回測（13:30 進，成本 3.35 點/口）\n')
P(f'樣本 {d["date"].min()} ~ {d["date"].max()}，{len(d)} 日。本金 {CAP/1e4:.0f} 萬（假設）、保證金 = 6% 名目（假設；2026 約 55 萬/口 → 上限約 18 口，名目槓桿 ≤ 16x）。Sharpe = 日報酬（含未交易日 0）× √252，扣成本。\n')
def stats(sub, ret, pnl_pts, x):
    n = len(sub); traded = (x > 0).sum(); r = ret
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else float('nan')
    eq = np.cumprod(1 + r); dd = (eq / np.maximum.accumulate(eq) - 1).min()
    yrs = n / 252; cagr = eq[-1] ** (1 / yrs) - 1 if yrs > 0 else float('nan')
    tp = pnl_pts[x > 0]; win = (tp > 0).mean() if traded else float('nan')
    srt = np.sort(tp)[::-1]; top = srt[:10].sum() / tp.sum() if traded and tp.sum() > 0 else float('nan')
    return dict(n=n, traded=traded, share=traded / n, win=win, sh=sh, cagr=cagr, dd=dd, top=top, mean_pts=tp.mean() if traded else float('nan'), worst=r.min(), xmed=np.median(x[x > 0]) if traded else 0, xmax=x.max())
def run(sub, kappa, exit_col, sizing, own_impact):
    sub = sub.to_pandas()
    e = kappa * sub['Delta'] - C
    if sizing == 'fixed1': x = (e > 0).astype(int).values
    else:
        xmax = np.floor(CAP / (MRATE * sub['I'].values * M)); x = np.minimum(np.floor(sub['V20'] * (np.clip(e, 0, None) / (3 * Y * sub['sig20'])) ** 2), xmax).clip(0, None).astype(int).values
    g = sub[exit_col].values
    imp = 2 * Y * sub['sig20'].values * np.sqrt(np.maximum(x, 0) / sub['V20'].values) if own_impact else 0
    pnl_pts = (g - C - imp) * (x > 0)                # per contract, points
    pnl_twd = pnl_pts * x * M; ret = pnl_twd / CAP
    return stats(sub, ret, pnl_pts, x), pnl_twd
periods = {'全 2021-26': d, '2024': d.filter(pl.col('date').str.starts_with('2024')), '2025': d.filter(pl.col('date').str.starts_with('2025')), '2026': d.filter(pl.col('date').str.starts_with('2026')), '2025-26': d.filter(pl.col('date') >= '2025-01-01')}
for exit_col, exn in [('g15', '13:45 出（預先登記）'), ('g14', '13:44 出（M3 峰值）')]:
    P(f'## {exn}\n')
    for sizing, sn in [('fixed1', '固定 1 口'), ('xstar', 'x* 部位（保證金上限）')]:
        P(f'### {sn}\n')
        P('| 期間 | κ | 交易日 (比例) | 勝率 | 每口淨均(點) | Sharpe | 年化報酬 | MaxDD | 最差日 | 前10天佔 | x* 中位/最大 | 含自身衝擊 Sharpe / 年化 |'); P('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for pn, sub in periods.items():
            for kappa in [0.3, 0.5, 0.7]:
                s, _ = run(sub, kappa, exit_col, sizing, False); s2, _ = run(sub, kappa, exit_col, sizing, True)
                if s['traded'] < 3: P(f'| {pn} | {kappa} | {s["traded"]} | - | - | - | - | - | - | - | - | - |'); continue
                P(f'| {pn} | {kappa} | {s["traded"]} ({s["share"]:.0%}) | {s["win"]:.0%} | {s["mean_pts"]:.1f} | {s["sh"]:.2f} | {s["cagr"]:.1%} | {s["dd"]:.1%} | {s["worst"]:.1%} | {s["top"]:.0%} | {s["xmed"]:.0f}/{s["xmax"]} | {s2["sh"]:.2f} / {s2["cagr"]:.1%} |')
        P('')
# equity + trade list for headline case: 13:45, x*, κ=0.5
s, pnl = run(d, 0.5, 'g15', 'xstar', True)
pd_ = d.to_pandas(); pd_['pnl_twd'] = pnl; pd_['x'] = np.minimum(np.floor(pd_['V20'] * (np.clip(0.5 * pd_['Delta'] - C, 0, None) / (3 * Y * pd_['sig20'])) ** 2), np.floor(CAP / (MRATE * pd_['I'] * M))).clip(0, None).astype(int)
tr = pd_[pd_['x'] > 0][['date', 'r', 'N', 'Delta', 'sig20', 'x', 'g15', 'pnl_twd']]
tr.to_csv(R / 'trades_k05_1345.csv', index=False)
P('## 主情境明細（13:45 出、x*、κ=0.5、含自身衝擊）\n')
P(f'交易 {len(tr)} 筆，總損益 {tr["pnl_twd"].sum()/1e4:,.0f} 萬；逐年：')
P(tr.assign(y=tr['date'].str[:4]).groupby('y').agg(n=('x', 'size'), pnl_wan=('pnl_twd', lambda v: round(v.sum() / 1e4)), x_med=('x', 'median'), win=('g15', lambda v: round((v > C).mean(), 2))).to_string())
P('\n最好 5 天：'); P(tr.nlargest(5, 'pnl_twd').round(2).to_string(index=False)); P('\n最差 5 天：'); P(tr.nsmallest(5, 'pnl_twd').round(2).to_string(index=False))
open(R / 'backtest.md', 'w').write('\n'.join(L)); print('\n'.join(L))

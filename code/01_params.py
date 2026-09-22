"""Stage 1: descriptive parameter measurement for the TX leveraged-ETF rebalance study.

No hypothesis tests. Outputs out/params.md and out/daily.parquet.
"""
import polars as pl, numpy as np, glob
from pathlib import Path
R = Path('research/tx_etf_rebalance/out'); R.mkdir(exist_ok=True)
M = 200

# ---------- futures 1s bars, day session, main contract ----------
tx = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob('data/gold/tx_mr/tx_1s/year=*/data.parquet'))])
tx = tx.filter(pl.col('session') == 'D').with_columns(t=pl.col('ts').dt.time())
mc = pl.read_parquet('data/gold/tx_mr/main_contract.parquet').select('date', 'contract', 'is_settle')
tx = tx.join(mc, left_on='session_date', right_on='date', how='inner').filter(pl.col('contract') == pl.col('contract_right'))
print('days', tx['session_date'].n_unique(), tx['session_date'].min(), tx['session_date'].max())

def px_at(df, t0, before=False):
    """last trade price at/after t0 (or last before t0 if before=True)."""
    if before:
        return df.filter(pl.col('t') < t0).group_by('session_date').agg(pl.col('last').last().alias('p'))
    return df.filter(pl.col('t') >= t0).group_by('session_date').agg(pl.col('last').first().alias('p'))

w = tx.filter((pl.col('t') >= pl.time(13, 30)) & (pl.col('t') < pl.time(13, 45)))
win = (w.group_by('session_date').agg(
        V_w=pl.col('vol').sum(), q_w=(pl.col('buy_vol') - pl.col('sell_vol')).sum(),
        spread_est=(pl.col('ask_est') - pl.col('bid_est')).mean(),
        spread_med=(pl.col('ask_est') - pl.col('bid_est')).median(), n_sec=pl.len())
    .join(px_at(tx, pl.time(13, 30)).rename({'p': 'p1330'}), on='session_date')
    .join(px_at(tx, pl.time(13, 45), before=True).rename({'p': 'p1345'}), on='session_date')
    .join(px_at(tx, pl.time(13, 25), before=True).rename({'p': 'p1325'}), on='session_date')
    .join(px_at(tx, pl.time(13, 30), before=True).rename({'p': 'p1329'}), on='session_date')
    .with_columns(dP_w=pl.col('p1345') - pl.col('p1330')).sort('session_date'))
# day session volume 08:45-13:45
day = tx.group_by('session_date').agg(V_d=pl.col('vol').sum(), p_open=pl.col('last').first(), p_close=pl.col('last').last())
win = win.join(day, on='session_date')

# ---------- ETF AUM ----------
sh = pl.read_parquet(R / 'etf_shares_issued.parquet')
px = pl.read_parquet(R / 'etf_price.parquet')
etf = sh.join(px, on=['date', 'stock_id']).with_columns(A=pl.col('NumberOfSharesIssued') * pl.col('close') / 1e8)  # 億
etf = etf.with_columns(side=pl.when(pl.col('stock_id').str.ends_with('L')).then(pl.lit('L')).otherwise(pl.lit('R')))
aum = etf.group_by('date').agg(A_L=pl.col('A').filter(pl.col('side') == 'L').sum(), A_R=pl.col('A').filter(pl.col('side') == 'R').sum(),
                              n_etf=pl.len()).with_columns(A_sum=pl.col('A_L') + pl.col('A_R')).sort('date')
aum_wide = etf.pivot(index='date', on='stock_id', values='A').sort('date')

# ---------- TAIEX daily return ----------
tw = pl.read_parquet('data/raw/finmind_daily/taiex.parquet').select(pl.col('date').cast(pl.Utf8), I=pl.col('close')).sort('date')
tw = tw.with_columns(r=pl.col('I') / pl.col('I').shift(1) - 1)

# ---------- 投信 OI ----------
oi = (pl.read_parquet('data/raw/finmind_futopt/daily/futures_inst_TX.parquet').filter(pl.col('institutional_investors') == '投信')
      .select('date', oi_L=pl.col('long_open_interest_balance_volume'), oi_S=pl.col('short_open_interest_balance_volume'),
              oi_L_amt=pl.col('long_open_interest_balance_amount'), oi_S_amt=pl.col('short_open_interest_balance_amount')))

d = (win.rename({'session_date': 'date'}).join(aum, on='date', how='left').join(tw, on='date', how='left').join(oi, on='date', how='left').sort('date')
     .with_columns(A_L_prev=pl.col('A_L').shift(1), A_R_prev=pl.col('A_R').shift(1), A_prev=pl.col('A_sum').shift(1))
     .with_columns(F=2 * pl.col('r') * pl.col('A_prev'),                                  # 億 TWD, total rebalance flow
                   fut_L=pl.col('oi_L') * pl.col('I') * M / 1e8, fut_S=pl.col('oi_S') * pl.col('I') * M / 1e8)  # 億
     .with_columns(theta_L=pl.col('fut_L') / (2 * pl.col('A_L')), theta_S=pl.col('fut_S') / (1 * pl.col('A_R')),
                   theta_all=(pl.col('fut_L') + pl.col('fut_S')) / (2 * pl.col('A_L') + pl.col('A_R'))))
d.write_parquet(R / 'daily.parquet')

# ---------- Y: impact coefficient ----------
# 15-min windows over the day session; normalise by rolling 20-day typical window vol / volume at same time-of-day
tx = tx.with_columns(mn=pl.col('ts').dt.hour().cast(pl.Int32) * 60 + pl.col('ts').dt.minute().cast(pl.Int32))
tx15 = tx.with_columns(k=(pl.col('mn') - (8 * 60 + 45)) // 15)
b = (tx15.group_by('session_date', 'k').agg(p0=pl.col('last').first(), p1=pl.col('last').last(), V=pl.col('vol').sum(),
                                            q=(pl.col('buy_vol') - pl.col('sell_vol')).sum())
     .with_columns(dP=pl.col('p1') - pl.col('p0')).sort('session_date', 'k'))
b = b.with_columns(sig=pl.col('dP').rolling_std(20).shift(1).over('k'), Vb=pl.col('V').rolling_mean(20).shift(1).over('k')).drop_nulls()
b = b.with_columns(y=pl.col('dP') / pl.col('sig'), x=pl.col('q').sign() * (pl.col('q').abs() / pl.col('Vb')).sqrt())
def ols0(df):
    df = df.filter(pl.col('x').is_finite() & pl.col('y').is_finite()); x, y = df['x'].to_numpy(), df['y'].to_numpy(); beta = (x * y).sum() / (x * x).sum()
    res = y - beta * x; se = np.sqrt((res ** 2).sum() / (len(x) - 1) / (x * x).sum()); return beta, se, len(x)
Y_all = ols0(b); Y_last = ols0(b.filter(pl.col('k') == 19)); thr = b['x'].abs().quantile(0.95); Y_big = ols0(b.filter(pl.col('x').abs() >= thr))
Y_big_last = ols0(b.filter((pl.col('k') == 19) & (pl.col('x').abs() >= b.filter(pl.col('k') == 19)['x'].abs().quantile(0.9))))
# 1-min version for robustness
tx1 = tx.with_columns(k=pl.col('mn'))
b1 = (tx1.group_by('session_date', 'k').agg(p0=pl.col('last').first(), p1=pl.col('last').last(), V=pl.col('vol').sum(), q=(pl.col('buy_vol') - pl.col('sell_vol')).sum())
      .with_columns(dP=pl.col('p1') - pl.col('p0')).sort('session_date', 'k'))
b1 = b1.with_columns(sig=pl.col('dP').rolling_std(20).shift(1).over('k'), Vb=pl.col('V').rolling_mean(20).shift(1).over('k')).drop_nulls()
b1 = b1.with_columns(y=pl.col('dP') / pl.col('sig'), x=pl.col('q').sign() * (pl.col('q').abs() / pl.col('Vb')).sqrt())
Y1_all = ols0(b1); Y1_big = ols0(b1.filter(pl.col('x').abs() >= b1['x'].abs().quantile(0.95)))


# ---------- Y from large single-lot events (avoids tick-rule tautology of net-volume version) ----------
# event = second whose largest print >= threshold; Q = that lot; dP = mid(t+h) - mid(t-1s); normalise by same-time-of-day 20d typical h-sec vol/volume
def lot_events(h, thr):
    base = tx.select('session_date', 'ts', 'mn', 'mid', 'vol')
    # typical h-second volatility & volume per 15-min slot, by day, then 20-day rolling mean shifted (no look-ahead)
    slot = tx15.with_columns(sec=pl.col('ts').dt.hour().cast(pl.Int32)*3600+pl.col('ts').dt.minute().cast(pl.Int32)*60+pl.col('ts').dt.second().cast(pl.Int32))
    slot = slot.with_columns(hb=(pl.col('sec') // h)).group_by('session_date', 'k', 'hb').agg(dp=pl.col('mid').last()-pl.col('mid').first(), v=pl.col('vol').sum())
    slot = slot.group_by('session_date', 'k').agg(sig_h=pl.col('dp').std(), v_h=pl.col('v').mean()).sort('session_date', 'k')
    slot = slot.with_columns(sig_h=pl.col('sig_h').rolling_mean(20).shift(1).over('k'), v_h=pl.col('v_h').rolling_mean(20).shift(1).over('k')).drop_nulls()
    ev = tx15.filter(pl.col('max_lot') >= thr).select('session_date', 'ts', 'k', Q=pl.col('max_lot'), sgn=pl.col('last_sgn'), m0=pl.col('mid'))
    ev = ev.with_columns(ts_h=pl.col('ts') + pl.duration(seconds=h), ts_b=pl.col('ts') - pl.duration(seconds=1))
    ev = ev.join_asof(base.select('session_date', ts_h=pl.col('ts'), mh=pl.col('mid')).sort('ts_h'), on='ts_h', by='session_date', strategy='backward')
    ev = ev.join_asof(base.select('session_date', ts_b=pl.col('ts'), mb=pl.col('mid')).sort('ts_b'), on='ts_b', by='session_date', strategy='backward')
    ev = ev.join(slot, on=['session_date', 'k']).drop_nulls().filter(pl.col('sgn') != 0)
    ev = ev.with_columns(y=(pl.col('mh') - pl.col('mb')) / pl.col('sig_h'), x=pl.col('sgn') * (pl.col('Q') / pl.col('v_h')).sqrt())
    return ev
lot_res = {}
for h in [60, 300]:
    for thr in [30, 50, 100]:
        ev = lot_events(h, thr); lot_res[(h, thr)] = ols0(ev) + (float(ev['Q'].median()), float((ev['Q']/ev['v_h']).median()))
        lot_res[(h, thr, 'w')] = ols0(ev.filter(pl.col('k') == 19)) + (float(ev.filter(pl.col('k') == 19)['Q'].median()) if (ev['k']==19).any() else np.nan, np.nan)
# ---------- theta by level regression: fut_L = theta*2*A_L + C (C = non-ETF 投信 OI) ----------
def ols(x, y):
    X = np.column_stack([np.ones(len(x)), x]); b, res, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ b
    cov = np.linalg.inv(X.T @ X) * (e @ e) / (len(x) - 2); return b, np.sqrt(np.diag(cov)), 1 - (e @ e) / ((y - y.mean()) ** 2).sum()
dl = d.drop_nulls(['fut_L', 'A_L'])
thL = ols(2 * dl['A_L'].to_numpy(), dl['fut_L'].to_numpy()); thS = ols(dl['A_R'].to_numpy(), dl['fut_S'].to_numpy())
thL26 = ols(2 * dl.filter(pl.col('date') >= '2025-01-01')['A_L'].to_numpy(), dl.filter(pl.col('date') >= '2025-01-01')['fut_L'].to_numpy())

# ---------- summary ----------
def q(s, ps=(0.1, 0.5, 0.9)): return [float(s.quantile(p)) for p in ps]
rec = d.filter(pl.col('date') >= '2026-01-01')
lines = []
P = lines.append
P('# Stage 1 參數實測\n')
P(f'樣本：{d["date"].min()} ~ {d["date"].max()}，{len(d)} 交易日（含結算日/轉倉日，未排除）；主力合約；秒 bar 來源 gold/tx_mr/tx_1s\n')
P('## 窗口 13:30–13:45 (主力合約)\n')
P('| 參數 | 全樣本 mean | median | p10 | p90 | 2026 至今 mean |')
P('|---|---|---|---|---|---|')
for c, name in [('V_w', 'V_w 成交口數'), ('dP_w', 'ΔP 13:30→13:45 (點)'), ('spread_est', 'spread 估計 (點, mean)'), ('spread_med', 'spread 估計 (點, median)'), ('V_d', '日盤總量')]:
    P(f'| {name} | {d[c].mean():.2f} | {d[c].median():.2f} | {q(d[c])[0]:.2f} | {q(d[c])[2]:.2f} | {rec[c].mean():.2f} |')
P(f'\nσ_w = std(ΔP 13:30→13:45) 全樣本 **{d["dP_w"].std():.1f} 點**（穩健 1.4826·MAD = {1.4826*float((d["dP_w"]-d["dP_w"].median()).abs().median()):.1f}）；逐年（sig=std, sig_r=穩健）：')
yr = d.with_columns(y=pl.col('date').str.slice(0, 4)).group_by('y').agg(sig=pl.col('dP_w').std(), sig_r=1.4826*(pl.col('dP_w')-pl.col('dP_w').median()).abs().median(), Vw=pl.col('V_w').mean(), sp=pl.col('spread_med').median(), I=pl.col('I').mean(), n=pl.len()).sort('y')
P(yr.to_pandas().to_string(index=False)); P('')
P(f'σ_w / I 全樣本 = {d["dP_w"].std() / d["I"].mean() * 1e4:.1f} bp；2026 = {rec["dP_w"].std():.1f} 點 = {rec["dP_w"].std() / rec["I"].mean() * 1e4:.1f} bp\n')
P('## ΣA 八檔槓反 ETF 淨資產（億 TWD，單位數 × 收盤價，收盤價當淨值代理）\n')
yy = aum.with_columns(y=pl.col('date').str.slice(0, 4)).group_by('y').agg(A_L=pl.col('A_L').mean(), A_R=pl.col('A_R').mean(), A_sum=pl.col('A_sum').mean(), A_min=pl.col('A_sum').min(), A_max=pl.col('A_sum').max()).sort('y')
P(yy.to_pandas().round(0).to_string(index=False)); P('')
P('最新一日各檔：'); P(aum_wide.tail(1).to_pandas().round(0).to_string(index=False)); P('')
P('## θ 行為版 = 投信 TX 未平倉名目 / 目標曝險（日序列）\n')
th = d.with_columns(y=pl.col('date').str.slice(0, 4)).group_by('y').agg(theta_L=pl.col('theta_L').median(), theta_S=pl.col('theta_S').median(), theta_all=pl.col('theta_all').median(),
                                                                      oi_L=pl.col('oi_L').median(), oi_S=pl.col('oi_S').median()).sort('y')
P(th.to_pandas().round(3).to_string(index=False)); P('')
P(f'全樣本 θ_L median {d["theta_L"].median():.3f} (p10 {q(d["theta_L"])[0]:.3f}, p90 {q(d["theta_L"])[2]:.3f})；θ_S median {d["theta_S"].median():.3f}；合併 θ median {d["theta_all"].median():.3f}')
P('（限制：投信 OI 含非 ETF 投信、只含大台不含小台；反1 ETF 也可能用小台或現股放空/借券）\n')
P('## Y 平方根衝擊係數（ΔP/σ = Y·sign(q)·√(|q|/V)，q=窗口買賣淨量，σ、V 用同時段前 20 日典型值）\n')
P('| 版本 | Y | SE | n |'); P('|---|---|---|---|')
for name, r_ in [('15 分鐘窗口 全部', Y_all), ('15 分鐘窗口 僅 13:30–13:45', Y_last), ('15 分鐘 大單事件 |x| top5%', Y_big), ('13:30–13:45 大單 top10%', Y_big_last), ('1 分鐘窗口 全部', Y1_all), ('1 分鐘 大單 top5%', Y1_big)]:
    P(f'| {name} | {r_[0]:.3f} | {r_[1]:.3f} | {r_[2]} |')
P('')
P('### Y 大單事件版（單筆最大成交 ≥ 門檻口數的秒，ΔP = mid(t+h) − mid(t−1s)，用同時段 20 日典型 h 秒波動/量正規化）\n')
P('| h 秒 | 門檻口 | Y 全日 | SE | n | Q 中位 | Q/V_h 中位 | Y 僅 13:30–13:45 | n |'); P('|---|---|---|---|---|---|---|---|---|')
for h in [60, 300]:
    for thr in [30, 50, 100]:
        a = lot_res[(h, thr)]; w_ = lot_res[(h, thr, 'w')]
        P(f'| {h} | {thr} | {a[0]:.3f} | {a[1]:.3f} | {a[2]} | {a[3]:.0f} | {a[4]:.3f} | {w_[0]:.3f} | {w_[2]} |')
P('')
P('### θ 水準迴歸 fut_L = θ·2·A_L + C（C = 非 ETF 投信多單名目）\n')
P(f'全樣本：θ_L = {thL[0][1]:.3f} (SE {thL[1][1]:.3f}), C = {thL[0][0]:.0f} 億, R² = {thL[2]:.3f}')
P(f'2025 起：θ_L = {thL26[0][1]:.3f} (SE {thL26[1][1]:.3f}), C = {thL26[0][0]:.0f} 億, R² = {thL26[2]:.3f}')
P(f'反1：fut_S = θ_S·A_R + C：θ_S = {thS[0][1]:.3f} (SE {thS[1][1]:.3f}), C = {thS[0][0]:.0f} 億, R² = {thS[2]:.3f}')
P('')
# ---------- recompute Δ and r* ----------
Y_tick = Y_last[0]; Y_lot = lot_res[(300, 50)][0]; Y = Y_lot; sig_w = rec['dP_w'].std(); Vw = rec['V_w'].mean(); I = rec['I'].mean()
A_now = float(aum['A_sum'][-1]); A_L_now = float(aum['A_L'][-1]); A_R_now = float(aum['A_R'][-1])
theta = float(thL26[0][1])
c = 3.35
P('## 重算 Δ 與 |r*|（用 2026 至今參數）\n')
P(f'Y_lot={Y_lot:.3f} (大單事件 h=300 ≥50口), Y_tick={Y_tick:.3f} (淨量 15 分鐘窗口, 含 tick-rule 套套邏輯偏高), σ_w={sig_w:.1f} 點, V_w={Vw:.0f} 口, I={I:.0f}, ΣA={A_now:.0f} 億 (正2 {A_L_now:.0f} + 反1 {A_R_now:.0f}), θ={theta:.3f}, c={c} 點\n')
P('| |r| | F=2·r·ΣA (億) | N=θF/(I·m) 口 | N/V_w | Δ (Y=0.5 假設) | Δ (Y_lot) | Δ (Y_tick) |'); P('|---|---|---|---|---|---|---|')
for rr in [0.005, 0.01, float(d['r'].abs().median()), float(d['r'].abs().mean()), 0.02, 0.03]:
    F = 2 * rr * A_now; N = theta * F * 1e8 / (I * M); s_ = sig_w * np.sqrt(N / Vw)
    P(f'| {rr*100:.2f}% | {F:.0f} | {N:.0f} | {N/Vw:.2f} | {0.5*s_:.1f} | {Y_lot*s_:.1f} | {Y_tick*s_:.1f} |')
P('')
for kappa in [1.0, 0.5, 0.3]:
  for Y in [0.5, Y_lot, Y_tick]:
    rstar = (I * M * Vw) / (2 * theta * A_now * 1e8) * (c / (kappa * Y * sig_w)) ** 2
    P(f'κ={kappa}（假設值）, Y={Y:.2f}：|r*| = {rstar*100:.2f}%')
P('\nΔ 日序列（用每日實際 r、A_{t-1}、θ 當年中位、當年 σ_w/V_w）：')
dd = d.with_columns(y=pl.col('date').str.slice(0, 4)).join(yr.select('y', 'sig', 'Vw'), on='y').join(th.select('y', 'theta_all'), on='y')
dd = dd.with_columns(N=theta * pl.col('F').abs() * 1e8 / (pl.col('I') * M)).with_columns(Delta_lot=Y_lot * pl.col('sig') * (pl.col('N') / pl.col('Vw')).sqrt(), Delta_05=0.5 * pl.col('sig') * (pl.col('N') / pl.col('Vw')).sqrt())
P(dd.group_by('y').agg(N_med=pl.col('N').median(), N_p90=pl.col('N').quantile(0.9), Dlot_med=pl.col('Delta_lot').median(), Dlot_p90=pl.col('Delta_lot').quantile(0.9), D05_med=pl.col('Delta_05').median(), D05_p90=pl.col('Delta_05').quantile(0.9), share_D05_ge5=(pl.col('Delta_05') >= 5).mean()).sort('y').to_pandas().round(2).to_string(index=False))
dd.write_parquet(R / 'daily.parquet')
open(R / 'params.md', 'w').write('\n'.join(lines)); print('\n'.join(lines))

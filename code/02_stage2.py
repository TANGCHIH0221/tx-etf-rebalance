"""Stage 2: P1 rebalance instrument, P2 flow-to-volume, P3 pre-pricing. Pre-registered windows: 13:30->13:35 and 13:30->13:45."""
import polars as pl, numpy as np, glob, os, datetime as dt
import statsmodels.api as sm
from pathlib import Path
R = Path('research/tx_etf_rebalance/out'); M = 200
d = pl.read_parquet(R / 'daily.parquet').sort('date')

# ---------- futures minute path after 13:30 + 13:25-13:30 volumes + pre-window realised vol ----------
tx = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob('data/gold/tx_mr/tx_1s/year=*/data.parquet'))]).filter(pl.col('session') == 'D')
mc = pl.read_parquet('data/gold/tx_mr/main_contract.parquet').select('date', 'contract')
tx = tx.join(mc, left_on='session_date', right_on='date').filter(pl.col('contract') == pl.col('contract_right')).with_columns(t=pl.col('ts').dt.time())
paths = []
for k in range(1, 16):
    paths.append(tx.filter(pl.col('t') < pl.time(13, 30 + k)).group_by('session_date').agg(pl.col('last').last().alias(f'p{k}')))
pm = paths[0]
for p in paths[1:]: pm = pm.join(p, on='session_date')
v5 = tx.filter((pl.col('t') >= pl.time(13, 30)) & (pl.col('t') < pl.time(13, 35))).group_by('session_date').agg(V_5=pl.col('vol').sum())
vpre = tx.filter((pl.col('t') >= pl.time(13, 25)) & (pl.col('t') < pl.time(13, 30))).group_by('session_date').agg(V_pre=pl.col('vol').sum())
# realised vol 08:45-13:25 from 1-min last prices, scaled to 15-min horizon (points)
m1 = tx.filter(pl.col('t') < pl.time(13, 25)).with_columns(mn=pl.col('ts').dt.hour().cast(pl.Int32) * 60 + pl.col('ts').dt.minute().cast(pl.Int32)).group_by('session_date', 'mn').agg(pl.col('last').last()).sort('session_date', 'mn')
rv = m1.with_columns(dp=pl.col('last').diff().over('session_date')).group_by('session_date').agg(sig_t=pl.col('dp').std() * np.sqrt(15))
d = d.join(pm, left_on='date', right_on='session_date', how='left').join(v5, left_on='date', right_on='session_date', how='left').join(vpre, left_on='date', right_on='session_date', how='left').join(rv, left_on='date', right_on='session_date', how='left')

# ---------- TAIEX at 13:25 (last continuous value) and close from 5s files ----------
rows = []
for f in glob.glob('data/raw/finmind_intraday/taiex_5s/*'):
    df = pl.read_parquet(f); rows.append(df.rename({'date': 'ts'}))
for f in glob.glob('data/raw/finmind_intraday/index_5s/*.parquet'):
    df = pl.read_parquet(f).filter(pl.col('stock_id') == 'TAIEX').select(ts=pl.col('date') + ' ' + pl.col('time'), TAIEX=pl.col('price')); rows.append(df)
for f in glob.glob('data/raw/finmind_intraday/taiex_5s_tail/*.parquet'):
    df = pl.read_parquet(f)
    if len(df): rows.append(df.select(ts=pl.col('date'), TAIEX=pl.col('TAIEX')))
i5 = pl.concat(rows).with_columns(date=pl.col('ts').str.slice(0, 10), tm=pl.col('ts').str.slice(11, 8)).unique(['date', 'tm']).sort('date', 'tm')
idx = i5.group_by('date').agg(I_1325=pl.col('TAIEX').filter(pl.col('tm') <= '13:25:00').last(), I_1330=pl.col('TAIEX').filter(pl.col('tm') >= '13:30:00').first())
d = d.join(idx, on='date', how='left')
print('days with I_1325:', d['I_1325'].is_not_null().sum())

# ---------- ETF units (creation flow control) ----------
sh = pl.read_parquet(R / 'etf_shares_issued.parquet').join(pl.read_parquet(R / 'etf_price.parquet'), on=['date', 'stock_id'])
sh = sh.sort('stock_id', 'date').with_columns(dU=pl.col('NumberOfSharesIssued').diff().over('stock_id')).with_columns(inflow=pl.col('dU') * pl.col('close') / 1e8, side=pl.col('stock_id').str.slice(-1))
infl = sh.group_by('date').agg(inflow_L=pl.col('inflow').filter(pl.col('side') == 'L').sum(), inflow_R=pl.col('inflow').filter(pl.col('side') == 'R').sum())
d = d.join(infl, on='date', how='left')

# ---------- OI changes valued at today's index ----------
d = d.with_columns(dN_L=pl.col('oi_L').diff(), dN_S=pl.col('oi_S').diff()).with_columns(
    y_L=pl.col('dN_L') * pl.col('I') * M / 1e8, y_S=-pl.col('dN_S') * pl.col('I') * M / 1e8,
    x_L=2 * pl.col('A_L_prev') * pl.col('r'), x_R=2 * pl.col('A_R_prev') * pl.col('r'), x_all=2 * pl.col('A_prev') * pl.col('r'))

# ---------- exclusion calendar ----------
dates = d['date'].to_list(); D = [dt.date.fromisoformat(x) for x in dates]; pos = {x: i for i, x in enumerate(dates)}
st = pl.read_parquet('data/raw/finmind_futopt/daily/settlement_TX.parquet').filter(~pl.col('contract_month').str.contains('W'))['date'].cast(pl.Utf8).to_list()
ex = {}
def mark(day, tag):
    if day in pos: ex.setdefault(day, set()).add(tag)
for s in st:
    if s in pos:
        for k in range(0, 4): mark(dates[pos[s] - k], 'settle_roll')
    else:  # settlement day itself absent (contract expired); mark 3 prior trading days
        prior = [x for x in dates if x < s][-3:]
        for x in prior: mark(x, 'settle_roll')
for i, day in enumerate(D):
    nxt = D[i + 1] if i + 1 < len(D) else None
    if nxt is None: continue
    if nxt.month != day.month: mark(dates[i], 'month_end'); 
    if nxt.month != day.month and day.month in (3, 6, 9, 12): mark(dates[i], 'quarter_end')
    if nxt.month != day.month and day.month in (2, 5, 8, 11): mark(dates[i], 'msci')
    if i + 2 < len(D) and D[i + 2].month != day.month: mark(dates[i], 'sgx_msci_settle')  # 摩台結算 = 倒數第二營業日
    if (nxt - day).days >= 4: mark(dates[i], 'holiday_eve')
# TSMC revenue: 10th or last trading day before, ±1 trading day; Hon Hai: 5th or last trading day before
for y in range(2021, 2027):
    for mth in range(1, 13):
        for dayn, tag, pm_ in [(10, 'tsmc_rev', 1), (5, 'honhai_rev', 0)]:
            tgt = dt.date(y, mth, dayn); cand = [x for x in D if x <= tgt]
            if not cand: continue
            c = cand[-1]; j = pos[c.isoformat()]
            for k in range(-pm_, pm_ + 1):
                if 0 <= j + k < len(dates): mark(dates[j + k], tag)
for s in ['2021-01-14', '2021-04-15', '2021-07-15', '2021-10-14', '2022-01-13', '2022-04-14', '2022-07-14', '2022-10-13', '2023-01-12', '2023-04-20', '2023-07-20', '2023-10-19',
          '2024-01-18', '2024-04-18', '2024-07-18', '2024-10-17', '2025-01-16', '2025-04-17', '2025-07-17', '2025-10-16', '2026-01-15', '2026-04-16', '2026-07-16']:
    mark(s, 'tsmc_call')
d = d.with_columns(excl=pl.col('date').is_in(list(ex.keys())), excl_tags=pl.col('date').map_elements(lambda x: ','.join(sorted(ex.get(x, []))), return_dtype=pl.Utf8))
d.write_parquet(R / 'daily2.parquet')
from collections import Counter
tagc = Counter(t for v in ex.values() for t in v)

# ---------- regression helper ----------
def hac(df, ycol, xcols, lag=5):
    df = df.select([ycol] + xcols).drop_nulls().filter(pl.all_horizontal(pl.all().is_finite()))
    X = sm.add_constant(df.select(xcols).to_numpy()); y = df[ycol].to_numpy()
    m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    return m, len(df)
def row(name, m, n, xi=1):
    return f'| {name} | {m.params[xi]:.3f} | {m.bse[xi]:.3f} | {m.params[xi]/m.bse[xi]:.2f} | {n} | {m.rsquared:.3f} |'
L = []; P = L.append
P('# Stage 2 前提檢定\n'); P(f'樣本 {d["date"].min()} ~ {d["date"].max()}，{len(d)} 日；排除版剔除 {d["excl"].sum()} 日。排除標籤計數：{dict(tagc)}\n')
P('HAC (Newey-West, lag 5) 標準誤。「全」=全樣本，「排」=排除版。\n')

# ---------- P1 ----------
P('## P1 再平衡工具判定：Δ(投信期貨名目)_t 對 2·A_{t−1}·r_t（億 TWD）\n')
P('| 設定 | β | HAC SE | t | n | R² |'); P('|---|---|---|---|---|---|')
for lab, sub in [('全', d), ('排', d.filter(~pl.col('excl')))]:
    m, n = hac(sub, 'y_L', ['x_L']); P(row(f'正2 期貨多單 {lab}', m, n))
    m, n = hac(sub, 'y_L', ['x_L', 'inflow_L']); P(row(f'正2 + 申贖控制 {lab}', m, n) + f' 申贖係數 {m.params[2]:.2f} (t {m.params[2]/m.bse[2]:.1f})')
    m, n = hac(sub, 'y_S', ['x_R']); P(row(f'反1 期貨空單 {lab}', m, n))
    m, n = hac(sub, 'y_S', ['x_R', 'inflow_R']); P(row(f'反1 + 申贖控制 {lab}', m, n) + f' 申贖係數 {m.params[2]:.2f} (t {m.params[2]/m.bse[2]:.1f})')
    sub2 = sub.with_columns(y_all=pl.col('y_L') + pl.col('y_S'))
    m, n = hac(sub2, 'y_all', ['x_all']); P(row(f'合併 {lab}', m, n))
P('')
P('### 時間對齊：y_{t+k} 對 x_t（k=−1 為 placebo，k=+1 檢查是否延到隔天）\n')
P('| k | β 正2 | t | β 反1 | t | n |'); P('|---|---|---|---|---|---|')
for k in [-1, 0, 1]:
    s = d.with_columns(yL=pl.col('y_L').shift(-k), yS=pl.col('y_S').shift(-k))
    mL, n = hac(s, 'yL', ['x_L']); mS, _ = hac(s, 'yS', ['x_R'])
    P(f'| {k:+d} | {mL.params[1]:.3f} | {mL.params[1]/mL.bse[1]:.2f} | {mS.params[1]:.3f} | {mS.params[1]/mS.bse[1]:.2f} | {n} |')
P('')
P('### 逐年 正2 β\n'); P('| 年 | β | t | n | R² |'); P('|---|---|---|---|---|')
for y in range(2021, 2027):
    m, n = hac(d.filter(pl.col('date').str.starts_with(str(y))), 'y_L', ['x_L']); P(f'| {y} | {m.params[1]:.3f} | {m.params[1]/m.bse[1]:.2f} | {n} | {m.rsquared:.3f} |')
theta_beh = hac(d.filter(pl.col('date') >= '2025-01-01'), 'y_L', ['x_L', 'inflow_L'])[0].params[1]
theta_S_beh = hac(d.filter(pl.col('date') >= '2025-01-01'), 'y_S', ['x_R', 'inflow_R'])[0].params[1]
P(f'\n行為版 θ（2025 起、含申贖控制）：正2 **{theta_beh:.3f}**，反1 **{theta_S_beh:.3f}**。現股端無 PCF 歷史，現股份額 = 1 − θ 推得。\n')

# ---------- P2 ----------
d = d.with_columns(N=(theta_beh * 2 * pl.col('A_L_prev') + theta_S_beh * 2 * pl.col('A_R_prev')) * pl.col('r').abs() * 1e8 / (pl.col('I') * M)).with_columns(N2=pl.col('N') ** 2 / 1000)
d = d.with_columns(V_w20=pl.col('V_w').rolling_mean(20).shift(1), V_520=pl.col('V_5').rolling_mean(20).shift(1))
P('## P2 流量是否落在窗口：V = α + β₁N + β₂N²/1000 + γσ_t (+ 20 日均量控制)\n')
P('N = 行為版 θ × 2|r|A_{t−1}/(I·m) 口；σ_t = 08:45–13:25 一分鐘變動 std × √15（點）。理論 β₁ ≈ 1。\n')
P('| 窗口 | 版本 | β₁ | HAC SE | t | β₂ | t | γ | t | n | R² |'); P('|---|---|---|---|---|---|---|---|---|---|---|')
for win, vc, ctl in [('13:30–13:45', 'V_w', 'V_w20'), ('13:30–13:35', 'V_5', 'V_520')]:
    for lab, sub in [('全', d), ('排', d.filter(~pl.col('excl')))]:
        for cv, xs in [('spec', ['N', 'N2', 'sig_t']), ('+均量', ['N', 'N2', 'sig_t', ctl])]:
            m, n = hac(sub, vc, xs)
            P(f'| {win} | {lab} {cv} | {m.params[1]:.3f} | {m.bse[1]:.3f} | {m.params[1]/m.bse[1]:.2f} | {m.params[2]:.3f} | {m.params[2]/m.bse[2]:.2f} | {m.params[3]:.2f} | {m.params[3]/m.bse[3]:.2f} | {n} | {m.rsquared:.3f} |')
P('')
P('### 對照：同樣模型解釋 13:25–13:30 量與 13:00–13:25 之外的窗口（流量若提前，會出現在這裡）\n')
P('| 窗口 | β₁ | t | n | R² |'); P('|---|---|---|---|---|')
for vc in ['V_pre']:
    m, n = hac(d, vc, ['N', 'N2', 'sig_t']); P(f'| 13:25–13:30 | {m.params[1]:.3f} | {m.params[1]/m.bse[1]:.2f} | {n} | {m.rsquared:.3f} |')
P('')

# ---------- P3 ----------
d = d.with_columns(rhat=pl.col('I_1330') - pl.col('I_1325'), db_pre=(pl.col('p1329') - pl.col('I_1330')) - (pl.col('p1325') - pl.col('I_1325')),
                   jump_1330=pl.col('p1330') - pl.col('p1329'), db_5=pl.col('p5') - pl.col('p1330'), db_15=pl.col('p1345') - pl.col('p1330'),
                   F_sgn=pl.col('r').sign() * pl.col('N').sqrt(), r_pts=pl.col('r') * pl.col('I'))
P('## P3 擁擠度：Δbasis 對 r̂（13:25→13:30 收盤 surprise，指數點）與 sign(F)√N\n')
P(f'有 13:25 指數的日子：{d["I_1325"].is_not_null().sum()}（全部補齊）。Δbasis_pre = (F_13:29:59 − I_close) − (F_13:25 − I_13:25)。13:30 後指數凍結，Δbasis = ΔF。\n')
P('| 被解釋 | 版本 | β(r̂) | t | β(sign F·√N) | t | n | R² |'); P('|---|---|---|---|---|---|---|---|')
for yc, name in [('db_pre', '13:25→13:30 (進場前)'), ('jump_1330', '13:29:59→13:30:00'), ('db_5', '13:30→13:35'), ('db_15', '13:30→13:45')]:
    for lab, sub in [('全', d), ('排', d.filter(~pl.col('excl')))]:
        m1_, n1 = hac(sub, yc, ['rhat']); m2_, n2 = hac(sub, yc, ['rhat', 'F_sgn'])
        P(f'| {name} | {lab} | {m1_.params[1]:.3f} | {m1_.params[1]/m1_.bse[1]:.2f} | {m2_.params[2]:.3f} | {m2_.params[2]/m2_.bse[2]:.2f} | {n2} | {m2_.rsquared:.3f} |')
P('')
P('### 全樣本版（不需 13:25 指數）：ΔF 對 sign(F)√N，含 r_t（點）控制\n')
P('| 被解釋 | 版本 | β(sign F·√N) | HAC SE | t | β(r_pts) | t | n | R² |'); P('|---|---|---|---|---|---|---|---|---|')
for yc, name in [('jump_1330', '13:29:59→13:30:00'), ('db_5', '13:30→13:35'), ('db_15', '13:30→13:45')]:
    for lab, sub in [('全', d), ('排', d.filter(~pl.col('excl')))]:
        m, n = hac(sub, yc, ['F_sgn', 'r_pts']); P(f'| {name} | {lab} | {m.params[1]:.3f} | {m.bse[1]:.3f} | {m.params[1]/m.bse[1]:.2f} | {m.params[2]:.4f} | {m.params[2]/m.bse[2]:.2f} | {n} | {m.rsquared:.3f} |')
d.write_parquet(R / 'daily2.parquet')
open(R / 'stage2.md', 'w').write('\n'.join(L)); print('\n'.join(L))

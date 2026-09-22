"""Stage 3: M1 impact, M2 reversal, M3 decay shape, M4 capacity. Pre-registered exits 13:35 and 13:45."""
import polars as pl, numpy as np, glob
import statsmodels.api as sm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
R = Path('research/tx_etf_rebalance/out'); M = 200; Y = 0.5; C = 3.35
d = pl.read_parquet(R / 'daily3.parquet').sort('date')
tx = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob('data/gold/tx_mr/tx_1s/year=*/data.parquet'))])
mc = pl.read_parquet('data/gold/tx_mr/main_contract.parquet').select('date', 'contract')
tx = tx.join(mc, left_on='session_date', right_on='date').filter(pl.col('contract') == pl.col('contract_right')).with_columns(t=pl.col('ts').dt.time())
# night session prices (session_date = same day, 15:00 start) and next-day open
N_ = tx.filter(pl.col('session') == 'N')
night = N_.group_by('session_date').agg(p_n1500=pl.col('last').first())
for mm in [5, 10, 15, 20, 25, 30]:
    night = night.join(N_.filter(pl.col('t') < pl.time(15, mm)).group_by('session_date').agg(pl.col('last').last().alias(f'p_n{mm}')), on='session_date', how='left')
nxt = tx.filter(pl.col('session') == 'D').group_by('session_date').agg(p_open=pl.col('last').first()).sort('session_date').with_columns(p_next_open=pl.col('p_open').shift(-1), next_date=pl.col('session_date').shift(-1))
d = d.join(night, left_on='date', right_on='session_date', how='left').join(nxt.select('session_date', 'p_next_open'), left_on='date', right_on='session_date', how='left')
d = d.with_columns(s_abs=pl.col('r').sign() * (pl.col('r').abs() * pl.col('I')).sqrt())  # sign(r)√|r_pts|
d = d.with_columns(m2_night=pl.col('p_n30') - pl.col('p1345'), m2_gap=pl.col('p_n1500') - pl.col('p1345'), m2_open=pl.col('p_next_open') - pl.col('p1345'))

def hac(df, ycol, xcols, lag=5):
    df = df.select([ycol] + xcols).drop_nulls().filter(pl.all_horizontal(pl.all().is_finite()))
    X = sm.add_constant(df.select(xcols).to_numpy()); y = df[ycol].to_numpy()
    return sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lag}), len(df)
samples = {'全': d, '排': d.filter(~pl.col('excl')), '2025+': d.filter(pl.col('date') >= '2025-01-01'), '2026-05+': d.filter(pl.col('date') >= '2026-05-01'), '2026-05+ 排': d.filter((pl.col('date') >= '2026-05-01') & ~pl.col('excl'))}
N26 = d.filter(pl.col('date') >= '2026-05-01')['N']; Nmed, Np90 = float(N26.median()), float(N26.quantile(0.9))
L = []; P = L.append
P('# Stage 3 主檢定與容量\n'); P(f'N（行為版 θ 換算口數）2026-05 後中位 {Nmed:.0f}、p90 {Np90:.0f}；Δ 換算 = β·√N。\n')
# ---------- M1 ----------
P('## M1 衝擊估計：ΔF(13:30→出場) = α + β·sign(F)√N + controls\n')
P('controls：(a) 無；(b) r̂ 收盤 surprise（需 13:25 指數子樣本）；(c) sign(r)√|r_pts|（嚴格版，β 只靠 A_{t−1} 變異識別）；(d) b+c+σ_t。\n')
P('| 出場 | 樣本 | controls | β | HAC SE | t | n | R² | Δ@N中位 | Δ@N p90 |'); P('|---|---|---|---|---|---|---|---|---|---|')
M1 = {}
for yc, nm in [('db_5', '13:35'), ('db_15', '13:45')]:
    for sl, sub in samples.items():
        for cl, xs in [('a', ['F_sgn']), ('b', ['F_sgn', 'rhat']), ('c', ['F_sgn', 's_abs']), ('d', ['F_sgn', 'rhat', 's_abs', 'sig_t'])]:
            m, n = hac(sub, yc, xs); b = m.params[1]; M1[(yc, sl, cl)] = (b, m.bse[1], n)
            P(f'| {nm} | {sl} | {cl} | {b:.3f} | {m.bse[1]:.3f} | {b/m.bse[1]:.2f} | {n} | {m.rsquared:.3f} | {b*np.sqrt(Nmed):.1f} | {b*np.sqrt(Np90):.1f} |')
P('')
# ---------- M2 ----------
P('## M2 反轉檢定：ret(13:45→夜盤 15:30)、ret(13:45→次日 08:45) 對 sign(F)√N；以及對窗口內實際漂移 ΔF(13:30→13:45)\n')
P('| 被解釋 | 樣本 | β(sign F√N) | t | β(ΔF 13:30→13:45) | t | n |'); P('|---|---|---|---|---|---|---|')
for yc, nm in [('m2_gap', '13:45→夜盤 15:00 開'), ('m2_night', '13:45→夜盤 15:30'), ('m2_open', '13:45→次日 08:45')]:
    for sl in ['全', '排', '2026-05+']:
        sub = samples[sl]; m1_, n = hac(sub, yc, ['F_sgn']); m2_, _ = hac(sub, yc, ['F_sgn', 'db_15'])
        P(f'| {nm} | {sl} | {m1_.params[1]:.3f} | {m1_.params[1]/m1_.bse[1]:.2f} | {m2_.params[2]:.3f} | {m2_.params[2]/m2_.bse[2]:.2f} | {n} |')
b15 = M1[('db_15', '全', 'a')][0]
P(f'\n回吐比例（全樣本，β_M2 / β_M1(13:45, a)={b15:.3f}）：')
for yc, nm in [('m2_night', '夜盤 15:30'), ('m2_open', '次日開盤')]:
    m, n = hac(d, yc, ['F_sgn']); P(f'- {nm}: {m.params[1]/b15:+.2f}（負 = 回吐）')
P('')
# ---------- M3 ----------
P('## M3 衰減形狀：β_k = ΔF(13:30→13:30+k) 對 sign(F)√N，k=1..15，再接夜盤\n')
ks = list(range(1, 16)); rows = []
for sl in ['全', '排', '2026-05+']:
    sub = samples[sl]; bs = []; ts = []
    for k in ks:
        m, n = hac(sub.with_columns(yk=pl.col(f'p{k}') - pl.col('p1330')), 'yk', ['F_sgn']); bs.append(m.params[1]); ts.append(m.params[1] / m.bse[1])
    for lab, col in [('15:00', 'p_n1500'), ('15:15', 'p_n15'), ('15:30', 'p_n30'), ('次開', 'p_next_open')]:
        m, n = hac(sub.with_columns(yk=pl.col(col) - pl.col('p1330')), 'yk', ['F_sgn']); bs.append(m.params[1]); ts.append(m.params[1] / m.bse[1])
    rows.append((sl, bs, ts))
labels = [f'{k}m' for k in ks] + ['15:00', '15:15', '15:30', '次開']
P('| 樣本 | ' + ' | '.join(labels) + ' |'); P('|---|' + '---|' * len(labels))
for sl, bs, ts in rows:
    P(f'| {sl} β | ' + ' | '.join(f'{b:.2f}' for b in bs) + ' |'); P(f'| {sl} t | ' + ' | '.join(f'{t:.1f}' for t in ts) + ' |')
pk = int(np.argmax(rows[0][1][:15])) + 1; pk26 = int(np.argmax(rows[2][1][:15])) + 1
P(f'\n峰值：全樣本 13:30+{pk} 分（β {max(rows[0][1][:15]):.2f}）；2026-05+ 13:30+{pk26} 分（β {max(rows[2][1][:15]):.2f}）。\n')
# conditional mean path (points) for 2026-05+ split by |N| tercile, sign-adjusted
sub = samples['2026-05+'].with_columns(sg=pl.col('r').sign())
P('2026-05+ 依 N 三分位的符號調整平均路徑（點，sign(F)·ΔF）：')
q1, q2 = float(sub['N'].quantile(1/3)), float(sub['N'].quantile(2/3))
P('| N 分位 | n | ' + ' | '.join(f'{k}m' for k in [1, 3, 5, 10, 15]) + ' | 15:30 | 次開 |'); P('|---|---|' + '---|' * 7)
for nm, s_ in [('低', sub.filter(pl.col('N') < q1)), ('中', sub.filter((pl.col('N') >= q1) & (pl.col('N') < q2))), ('高', sub.filter(pl.col('N') >= q2))]:
    vals = [float((s_['sg'] * (s_[f'p{k}'] - s_['p1330'])).mean()) for k in [1, 3, 5, 10, 15]] + [float((s_['sg'] * (s_['p_n30'] - s_['p1330'])).mean()), float((s_['sg'] * (s_['p_next_open'] - s_['p1330'])).mean())]
    P(f'| {nm} (N 中位 {float(s_["N"].median()):.0f}) | {len(s_)} | ' + ' | '.join(f'{v:.1f}' for v in vals) + ' |')
P('')
# chart
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
xs = list(range(1, 16)) + [18, 20, 22, 25]; cols = {'全': '#2563EB', '排': '#D97706', '2026-05+': '#059669'}; en = {'全': 'all days', '排': 'event-excluded', '2026-05+': '2026-05 onward'}
for sl, bs, ts in rows:
    ax.plot(xs, bs, lw=2, color=cols[sl], label=en[sl], marker='o', ms=3)
ax.axvline(15.5, color='#9CA3AF', lw=1, ls='--'); ax.axhline(0, color='#9CA3AF', lw=1)
ax.set_xticks(xs); ax.set_xticklabels([str(k) for k in range(1, 16)] + ['15:00', '15:15', '15:30', 'next open'], fontsize=7)
ax.set_xlabel('minutes after 13:30  |  night / next open'); ax.set_ylabel('β  (points per √contract)')
ax.set_title('M3: impact coefficient by horizon,  ΔF(13:30→t) on sign(F)√N'); ax.legend(frameon=False); ax.grid(alpha=0.2)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(R / 'm3_decay.png'); plt.close()
# ---------- M4 ----------
P('## M4 容量與最適部位（x* = V_w·[(κΔ−c)/(3Yσ_w)]², P = x*(κΔ−c)/3；Y=0.5、c=3.35 點）\n')
reg = samples['2026-05+']; Vw, sw, swr = float(reg['V_w'].mean()), float(reg['dP_w'].std()), 1.4826 * float((reg['dP_w'] - reg['dP_w'].median()).abs().median())
V5, s5 = float(reg['V_5'].mean()), float(reg['db_5'].std())
P(f'制度參數 2026-05+（n={len(reg)}）：V_w(15m)={Vw:.0f} 口、σ_w(15m) std {sw:.1f} / 穩健 {swr:.1f} 點；V_5={V5:.0f}、σ_5={s5:.1f} 點。\n')
def capacity(beta, Vw_, sig, kappa, Nser, dscale=1.0, vscale=1.0):
    tot = 0.0; days = 0; xs_ = []
    for n_ in Nser:
        delta = beta * np.sqrt(n_) * dscale; e = kappa * delta - C
        if e <= 0: continue
        x = Vw_ * vscale * (e / (3 * Y * sig)) ** 2; tot += x * e / 3; days += 1; xs_.append(x)
    return tot, days, (np.median(xs_) if xs_ else 0), (np.max(xs_) if xs_ else 0)
Nser = reg['N'].drop_nulls().to_list(); nd = len(Nser)
for yc, nm, Vx, sx in [('db_15', '13:45 出場', Vw, sw), ('db_5', '13:35 出場', V5, s5)]:
    for cl in ['a', 'c']:
        beta = M1[(yc, '2026-05+', cl)][0]
        P(f'### {nm}，β 取 2026-05+ controls {cl} = {beta:.3f}（Δ@N中位 {beta*np.sqrt(Nmed):.1f} 點）\n')
        P('| κ | 可交易日比例 | x* 中位 (口) | x* 最大 | 年化損益 (點·口) | 年化 (萬 TWD) |'); P('|---|---|---|---|---|---|')
        for kappa in [1.0, 0.7, 0.5, 0.3]:
            tot, days, xmed, xmax = capacity(beta, Vx, sx, kappa, Nser)
            P(f'| {kappa} | {days/nd:.0%} | {xmed:.0f} | {xmax:.0f} | {tot*252/nd:,.0f} | {tot*252/nd*M/1e4:,.0f} |')
        P('\n敏感度（κ=0.5，年化 萬 TWD）：')
        base = capacity(beta, Vx, sx, 0.5, Nser)[0] * 252 / nd * M / 1e4
        for lab, kw in [('Δ −30%', dict(dscale=0.7)), ('Δ +30%', dict(dscale=1.3)), ('V_w −30%', dict(vscale=0.7)), ('V_w +30%', dict(vscale=1.3))]:
            v = capacity(beta, Vx, sx, 0.5, Nser, **kw)[0] * 252 / nd * M / 1e4; P(f'- {lab}: {v:,.0f}（基準 {base:,.0f}）')
        P('')
# power
b, se, n = M1[('db_15', '全', 'a')]
P('## 檢定力\n'); P(f'M1 全樣本 13:45 raw：t = {b/se:.1f}（n={n}）。策略損益層：2026-05+ 淨 edge 若為 κΔ−c 且 σ_w={sw:.0f}，κ=0.5 時單筆 Sharpe ≈ {(0.5*b*np.sqrt(Nmed)-C)/sw:.3f}，達 t=2 需 {(2/max((0.5*b*np.sqrt(Nmed)-C)/sw,1e-6))**2:,.0f} 個交易日。')
d.write_parquet(R / 'daily4.parquet'); open(R / 'stage3.md', 'w').write('\n'.join(L)); print('\n'.join(L))

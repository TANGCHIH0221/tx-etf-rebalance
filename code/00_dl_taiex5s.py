"""Download TAIEX 5-second index (FinMind TaiwanVariousIndicators5Seconds) for trading days missing locally.
Keeps only 13:20:00-13:30:00 rows (need I_13:25 and close). One request per day; free-tier rate limit handled by backoff."""
import polars as pl, requests, re, time, glob, os, sys
from pathlib import Path
tok = re.search(r'FINMIND_TOKEN=(.*)', open('.env').read()).group(1).strip()
OUT = Path('data/raw/finmind_intraday/taiex_5s_tail'); OUT.mkdir(exist_ok=True)
days = pl.read_parquet('research/tx_etf_rebalance/out/daily.parquet')['date'].to_list()
have = {os.path.basename(p).split('date=')[-1][:10] for p in glob.glob('data/raw/finmind_intraday/taiex_5s/*')}
have |= {os.path.basename(p)[:10] for p in glob.glob('data/raw/finmind_intraday/index_5s/*')}
have |= {os.path.basename(p)[:10] for p in glob.glob(str(OUT / '*.parquet'))}
todo = [d for d in days if d not in have]; print('todo', len(todo), flush=True)
for i, d in enumerate(todo):
    for attempt in range(6):
        try:
            r = requests.get('https://api.finmindtrade.com/api/v4/data', params=dict(dataset='TaiwanVariousIndicators5Seconds', start_date=d, token=tok), timeout=60)
            j = r.json()
        except Exception as e:
            print(d, 'err', e, flush=True); time.sleep(30); continue
        if r.status_code == 200 and j.get('msg') == 'success':
            df = pl.DataFrame(j['data']) if j['data'] else pl.DataFrame({'date': [], 'TAIEX': []})
            if len(df):
                df = df.filter(pl.col('date').str.slice(11, 8) >= '13:20:00')
            df.write_parquet(OUT / f'{d}.parquet'); break
        else:
            print(d, r.status_code, j.get('msg'), flush=True); time.sleep(60 * (attempt + 1))
    if i % 50 == 0: print(i, d, time.strftime('%H:%M:%S'), flush=True)
    time.sleep(0.3)
print('done', flush=True)

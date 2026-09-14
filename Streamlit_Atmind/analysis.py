from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent

def load_data():
    df = pd.read_csv(ROOT / 'data/busy_buffet_clean.csv', dtype={'source_sheet': str, 'group_id': str})
    df['guest_type'] = df.guest_type.replace({'In house': 'Inhouse', 'Walk in': 'Walkin'})
    return df

def minutes(value):
    if pd.isna(value):
        return None
    h, m, *_ = str(value).split(':')
    return int(h) * 60 + int(m)

def queue_union(df):
    result = {}
    for day, rows in df[df.queue_observed == 1].groupby('source_sheet'):
        intervals = sorted((minutes(r.queue_start), minutes(r.queue_end)) for r in rows.itertuples()
                           if pd.notna(r.queue_start) and pd.notna(r.queue_end))
        merged = []
        for start, end in intervals:
            if end < start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        result[day] = merged
    return result

def cap_model(df, cap, days=None):
    unions = queue_union(df)
    meal = df[(df.guest_type == 'Walkin') & (df.meal_eligible == 1)]
    if days is not None:
        meal = meal[meal.source_sheet.isin(days)]
    affected = meal[meal.meal_minutes > cap]
    overlap = 0
    for r in affected.itertuples():
        start = minutes(r.meal_start_clean) + cap
        end = minutes(r.meal_end_clean)
        overlap += sum(max(0, min(end, b) - max(start, a)) for a, b in unions.get(r.source_sheet, []))
    return {'eligible': len(meal), 'affected': len(affected),
            'released': float((affected.meal_minutes - cap).sum()), 'overlap': overlap}

def daily_summary(df):
    return df.groupby('source_sheet', sort=False).agg(
        groups=('group_id', 'size'), queued=('queue_observed', 'sum'),
        walkaway=('walkaway', 'sum'), wait=('wait_minutes_strict', 'mean')).reset_index()

def queue_profile(df, day):
    rows = df[(df.source_sheet == day) & (df.queue_observed == 1)]
    output = []
    for t in range(360, 781, 5):
        for guest in ['Inhouse', 'Walkin']:
            n = sum(minutes(r.queue_start) <= t < minutes(r.queue_end) for r in rows.itertuples()
                    if r.guest_type == guest and pd.notna(r.queue_start) and pd.notna(r.queue_end))
            output.append({'เวลา': f'{t//60:02d}:{t%60:02d}', 'ประเภท': guest, 'กลุ่มที่รอ': n})
    return pd.DataFrame(output)

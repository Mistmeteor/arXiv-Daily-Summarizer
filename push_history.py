"""
Push-history tracking with spaced-repetition schedule.

Persisted to pushed_papers.json in this format:
    {
      "2401.12345": {
        "title": "...",
        "push_dates": ["2026-08-01", "2026-08-08"]
      },
      ...
    }

Schedule (anchored to the first push at day D):
    push #1 : day D                     (any day the paper first surfaces)
    push #2 : day D + REPUSH_OFFSETS_DAYS[0]   (default 7)
    push #3 : day D + REPUSH_OFFSETS_DAYS[1]   (default 30)
After TOTAL_PUSHES_CAP pushes the paper is permanently retired until pruned.
Entries with no push in the last LONG_TERM_RESET_DAYS are dropped entirely,
so the paper becomes eligible from scratch if arXiv ever surfaces it again.
"""

import json
import os
import re
from datetime import date, timedelta


PUSH_HISTORY_FILE = os.environ.get('PUSH_HISTORY_FILE', 'pushed_papers.json')
TOTAL_PUSHES_CAP = int(os.environ.get('TOTAL_PUSHES_CAP', '3'))
REPUSH_OFFSETS_DAYS = [7, 30]  # days after first push at which pushes #2 and #3 unlock
LONG_TERM_RESET_DAYS = int(os.environ.get('LONG_TERM_RESET_DAYS', '60'))


def paper_key(entry_id):
    """Normalize an arXiv entry_id/URL to a stable dedup key.

    'http://arxiv.org/abs/2401.12345v3' -> '2401.12345'
    """
    if not entry_id:
        return entry_id
    m = re.search(r'(\d{4}\.\d{4,5})(?:v\d+)?$', entry_id)
    if m:
        return m.group(1)
    tail = entry_id.rsplit('/', 1)[-1]
    return re.sub(r'v\d+$', '', tail)


def _parse_iso_date(s):
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _sorted_push_dates(entry):
    return sorted({d for d in (_parse_iso_date(x) for x in entry.get('push_dates', [])) if d})


def _next_allowed_date(dates):
    """Return the next date on which a push is allowed, or None if the paper
    has already hit TOTAL_PUSHES_CAP."""
    n = len(dates)
    if n >= TOTAL_PUSHES_CAP:
        return None
    if n == 0:
        return date.min  # first push always allowed
    offset_idx = n - 1
    if offset_idx >= len(REPUSH_OFFSETS_DAYS):
        return None  # schedule doesn't define this slot; treat as retired
    return dates[0] + timedelta(days=REPUSH_OFFSETS_DAYS[offset_idx])


def load(path=None):
    path = path or PUSH_HISTORY_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  ⚠️ Failed to load push history from {path}: {e}. Starting fresh.")
        return {}


def save(history, path=None):
    path = path or PUSH_HISTORY_FILE
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  💾 Push history saved to {path} ({len(history)} entries)")
    except IOError as e:
        print(f"  ⚠️ Failed to save push history: {e}")


def prune(history, today=None):
    """Drop entries whose most-recent push is older than LONG_TERM_RESET_DAYS."""
    if today is None:
        today = date.today()
    kept = {}
    for key, entry in history.items():
        dates = _sorted_push_dates(entry)
        if not dates or (today - dates[-1]).days >= LONG_TERM_RESET_DAYS:
            continue
        new_entry = dict(entry)
        new_entry['push_dates'] = [d.isoformat() for d in dates]
        kept[key] = new_entry
    return kept


def filter_papers(papers, history, today=None):
    """Split `papers` into (kept, skipped) per the schedule.

    A paper is kept if today >= its next-allowed-push date, and skipped if
    either the schedule cap is reached or its next window hasn't opened yet.
    """
    if today is None:
        today = date.today()
    kept, skipped = [], []
    for paper in papers:
        key = paper_key(paper.get('entry_id'))
        entry = history.get(key) or {}
        dates = _sorted_push_dates(entry)
        next_date = _next_allowed_date(dates)
        title = (paper.get('title') or '')[:60]
        if next_date is None:
            skipped.append(paper)
            last = dates[-1].isoformat() if dates else '?'
            print(f"  ⏩ Skip (retired after {len(dates)} pushes, last {last}): {title}...")
        elif today < next_date:
            skipped.append(paper)
            print(f"  ⏩ Skip (pushed {len(dates)}x, next window opens "
                  f"{next_date.isoformat()}): {title}...")
        else:
            kept.append(paper)
    return kept, skipped


def due_repush_ids(history, today=None):
    """arXiv IDs (normalized) whose scheduled repush window is open today and
    haven't hit the cap. Caller should re-fetch these by id_list because the
    natural recent-window arXiv search likely no longer surfaces them."""
    if today is None:
        today = date.today()
    due = []
    for key, entry in history.items():
        dates = _sorted_push_dates(entry)
        next_date = _next_allowed_date(dates)
        if next_date is None or dates == []:
            continue
        if today >= next_date:
            due.append(key)
    return due


def update(history, pushed_papers, today=None):
    """Record today's date as a push for each paper in `pushed_papers`."""
    if today is None:
        today = date.today()
    today_str = today.isoformat()
    for paper in pushed_papers:
        key = paper_key(paper.get('entry_id'))
        if not key:
            continue
        entry = history.setdefault(key, {'title': paper.get('title', ''), 'push_dates': []})
        entry['title'] = paper.get('title', entry.get('title', ''))
        if today_str not in entry['push_dates']:
            entry['push_dates'].append(today_str)
    return history

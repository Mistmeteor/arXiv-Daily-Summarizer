"""
Local backlog pool of econ.EM papers.

Persisted to econ_backlog.json. Updated whenever a daily run successfully
fetches from arXiv's econ.EM category, so that on days when the live
econ.EM request fails (HTTP 429 / 503 — common when arXiv rate-limits a
runner) or returns too few candidates to meet MIN_ECONOMETRICS_PER_PUSH,
the digest can still fill the "econ floor" from recently-seen papers that
haven't yet been pushed.

The pool is intentionally a short rolling window (default 180 days). The
goal is not "every econ.EM paper ever" — pushed_papers.json already serves
as the permanent archive. The pool is just a candidate list we can draw
from when arXiv's export API is temporarily unreachable.

File format:
    {
      "2601.01234": {
        "title": "...",
        "authors": "...",
        "abstract": "...",
        "pdf_url": "https://arxiv.org/pdf/2601.01234",
        "published": "2026-01-15T00:00:00+00:00",   # ISO; parsed back to datetime
        "categories": ["econ.EM", "stat.ME"],
        "entry_id": "http://arxiv.org/abs/2601.01234v1",
        "primary_category": "econ.EM",
        "quality_score": 7.2,
        "first_seen_at": "2026-09-10"               # earliest day we saw it
      },
      ...
    }
"""

import json
import os
from datetime import date, datetime, timedelta

import push_history


BACKLOG_FILE = os.environ.get('ECON_BACKLOG_FILE', 'econ_backlog.json')
BACKLOG_WINDOW_DAYS = int(os.environ.get('ECON_BACKLOG_WINDOW_DAYS', '180'))


def load(path=None):
    path = path or BACKLOG_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  ⚠️ Failed to load econ backlog from {path}: {e}. Starting fresh.")
        return {}


def save(backlog, path=None):
    path = path or BACKLOG_FILE
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(backlog, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  💾 econ backlog saved to {path} ({len(backlog)} entries)")
    except IOError as e:
        print(f"  ⚠️ Failed to save econ backlog: {e}")


def _paper_to_record(paper, today):
    pub = paper.get('published')
    if hasattr(pub, 'isoformat'):
        pub = pub.isoformat()
    return {
        'title': paper.get('title', ''),
        'authors': paper.get('authors', ''),
        'abstract': paper.get('abstract', ''),
        'pdf_url': paper.get('pdf_url', ''),
        'published': pub,
        'categories': list(paper.get('categories') or []),
        'entry_id': paper.get('entry_id', ''),
        'primary_category': paper.get('primary_category', 'econ.EM'),
        'quality_score': float(paper.get('quality_score', 0.0) or 0.0),
        'first_seen_at': today.isoformat(),
    }


def _record_to_paper(record):
    pub = record.get('published')
    if isinstance(pub, str):
        try:
            pub = datetime.fromisoformat(pub)
        except ValueError:
            pub = datetime.now()
    elif pub is None:
        pub = datetime.now()
    return {
        'title': record.get('title', ''),
        'authors': record.get('authors', ''),
        'abstract': record.get('abstract', ''),
        'pdf_url': record.get('pdf_url', ''),
        'published': pub,
        'categories': list(record.get('categories') or []),
        'entry_id': record.get('entry_id', ''),
        'primary_category': record.get('primary_category', 'econ.EM'),
        'quality_score': record.get('quality_score', 0.0),
    }


def upsert(backlog, papers, today=None):
    """Add econ.EM papers to the backlog. Keeps the earliest first_seen_at
    for entries we've seen before, refreshes other fields in case arXiv
    revised the record (new version, updated abstract, …)."""
    if today is None:
        today = date.today()
    added = 0
    for p in papers:
        key = push_history.paper_key(p.get('entry_id'))
        if not key:
            continue
        rec = _paper_to_record(p, today)
        if key in backlog:
            prev_seen = backlog[key].get('first_seen_at')
            if prev_seen:
                rec['first_seen_at'] = prev_seen
        else:
            added += 1
        backlog[key] = rec
    if added:
        print(f"  🧺 econ backlog: +{added} new entries (total {len(backlog)})")
    return backlog


def prune(backlog, today=None):
    """Drop entries first seen more than BACKLOG_WINDOW_DAYS ago."""
    if today is None:
        today = date.today()
    if BACKLOG_WINDOW_DAYS <= 0:
        return backlog
    cutoff = today - timedelta(days=BACKLOG_WINDOW_DAYS)
    kept = {}
    for key, rec in backlog.items():
        first_seen = rec.get('first_seen_at')
        try:
            seen_date = date.fromisoformat(first_seen) if first_seen else None
        except ValueError:
            seen_date = None
        if seen_date is None or seen_date >= cutoff:
            kept[key] = rec
    return kept


def _first_seen_ordinal(record):
    s = record.get('first_seen_at', '')
    try:
        return date.fromisoformat(s).toordinal() if s else 0
    except ValueError:
        return 0


def pick_fillers(backlog, history, need, exclude_ids=None, today=None):
    """Pick up to `need` econ.EM papers from the backlog that
       - are not already in today's candidate pool (exclude_ids),
       - pass the push_history schedule (not retired, not in cooldown).
    Order: never-pushed first, then most-recently-seen first."""
    if need <= 0:
        return []
    if today is None:
        today = date.today()
    exclude_ids = exclude_ids or set()
    exclude_keys = {push_history.paper_key(x) for x in exclude_ids}

    scored = []
    for key, rec in backlog.items():
        if key in exclude_keys:
            continue
        entry = history.get(key) or {}
        dates = push_history._sorted_push_dates(entry)
        next_date = push_history._next_allowed_date(dates)
        if next_date is None:
            continue
        if today < next_date:
            continue
        scored.append((len(dates), -_first_seen_ordinal(rec), rec))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [_record_to_paper(x[2]) for x in scored[:need]]

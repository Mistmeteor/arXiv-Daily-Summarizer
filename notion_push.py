"""Push arXiv digest papers as rows to a Notion database.

Secondary push channel alongside PushPlus. Users in latency-sensitive regions
(e.g. Japan → Chinese PushPlus server) can open Notion instead of waiting for
the WeChat webview.

Adaptive schema: queries the target database once, auto-detects the title
property, and only sets known columns that actually exist. Name matching is
whitespace- and case-insensitive (so "arXiv编号" == "arXiv 编号" == "ARXIV ID").
Type matching prefers the semantic type below but falls back to rich_text if
that's what the user set — the value is written as a string in that case.

  - <title col>            (title)                       paper title  [required]
  - Category   / 分类       (multi_select)                arxiv category codes
  - Date       / 日期       (date)                        arxiv published date
  - arXiv ID   / arXiv 编号 (rich_text)                   arxiv id (e.g. 2609.12345)
  - URL        / 链接/连接  (url, or rich_text fallback)  arxiv abs page link
  - Quality    / 评分       (number, or rich_text)        quality_score
  - Badges     / 标签       (multi_select, or rich_text)  BLP / Pinned / Repush

AI summary is written to the page body as paragraph blocks only — the DB list
view stays scannable (open the page to read the summary). A rich_text
Summary/摘要 column, if present, is intentionally left empty.

Uses only stdlib (urllib). Never raises — the caller treats this as a best-
effort sink so a Notion outage cannot take down the PushPlus path.
"""
import json
import re
import time
import urllib.error
import urllib.request

NOTION_API_VERSION = '2022-06-28'
NOTION_API_ROOT = 'https://api.notion.com/v1'
BLOCK_TEXT_LIMIT = 1900


def _request(method, url, token, payload=None, timeout=30):
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    headers = {
        'Authorization': f'Bearer {token}',
        'Notion-Version': NOTION_API_VERSION,
        'Content-Type': 'application/json',
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8', errors='replace'))


def _fetch_schema(token, database_id):
    resp = _request('GET', f'{NOTION_API_ROOT}/databases/{database_id}', token)
    props = resp.get('properties', {})
    return {name: meta.get('type') for name, meta in props.items()}


def _find_title_prop(schema):
    for name, ptype in schema.items():
        if ptype == 'title':
            return name
    return None


def _norm(s):
    """Normalize a column name for matching: strip whitespace, lowercase."""
    return re.sub(r'\s+', '', s or '').lower()


def _match(schema, candidates, expected_type):
    lower_map = {_norm(n): n for n in schema}
    for cand in candidates:
        real = lower_map.get(_norm(cand))
        if real and schema.get(real) == expected_type:
            return real
    return None


def _match_flex(schema, candidates, preferred_type):
    """Match by name (whitespace-insensitive) and return (name, actual_type).

    Prefers columns whose actual type is `preferred_type`. If none match by
    name at that type, falls back to any column matching by name with type
    `rich_text` — so users who set the column to text still get a value
    written (as a string) instead of a silent skip.
    """
    lower_map = {_norm(n): n for n in schema}
    # Preferred type first
    for cand in candidates:
        real = lower_map.get(_norm(cand))
        if real and schema.get(real) == preferred_type:
            return real, preferred_type
    # rich_text fallback
    for cand in candidates:
        real = lower_map.get(_norm(cand))
        if real and schema.get(real) == 'rich_text':
            return real, 'rich_text'
    return None, None


def _extract_arxiv_id(entry_id):
    if not entry_id:
        return ''
    m = re.search(r'/abs/([^/]+?)(?:v\d+)?$', entry_id)
    return m.group(1) if m else entry_id


def _chunk_text(text, limit):
    text = text or ''
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = text.rfind('\n', 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    return chunks


def _summary_text(summary):
    if isinstance(summary, dict):
        return summary.get('zh') or summary.get('en') or ''
    return summary or ''


def _build_properties(paper, summary, schema, title_prop):
    props = {
        title_prop: {
            'title': [{'text': {'content': (paper.get('title') or '')[:2000]}}]
        }
    }

    cat_prop = _match(schema, ['Category', 'Categories', '分类', '类别', 'Tag', 'Tags'], 'multi_select')
    if cat_prop:
        cats = list(paper.get('categories') or [])[:5]
        if cats:
            props[cat_prop] = {'multi_select': [{'name': c} for c in cats]}

    date_prop = _match(schema, ['Date', 'Published', 'Publish Date', '日期', '发布日期'], 'date')
    if date_prop and paper.get('published'):
        props[date_prop] = {'date': {'start': paper['published'].date().isoformat()}}

    id_prop = _match(schema, ['arXiv ID', 'arXivID', 'arXiv', 'ID', 'arXiv 编号', '编号'], 'rich_text')
    if id_prop:
        aid = _extract_arxiv_id(paper.get('entry_id'))
        props[id_prop] = {'rich_text': [{'text': {'content': aid[:2000]}}]}

    url_prop, url_type = _match_flex(
        schema, ['URL', 'Link', 'arXiv URL', '链接', '连接'], 'url'
    )
    if url_prop:
        url = paper.get('entry_id') or paper.get('pdf_url')
        if url:
            if url_type == 'url':
                props[url_prop] = {'url': url}
            else:  # rich_text fallback
                props[url_prop] = {'rich_text': [{'text': {'content': url[:2000]}}]}

    quality_prop, quality_type = _match_flex(
        schema, ['Quality', 'Score', 'Quality Score', '评分', '质量'], 'number'
    )
    if quality_prop:
        value = round(paper.get('quality_score', 0), 2)
        if quality_type == 'number':
            props[quality_prop] = {'number': value}
        else:  # rich_text fallback
            props[quality_prop] = {'rich_text': [{'text': {'content': str(value)}}]}

    # Summary text goes into the page body via _build_children, not into a
    # property column — keeps the DB list view scannable.

    badges_prop, badges_type = _match_flex(
        schema, ['Badges', 'Flags', 'Flag', '标签', '徽章'], 'multi_select'
    )
    if badges_prop:
        tags = []
        if paper.get('is_blp_recommend'):
            tags.append('BLP')
        if paper.get('is_pinned'):
            tags.append('Pinned')
        if paper.get('is_repush'):
            tags.append('Repush')
        if tags:
            if badges_type == 'multi_select':
                props[badges_prop] = {'multi_select': [{'name': t} for t in tags]}
            else:  # rich_text fallback
                props[badges_prop] = {
                    'rich_text': [{'text': {'content': ', '.join(tags)}}]
                }

    return props


def _build_children(summary):
    text = _summary_text(summary)
    if not text:
        return []
    return [
        {
            'object': 'block',
            'type': 'paragraph',
            'paragraph': {
                'rich_text': [{'type': 'text', 'text': {'content': chunk}}]
            },
        }
        for chunk in _chunk_text(text, BLOCK_TEXT_LIMIT)
    ]


def _query_existing_ids(token, database_id, id_prop):
    """Return the set of arXiv IDs already present in the DB (best effort).

    Paginates through every row and reads the id_prop rich_text value. If the
    query fails we return None so the caller falls back to creating without
    dedup — a duplicate is less bad than skipping the push entirely.
    """
    existing = set()
    cursor = None
    try:
        for _ in range(50):
            payload = {'page_size': 100}
            if cursor:
                payload['start_cursor'] = cursor
            resp = _request(
                'POST',
                f'{NOTION_API_ROOT}/databases/{database_id}/query',
                token,
                payload,
            )
            for row in resp.get('results', []):
                prop = row.get('properties', {}).get(id_prop, {})
                for span in prop.get('rich_text', []):
                    val = (span.get('plain_text') or '').strip()
                    if val:
                        existing.add(val)
                        break
            if not resp.get('has_more'):
                break
            cursor = resp.get('next_cursor')
    except Exception as e:
        print(f"  ⚠️ Notion dedup query failed ({e}); will create without dedup.")
        return None
    return existing


def _create_page(token, database_id, properties, children, attempts=3):
    payload = {'parent': {'database_id': database_id}, 'properties': properties}
    if children:
        payload['children'] = children[:100]
    backoff = 3
    for i in range(1, attempts + 1):
        try:
            return _request('POST', f'{NOTION_API_ROOT}/pages', token, payload)
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')
            print(f"  ⚠️ Notion create failed ({i}/{attempts}): HTTP {e.code} {body[:200]}")
            if e.code in (429, 500, 502, 503, 504) and i < attempts:
                time.sleep(backoff)
                backoff *= 2
                continue
            return None
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  ⚠️ Notion create network error ({i}/{attempts}): {e}")
            if i < attempts:
                time.sleep(backoff)
                backoff *= 2
                continue
            return None
    return None


def push_to_notion(papers_with_summaries, token, database_id):
    """Create one Notion page per paper. Returns (ok_count, fail_count)."""
    if not token or not database_id:
        print("\n📓 Notion: NOTION_API_KEY or NOTION_DATABASE_ID missing, skipping.")
        return 0, 0

    print(f"\n📓 Pushing {len(papers_with_summaries)} paper(s) to Notion...")

    try:
        schema = _fetch_schema(token, database_id)
    except Exception as e:
        print(f"❌ Notion: failed to fetch database schema ({e}). Skipping.")
        return 0, len(papers_with_summaries)

    title_prop = _find_title_prop(schema)
    if not title_prop:
        print("❌ Notion: target database has no title property. Skipping.")
        return 0, len(papers_with_summaries)

    id_prop = _match(schema, ['arXiv ID', 'arXivID', 'arXiv', 'ID', 'arXiv 编号', '编号'], 'rich_text')
    existing_ids = _query_existing_ids(token, database_id, id_prop) if id_prop else None

    other_cols = [n for n, t in schema.items() if t != 'title']
    print(f"  Title column: '{title_prop}'. Other columns: {other_cols}")
    if existing_ids is not None:
        print(f"  Dedup on '{id_prop}': {len(existing_ids)} existing row(s).")

    ok = fail = skipped = 0
    total = len(papers_with_summaries)
    for i, item in enumerate(papers_with_summaries, 1):
        paper = item['paper']
        summary = item['summary']
        aid = _extract_arxiv_id(paper.get('entry_id'))

        if existing_ids is not None and aid and aid in existing_ids:
            skipped += 1
            print(f"  ⏭️  [{i}/{total}] {paper['title'][:60]}... (already in Notion)")
            continue

        props = _build_properties(paper, summary, schema, title_prop)
        children = _build_children(summary)
        result = _create_page(token, database_id, props, children)
        if result:
            ok += 1
            print(f"  ✅ [{i}/{total}] {paper['title'][:60]}...")
            if existing_ids is not None and aid:
                existing_ids.add(aid)
        else:
            fail += 1
            print(f"  ❌ [{i}/{total}] {paper['title'][:60]}...")

    print(f"📓 Notion done: {ok} created, {skipped} skipped (dedup), {fail} failed.")
    return ok, fail

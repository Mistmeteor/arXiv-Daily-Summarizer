import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import arxiv
from datetime import datetime, timedelta, date
from openai import OpenAI
from collections import Counter, defaultdict
import re
from difflib import SequenceMatcher

import push_history

# ========== Configuration ==========

# arXiv search configuration
# Categories were expanded beyond pure econ.EM so the digest can rotate through
# BLP-adjacent math/stat/ML work instead of only surfacing econometrics twice
# (see CATEGORY_WEIGHTS for the sampling mix).
CATEGORIES = [
    'econ.EM',   # Econometrics — primary field, weighted heaviest
    'math.OC',   # Optimization & control (non-convex opt underlies BLP estimation)
    'math.ST',   # Mathematical statistics (BLP inference theory)
    'stat.ML',   # Statistical machine learning
    'stat.ME',   # Statistical methodology (semi-parametric / IV methods live here)
    'stat.TH',   # Statistical theory
    'cs.LG',     # Machine learning (filtered to econ/stat cross-listings only)
]
PRIMARY_CATEGORY = 'econ.EM'  # Weighted heaviest in the sampler; also gets the pin badge

# Weighted-random category mix used when building the candidate pool. Higher
# weight → more slots on average. econ.EM dominates, but on days with sparse
# econ output the other fields naturally fill in instead of forcing 2nd/3rd
# pushes of the same econometrics papers.
CATEGORY_WEIGHTS = {
    'econ.EM': 5.0,
    'stat.ME': 1.2,
    'math.ST': 1.0,
    'stat.ML': 1.0,
    'stat.TH': 0.8,
    'math.OC': 0.6,
    'cs.LG':   0.4,
}

# For non-primary categories, a paper is kept only if at least one of its
# categories is in this "relevant to BLP / econometrics research" set. Since
# math.ST / stat.* are themselves in the set, papers primary-listed there pass
# trivially; this filter really only prunes pure cs.LG (or future edge
# additions) that have no stat/econ/math.ST/math.OC cross-listing.
RELEVANT_CATS = {
    'econ.EM', 'econ.TH', 'econ.GN',
    'stat.ML', 'stat.ME', 'stat.TH', 'stat.AP', 'stat.CO',
    'math.ST', 'math.OC', 'math.PR', 'math.NA',
    'q-fin.EC', 'q-fin.ST', 'q-fin.RM',
}

MAX_RESULTS = 10  # Daily cap (final email size, applied AFTER push-history filter)
MIN_PAPERS_PER_CATEGORY = 1  # (kept for backward compat; not used by the sampler below)

# Pre-filter candidate pool: rank this many papers BEFORE push-history filtering.
# When today's recent Top-MAX_RESULTS are all in the "already-pushed" cooldown
# window, the extra 2×MAX_RESULTS candidates give the schedule filter something
# to fall back on (older papers that haven't been pushed yet).
CANDIDATE_POOL_SIZE = MAX_RESULTS * 3

# Per-category cap fed into the weighted sampler. Prevents any single category
# from dumping its full FETCH_DEPTH_MULTIPLIER×MAX_RESULTS backlog into the
# sampler pool; we only sample from each category's top-quality slice.
PER_CATEGORY_SAMPLING_POOL = MAX_RESULTS * 2

# arXiv fetch depth per category, in units of MAX_RESULTS. 9× ≈ 90 papers per
# category; for econ.EM's ~3 papers/day this covers roughly the last month —
# wide enough to backfill from unpushed older papers, not so wide that we're
# trawling ancient history.
FETCH_DEPTH_MULTIPLIER = 9

# Language configuration
# Supported values: 'zh' (Chinese), 'en' (English), 'both' (Bilingual)
EMAIL_LANGUAGE = os.environ.get('EMAIL_LANGUAGE', 'zh')  # Default to Chinese

# DeepSeek API configuration (official DeepSeek platform)
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = 'https://api.deepseek.com/v1'
DEEPSEEK_MODEL = 'deepseek-chat'  # points to the latest DeepSeek-V3

# PushPlus (WeChat push) configuration. Replaces SMTP because QQ Mail
# blocks GitHub Actions IPs even on 465/SSL.
PUSHPLUS_TOKEN = os.environ.get('PUSHPLUS_TOKEN')
PUSHPLUS_URL = 'https://www.pushplus.plus/send'

# Quality filtering thresholds
MIN_ABSTRACT_LENGTH = 100  # Minimum abstract length (characters)
SIMILARITY_THRESHOLD = 0.85  # Title similarity threshold for duplicate detection

# Push-history / dedup schedule lives in push_history.py (spaced-repetition curve:
# first push on day D, repush unlocks at D+7 and D+30, capped at 3 total pushes).

# BLP / demand-estimation strong-recommend keywords. Papers whose title or
# abstract match any of these are pinned to the very top of the digest and
# rendered with a red "🔥 强推荐" badge. Matching is lowercase substring.
BLP_KEYWORDS = [
    # Core BLP
    'blp', 'berry-levinsohn', 'berry levinsohn', 'levinsohn-pakes',
    'berry, levinsohn', 'berry and levinsohn', 'pyblp',
    'random coefficient', 'random-coefficient', 'random coefficients',
    # Demand estimation
    'demand estimation', 'demand model', 'demand system',
    'discrete choice', 'discrete-choice',
    # Structural IO adjacent
    'differentiated product', 'differentiated products',
    'merger simulation', 'merger analysis',
    'nested logit', 'mixed logit',
    'micro moments', 'micro-moments',
    'consumer heterogeneity', 'characteristic space',
    'market equilibrium',
]


def matches_blp(paper):
    """True if the paper's title or abstract mentions any BLP-related keyword."""
    text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
    return any(kw in text for kw in BLP_KEYWORDS)


def _priority_tier(paper):
    """0 = BLP strong-recommend (always on top), 1 = everything else.

    Deliberately flat below the BLP tier so the weighted-random mix isn't
    re-collapsed back into an econ-first ordering at display time.
    """
    if matches_blp(paper):
        return 0
    return 1


def _is_relevant_paper(paper):
    """Whether a non-primary-category paper is topically relevant enough to keep.

    Used to prune pure cs.LG (or any future non-core-relevance category) papers
    that have no stat/econ/math.ST/math.OC cross-listing. Categories that are
    themselves in RELEVANT_CATS (math.ST, stat.*, econ.*) pass trivially.
    """
    return any(c in RELEVANT_CATS for c in paper.get('categories', []))


def fetch_papers_by_ids(id_list):
    """Fetch paper metadata for a list of normalized arXiv IDs (e.g. '2401.12345').

    Used for scheduled repushes: papers whose D+7 / D+30 window has come due
    are usually no longer surfaced by the natural recent-window search, so we
    look them up directly.
    """
    if not id_list:
        return []
    client = arxiv.Client(page_size=100, delay_seconds=5.0, num_retries=5)
    search = arxiv.Search(id_list=list(id_list))
    out = []
    try:
        for result in client.results(search):
            paper = {
                'title': result.title,
                'authors': ', '.join([a.name for a in result.authors]),
                'abstract': result.summary if hasattr(result, 'summary') else '',
                'pdf_url': result.pdf_url,
                'published': result.published,
                'categories': result.categories,
                'entry_id': result.entry_id,
                'primary_category': (list(result.categories) or [PRIMARY_CATEGORY])[0],
                'is_repush': True,
            }
            paper['quality_score'] = calculate_paper_quality_score(paper)
            out.append(paper)
    except Exception as e:
        print(f"  ⚠️ Failed to fetch repush papers by id: {e}")
    return out

# arXiv category → Chinese label. Used to render category tags in the email.
# Unlisted categories fall back to the raw arXiv code.
CATEGORY_LABELS = {
    # Economics
    'econ.EM': '计量经济学',
    'econ.TH': '理论经济学',
    'econ.GN': '一般经济学',
    # Mathematics
    'math.OC': '优化与控制',
    'math.ST': '数理统计',
    'math.NA': '数值分析',
    'math.PR': '概率论',
    # Statistics
    'stat.ML': '统计机器学习',
    'stat.ME': '统计方法学',
    'stat.AP': '统计应用',
    'stat.TH': '统计理论',
    'stat.CO': '计算统计学',
    # Computer science
    'cs.LG': '机器学习',
    'cs.AI': '人工智能',
    'cs.CV': '计算机视觉',
    'cs.CL': '计算语言学',
    'cs.NA': '数值分析(CS)',
    # Quantitative finance
    'q-fin.EC': '金融经济学',
    'q-fin.ST': '金融统计',
    'q-fin.RM': '风险管理',
    'q-fin.PM': '投资组合管理',
    'q-fin.CP': '计算金融',
    'q-fin.MF': '数理金融',
    'q-fin.PR': '金融衍生品定价',
    'q-fin.TR': '金融交易',
    'q-fin.GN': '一般金融',
}

# Language text templates
TEXT_TEMPLATES = {
    'zh': {
        'title': 'arXiv 每日论文推送',
        'date_notice': '论文日期提醒',
        'today': '今天',
        'yesterday': '昨天',
        'days_ago': '天前',
        'published_today': '<strong>{count} 篇</strong>是今天发布',
        'published_yesterday': '<strong>{count} 篇</strong>是昨天发布',
        'published_older_single': '<strong>{count} 篇</strong>是 {days} 天前发布（可能已读过）',
        'published_older_multi': '<strong>{count} 篇</strong>是 2 天及更早前发布（可能已读过）',
        'notice_text': '本次推送的 {total} 篇论文中，{parts}。',
        'new_today': '今日新发布',
        'yesterday_label': '昨日发布',
        'days_ago_label': '{days} 天前',
        'high_quality': '⭐ 高质量',
        'pinned': '📌 置顶',
        'blp_recommend': '🔥 强推荐',
        'repush': '♻️ 二次推送',
        'authors': '作者',
        'published': '发布日期',
        'categories': '分类',
        'quality_score': '质量评分',
        'ai_summary': 'AI 摘要',
        'view_pdf': '查看 PDF',
        'footer_auto': '本邮件由 arXiv Daily Summarizer 自动生成',
        'footer_powered': '由 DeepSeek AI 提供摘要服务'
    },
    'en': {
        'title': 'arXiv Daily Paper Digest',
        'date_notice': 'Date Notice',
        'today': 'today',
        'yesterday': 'yesterday',
        'days_ago': 'days ago',
        'published_today': '<strong>{count} papers</strong> published today',
        'published_yesterday': '<strong>{count} papers</strong> published yesterday',
        'published_older_single': '<strong>{count} paper</strong> published {days} days ago (may have been read)',
        'published_older_multi': '<strong>{count} papers</strong> published 2+ days ago (may have been read)',
        'notice_text': 'Of the {total} papers in this digest, {parts}.',
        'new_today': 'NEW TODAY',
        'yesterday_label': 'YESTERDAY',
        'days_ago_label': '{days} DAYS AGO',
        'high_quality': '⭐ HIGH QUALITY',
        'pinned': '📌 PINNED',
        'blp_recommend': '🔥 STRONGLY RECOMMENDED',
        'repush': '♻️ REPUSH',
        'authors': 'Authors',
        'published': 'Published',
        'categories': 'Categories',
        'quality_score': 'Quality Score',
        'ai_summary': 'AI Summary',
        'view_pdf': 'View PDF',
        'footer_auto': 'Generated automatically by arXiv Daily Summarizer',
        'footer_powered': 'Powered by DeepSeek AI'
    }
}


def calculate_paper_quality_score(paper):
    """
    Calculate a quality score for a paper based on various factors
    
    Args:
        paper: Dictionary containing paper information
        
    Returns:
        float: Quality score (higher is better)
    """
    score = 0.0
    
    # Factor 1: Abstract length (longer abstracts usually indicate more detailed work)
    abstract_length = len(paper.get('abstract', ''))
    if abstract_length > 500:
        score += 2.0
    elif abstract_length > 300:
        score += 1.0
    elif abstract_length < MIN_ABSTRACT_LENGTH:
        score -= 2.0  # Penalize very short abstracts
    
    # Factor 2: Number of authors (more authors might indicate collaborative/important work)
    num_authors = len(paper.get('authors', '').split(','))
    if 3 <= num_authors <= 8:
        score += 1.0
    elif num_authors > 8:
        score += 0.5
    
    # Factor 3: Title characteristics
    title = paper.get('title', '').lower()
    
    # Bonus for important keywords
    important_keywords = [
        'novel', 'efficient', 'state-of-the-art', 'breakthrough', 'improved',
        'transformer', 'attention', 'neural', 'deep learning', 'framework',
        'benchmark', 'dataset', 'evaluation', 'survey', 'review'
    ]
    for keyword in important_keywords:
        if keyword in title:
            score += 0.5
    
    # Penalty for very short or very long titles
    title_words = len(title.split())
    if title_words < 5:
        score -= 0.5
    elif title_words > 25:
        score -= 0.3
    
    # Factor 4: Recency bonus (newer papers get higher scores)
    # Make datetime timezone-aware for comparison
    now = datetime.now(paper['published'].tzinfo)
    days_old = (now - paper['published']).days
    if days_old == 0:
        score += 3.0  # Strong bonus for today's papers
    elif days_old == 1:
        score += 1.5
    elif days_old == 2:
        score += 0.5
    else:
        score -= (days_old - 2) * 0.3  # Penalty for older papers
    
    return score


def calculate_title_similarity(title1, title2):
    """
    Calculate similarity between two paper titles
    
    Args:
        title1: First title string
        title2: Second title string
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Normalize titles: lowercase and remove special characters
    def normalize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    norm_title1 = normalize(title1)
    norm_title2 = normalize(title2)
    
    return SequenceMatcher(None, norm_title1, norm_title2).ratio()


def remove_duplicate_papers(papers):
    """
    Remove duplicate or very similar papers based on title similarity
    
    Args:
        papers: List of paper dictionaries
        
    Returns:
        list: Filtered list without duplicates
    """
    if not papers:
        return papers
    
    filtered_papers = []
    
    for paper in papers:
        is_duplicate = False
        
        for existing_paper in filtered_papers:
            similarity = calculate_title_similarity(
                paper['title'], 
                existing_paper['title']
            )
            
            if similarity >= SIMILARITY_THRESHOLD:
                print(f"  🔄 Detected similar paper (similarity: {similarity:.2f}):")
                print(f"     Original: {existing_paper['title'][:60]}...")
                print(f"     Duplicate: {paper['title'][:60]}...")
                
                # Keep the one with higher quality score
                if paper.get('quality_score', 0) > existing_paper.get('quality_score', 0):
                    filtered_papers.remove(existing_paper)
                    filtered_papers.append(paper)
                    print(f"     → Kept the higher quality version")
                else:
                    print(f"     → Skipped duplicate")
                
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_papers.append(paper)
    
    return filtered_papers


def get_latest_papers():
    """
    Fetch latest papers from arXiv with quality filtering and deduplication
    
    Returns:
        list: List of selected paper dictionaries
    """
    print(f"🔍 Searching for latest papers on arXiv...")
    print(f"📚 Categories: {', '.join(CATEGORIES)}")
    
    client = arxiv.Client(page_size=100, delay_seconds=5.0, num_retries=5)
    papers_by_category = defaultdict(list)
    seen_ids = set()
    
    # Step 1: Fetch papers from each category separately
    for category in CATEGORIES:
        print(f"\n🔎 Searching category: {category}")
        
        try:
            # Fetch a wide window (~last month for econ.EM at ~3 papers/day) so
            # that when the push-history filter later drops today's Top-N as
            # "already pushed in the cooldown window", we still have unpushed
            # older papers ranked below them to backfill from.
            search = arxiv.Search(
                query=f'cat:{category}',
                max_results=MAX_RESULTS * FETCH_DEPTH_MULTIPLIER,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            results = list(client.results(search))
            print(f"  API returned {len(results)} papers")
            
            for result in results:
                if result.entry_id not in seen_ids:
                    # Soft relevance filter for non-primary categories: keep the
                    # paper only if at least one of its arXiv categories is in
                    # the BLP-relevance set (RELEVANT_CATS). Papers in math.ST /
                    # stat.* pass trivially; this really only prunes pure cs.LG
                    # / edge additions with no stat/econ/math.ST cross-listing.
                    if category != PRIMARY_CATEGORY:
                        cats = list(result.categories)
                        if not any(c in RELEVANT_CATS for c in cats):
                            continue

                    seen_ids.add(result.entry_id)

                    abstract_text = result.summary if hasattr(result, 'summary') else ''

                    paper = {
                        'title': result.title,
                        'authors': ', '.join([author.name for author in result.authors]),
                        'abstract': abstract_text,
                        'pdf_url': result.pdf_url,
                        'published': result.published,
                        'categories': result.categories,
                        'entry_id': result.entry_id,
                        'primary_category': category
                    }

                    # Calculate quality score
                    paper['quality_score'] = calculate_paper_quality_score(paper)

                    papers_by_category[category].append(paper)

                    print(f"  ✓ {result.title[:60]}... (score: {paper['quality_score']:.1f})")
            
            # Sort papers in this category by quality score
            papers_by_category[category].sort(
                key=lambda x: x['quality_score'], 
                reverse=True
            )
            
            print(f"  Found {len(papers_by_category[category])} papers in {category}")
            
        except Exception as e:
            print(f"  ❌ Error searching {category}: {str(e)}")
            continue
    
    # Step 2: Build the candidate pool via BLP-first + weighted-random rotation.
    #
    # Why not primary-first anymore: with econ.EM's ~3 papers/day, "primary first"
    # + push-history dedup was collapsing most emails into 2nd/3rd repushes of the
    # same econometrics papers. Weighted-random mixing keeps econ.EM dominant
    # (highest weight) but naturally injects math/stat variety on slow econ days.
    #
    # The RNG is seeded by today's date, so a given day is reproducible (useful
    # for debugging) while consecutive days differ (the point of the mix).
    print(f"\n⚖️ Building candidate pool (weighted-random mix, pool={CANDIDATE_POOL_SIZE})...")
    rng = random.Random(date.today().isoformat())

    # Slice each category to its top-quality PER_CATEGORY_SAMPLING_POOL so no
    # single high-volume category (e.g. cs.LG) can drown the sampler.
    sampling_pools = {
        cat: list(papers_by_category.get(cat, []))[:PER_CATEGORY_SAMPLING_POOL]
        for cat in CATEGORIES
    }

    selected_papers = []
    seen_entry_ids = set()

    # 2a. BLP strong-recommend papers are pinned into the pool unconditionally,
    #     regardless of which category surfaced them. These are the highest
    #     signal for this specific researcher.
    for cat in CATEGORIES:
        for p in sampling_pools[cat]:
            if matches_blp(p) and p['entry_id'] not in seen_entry_ids:
                selected_papers.append(p)
                seen_entry_ids.add(p['entry_id'])
                if len(selected_papers) >= CANDIDATE_POOL_SIZE:
                    break
        if len(selected_papers) >= CANDIDATE_POOL_SIZE:
            break
    if selected_papers:
        print(f"  Pinned {len(selected_papers)} BLP strong-recommend paper(s)")

    # 2b. Weighted-random rotation fills the remaining slots. Each iteration
    #     samples a category by CATEGORY_WEIGHTS, then pulls the highest-quality
    #     unused paper from that category. When a category's sampling pool is
    #     exhausted, drop it from the weight table and re-sample.
    active_weights = {
        cat: w for cat, w in CATEGORY_WEIGHTS.items()
        if sampling_pools.get(cat)
    }
    while len(selected_papers) < CANDIDATE_POOL_SIZE and active_weights:
        cats = list(active_weights.keys())
        weights = [active_weights[c] for c in cats]
        chosen_cat = rng.choices(cats, weights=weights, k=1)[0]
        pool = sampling_pools[chosen_cat]
        picked = None
        for p in pool:
            if p['entry_id'] not in seen_entry_ids:
                picked = p
                break
        if picked is None:
            del active_weights[chosen_cat]
            continue
        selected_papers.append(picked)
        seen_entry_ids.add(picked['entry_id'])

    # Step 3: Remove duplicates using intelligent similarity detection
    print(f"\n🔍 Checking for duplicate/similar papers...")
    selected_papers = remove_duplicate_papers(selected_papers)

    # Step 4: Final sort. Priority tiers (0 = highest):
    #   0. BLP / demand-estimation match (strong recommend, always on top)
    #   1. everything else — sorted purely by quality DESC, so the deliberately
    #      mixed field selection actually renders as a mix (no more econ-first
    #      collapse at display time)
    selected_papers.sort(key=lambda x: (
        _priority_tier(x),
        -x.get('quality_score', 0),
        -x['published'].timestamp()
    ))

    # Tag papers for the email renderer. is_pinned now means "arXiv-declared
    # primary category is econ.EM", i.e. this paper is truly econometrics-first
    # (not merely cross-listed). Under the mix, this is a useful visual cue.
    for paper in selected_papers:
        arxiv_primary = (list(paper.get('categories', [])) or [None])[0]
        paper['is_pinned'] = arxiv_primary == PRIMARY_CATEGORY
        paper['is_blp_recommend'] = matches_blp(paper)
    
    print(f"\n✅ Candidate pool built: {len(selected_papers)} papers "
          f"(will be filtered by push history, then capped to {MAX_RESULTS})")
    
    # Print category distribution
    category_dist = Counter([p['primary_category'] for p in selected_papers])
    print(f"\n📊 Category distribution:")
    for cat, count in category_dist.items():
        print(f"   {cat}: {count} papers")
    
    return selected_papers


def analyze_paper_dates(papers):
    """
    Analyze the publication date distribution of papers
    
    Args:
        papers: List of paper dictionaries
        
    Returns:
        dict: Statistics about paper dates
    """
    now = datetime.now()
    today = now.date()
    yesterday = (now - timedelta(days=1)).date()
    
    date_stats = {
        'today': 0,
        'yesterday': 0,
        'older': 0,
        'date_distribution': Counter()
    }
    
    for paper in papers:
        paper_date = paper['published'].date()
        date_stats['date_distribution'][paper_date] += 1
        
        if paper_date == today:
            date_stats['today'] += 1
        elif paper_date == yesterday:
            date_stats['yesterday'] += 1
        else:
            date_stats['older'] += 1
    
    return date_stats


def strip_markdown(text):
    """
    Safety net: strip common Markdown syntax from AI-generated summaries so
    they render cleanly in the plain-body email. Preserves LaTeX math (\\(...\\))
    which is domain-meaningful even if not visually rendered.
    """
    if not text:
        return text
    # Remove ATX headers at line starts: "### foo" -> "foo"
    text = re.sub(r'^\s{0,3}#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Bold: **foo** / __foo__ -> foo
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    # Italic: *foo* / _foo_ -> foo (avoid matching inside numbers like a_1)
    text = re.sub(r'(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<![A-Za-z0-9_])_(?!\s)([^_\n]+?)(?<!\s)_(?![A-Za-z0-9_])', r'\1', text)
    # Inline code: `foo` -> foo
    text = re.sub(r'`([^`\n]+?)`', r'\1', text)
    # Blockquote markers at line starts
    text = re.sub(r'^\s{0,3}>\s?', '', text, flags=re.MULTILINE)
    # Unordered list bullets at line starts: "- foo" / "* foo" / "+ foo" -> "foo"
    text = re.sub(r'^\s{0,3}[-*+]\s+', '', text, flags=re.MULTILINE)
    # Normalize whitespace: strip trailing spaces on each line, then collapse
    # 2+ consecutive newlines down to 1 (DeepSeek often uses \n\n between sections,
    # which turns into <br><br> = extra blank line in the email)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def summarize_paper(paper, language='zh'):
    """
    Generate paper summary using DeepSeek AI
    
    Args:
        paper: Dictionary containing paper information
        language: 'zh' for Chinese, 'en' for English, 'both' for bilingual
        
    Returns:
        dict: AI-generated summaries {'zh': str, 'en': str} or single language str
    """
    print(f"\n🤖 Generating AI summary for:")
    print(f"   {paper['title'][:70]}...")
    
    summaries = {}
    
    # Define prompts for each language
    # IMPORTANT: explicitly forbid markdown so the summary renders as plain text in email
    plain_text_rule_zh = (
        "输出格式要求（重要）：请只输出纯文本，不要使用任何 Markdown 语法。"
        "禁止使用 **加粗**、*斜体*、# 标题、### 小标题、`代码`、> 引用、- 或 * 列表符号。"
        "四个小节请直接用『1. 』『2. 』『3. 』『4. 』开头即可，不要在小节标题前后加任何符号。"
    )
    plain_text_rule_en = (
        "Formatting rule (important): output plain text only, no Markdown. "
        "Do NOT use **bold**, *italics*, # or ### headers, `code`, > blockquotes, or - / * bullet markers. "
        "Start each of the four sections with '1. ', '2. ', '3. ', '4. ' directly, with no extra decoration."
    )
    prompts = {
        'zh': f"""请用中文总结以下学术论文，包括以下几个方面：
1. 研究背景和动机（1-2句话）
2. 主要方法和创新点（2-3句话）
3. 实验结果和结论（1-2句话）
4. 潜在应用价值（1句话）

论文标题：{paper['title']}

论文摘要：
{paper['abstract']}

请用简洁专业的语言总结，适合快速阅读理解。

{plain_text_rule_zh}""",
        'en': f"""Please summarize the following academic paper in English, including these aspects:
1. Research background and motivation (1-2 sentences)
2. Main methods and innovations (2-3 sentences)
3. Experimental results and conclusions (1-2 sentences)
4. Potential application value (1 sentence)

Paper title: {paper['title']}

Paper abstract:
{paper['abstract']}

Please use concise professional language suitable for quick reading.

{plain_text_rule_en}"""
    }
    
    # Determine which languages to generate
    langs_to_generate = ['zh', 'en'] if language == 'both' else [language]
    
    try:
        client = OpenAI(
            base_url=DEEPSEEK_BASE_URL,
            api_key=DEEPSEEK_API_KEY,
        )
        
        for lang in langs_to_generate:
            print(f"   Generating {'Chinese' if lang == 'zh' else 'English'} summary...")
            
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {
                        'role': 'user',
                        'content': prompts[lang]
                    }
                ],
                stream=True
            )
            
            # Collect streaming response
            summary = ""
            done_reasoning = False
            for chunk in response:
                # reasoning_content only exists on reasoning models (e.g. deepseek-reasoner,
                # ModelScope's DeepSeek-V3.2-Exp). Non-reasoning models like deepseek-chat
                # don't expose this attribute, so use getattr with a default to stay compatible.
                reasoning_chunk = getattr(chunk.choices[0].delta, 'reasoning_content', None) or ''
                answer_chunk = chunk.choices[0].delta.content or ''
                
                if reasoning_chunk:
                    continue  # Skip reasoning process
                elif answer_chunk:
                    if not done_reasoning:
                        done_reasoning = True
                    summary += answer_chunk
            
            summaries[lang] = strip_markdown(summary.strip())
            print(f"   ✅ {'Chinese' if lang == 'zh' else 'English'} summary completed")
        
        # Return format based on language mode
        if language == 'both':
            return summaries
        else:
            return summaries[language]
    
    except Exception as e:
        print(f"   ❌ AI summary generation failed: {str(e)}")
        error_msg = {
            'zh': "摘要生成失败，请直接查看原文。",
            'en': "Summary generation failed. Please read the original paper."
        }
        if language == 'both':
            return error_msg
        else:
            return error_msg.get(language, error_msg['en'])


def generate_date_notice(date_stats, papers, language='zh'):
    """
    Generate date reminder HTML for email
    
    Args:
        date_stats: Dictionary with date statistics
        papers: List of papers
        language: 'zh' or 'en'
        
    Returns:
        str: HTML string for date notice
    """
    total = len(papers)
    today_count = date_stats['today']
    yesterday_count = date_stats['yesterday']
    older_count = date_stats['older']
    
    # Don't show notice if all papers are today or yesterday
    if older_count == 0 and today_count > 0:
        return ""
    
    # Get text template
    txt = TEXT_TEMPLATES.get(language, TEXT_TEMPLATES['en'])
    
    # Build notice message
    notice_parts = []
    
    if today_count > 0:
        notice_parts.append(txt['published_today'].format(count=today_count))
    
    if yesterday_count > 0:
        notice_parts.append(txt['published_yesterday'].format(count=yesterday_count))
    
    if older_count > 0:
        earliest_date = min(date_stats['date_distribution'].keys())
        days_ago = (datetime.now().date() - earliest_date).days
        
        if older_count == 1:
            notice_parts.append(txt['published_older_single'].format(count=older_count, days=days_ago))
        else:
            notice_parts.append(txt['published_older_multi'].format(count=older_count))
    
    notice_text = ", ".join(notice_parts) if language == 'en' else "、".join(notice_parts)
    notice_message = txt['notice_text'].format(total=total, parts=notice_text)
    
    # Choose style based on older paper ratio
    if older_count >= total * 0.5:
        icon = "⚠️"
        bg_color = "#fff3cd"
        border_color = "#ffc107"
        text_color = "#856404"
    elif older_count > 0:
        icon = "ℹ️"
        bg_color = "#d1ecf1"
        border_color = "#17a2b8"
        text_color = "#0c5460"
    else:
        icon = "✨"
        bg_color = "#d4edda"
        border_color = "#28a745"
        text_color = "#155724"
    
    html = f"""
    <div style="background: {bg_color}; border-left: 4px solid {border_color}; padding: 15px 20px; margin-bottom: 25px; border-radius: 5px;">
        <div style="color: {text_color}; font-size: 15px; line-height: 1.6;">
            <span style="font-size: 20px; margin-right: 8px;">{icon}</span>
            <strong>{txt['date_notice']}:</strong> {notice_message}
        </div>
    </div>
    """
    
    return html


def generate_email_content(papers_with_summaries, language='zh'):
    """
    Generate HTML email content
    
    Args:
        papers_with_summaries: List of dictionaries containing papers and summaries
        language: 'zh', 'en', or 'both' for bilingual
        
    Returns:
        str: HTML formatted email content
    """
    today = datetime.now().strftime('%Y-%m-%d')
    
    papers = [item['paper'] for item in papers_with_summaries]
    date_stats = analyze_paper_dates(papers)
    
    # Get text template (use 'en' for bilingual mode header)
    txt = TEXT_TEMPLATES.get('zh' if language == 'zh' else 'en', TEXT_TEMPLATES['en'])
    
    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
            }}
            .date {{
                font-size: 14px;
                opacity: 0.9;
                margin-top: 10px;
            }}
            .paper {{
                background: white;
                padding: 25px;
                margin-bottom: 25px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                position: relative;
            }}
            .paper-title {{
                color: #667eea;
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 10px;
                line-height: 1.4;
            }}
            .quality-badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                margin-left: 8px;
                background: #ffd700;
                color: #856404;
            }}
            .pinned-badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                margin-left: 8px;
                background: #e53935;
                color: #ffffff;
            }}
            .paper-pinned {{
                border-left: 5px solid #e53935;
            }}
            .blp-badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                margin-left: 8px;
                background: #ff5722;
                color: #ffffff;
            }}
            .paper-blp {{
                border-left: 5px solid #ff5722;
                background: #fff8f5;
            }}
            .repush-badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                margin-left: 8px;
                background: #9e9e9e;
                color: #ffffff;
            }}
            .meta {{
                color: #666;
                font-size: 14px;
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 2px solid #f0f0f0;
            }}
            .meta-item {{
                margin: 5px 0;
            }}
            .date-badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                margin-left: 8px;
            }}
            .date-today {{
                background: #d4edda;
                color: #155724;
            }}
            .date-yesterday {{
                background: #d1ecf1;
                color: #0c5460;
            }}
            .date-older {{
                background: #f8d7da;
                color: #721c24;
            }}
            .categories {{
                display: inline-block;
            }}
            .category-tag {{
                background: #e8eaf6;
                color: #5c6bc0;
                padding: 3px 10px;
                border-radius: 12px;
                font-size: 12px;
                margin-right: 5px;
                display: inline-block;
            }}
            .summary {{
                background: #f8f9ff;
                padding: 18px;
                border-left: 4px solid #667eea;
                margin: 15px 0;
                border-radius: 4px;
            }}
            .summary-title {{
                font-weight: bold;
                color: #667eea;
                margin-bottom: 12px;
                font-size: 15px;
            }}
            .summary-body {{
                font-size: 16px;
                line-height: 1.75;
                color: #2c2c2c;
            }}
            .links {{
                margin-top: 15px;
            }}
            .link-button {{
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
                margin-right: 10px;
                font-size: 14px;
            }}
            .link-button:hover {{
                background: #5568d3;
            }}
            .footer {{
                text-align: center;
                color: #999;
                font-size: 12px;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📚 {txt['title']}</h1>
            <div class="date">{today}</div>
        </div>
        
        {generate_date_notice(date_stats, papers, 'zh' if language == 'zh' else 'en')}
    """
    
    now = datetime.now()
    today_date = now.date()
    yesterday_date = (now - timedelta(days=1)).date()
    
    for i, item in enumerate(papers_with_summaries, 1):
        paper = item['paper']
        summary = item['summary']
        
        # Add date badge
        paper_date = paper['published'].date()
        if paper_date == today_date:
            date_badge = f'<span class="date-badge date-today">{txt["new_today"]}</span>'
        elif paper_date == yesterday_date:
            date_badge = f'<span class="date-badge date-yesterday">{txt["yesterday_label"]}</span>'
        else:
            days_ago = (today_date - paper_date).days
            date_badge = f'<span class="date-badge date-older">{txt["days_ago_label"].format(days=days_ago)}</span>'
        
        # Add quality badge for high-quality papers
        quality_badge = ''
        if paper.get('quality_score', 0) >= 5.0:
            quality_badge = f'<span class="quality-badge">{txt["high_quality"]}</span>'

        # Badges: BLP strong-recommend > econ.EM pinned. BLP wins the left-border
        # styling because it's the strongest signal.
        pinned_badge = ''
        blp_badge = ''
        repush_badge = ''
        paper_class = 'paper'
        if paper.get('is_pinned'):
            pinned_badge = f'<span class="pinned-badge">{txt["pinned"]}</span>'
            paper_class = 'paper paper-pinned'
        if paper.get('is_blp_recommend'):
            blp_badge = f'<span class="blp-badge">{txt["blp_recommend"]}</span>'
            paper_class = 'paper paper-blp'
        if paper.get('is_repush'):
            repush_badge = f'<span class="repush-badge">{txt["repush"]}</span>'
        
        # Format category tags: translate to Chinese labels, fall back to raw code
        # for anything not in the map. Space-separated so tags don't visually merge.
        categories_html = ' '.join([
            f'<span class="category-tag">{CATEGORY_LABELS.get(cat, cat)}</span>'
            for cat in paper['categories'][:3]
        ])
        
        # Handle bilingual summaries
        if language == 'both' and isinstance(summary, dict):
            summary_html = f"""
                <div style="margin-bottom: 15px;">
                    <div style="font-weight: bold; color: #667eea; margin-bottom: 8px;">🇨🇳 中文摘要</div>
                    <div>{summary.get('zh', '').replace(chr(10), '<br>')}</div>
                </div>
                <div>
                    <div style="font-weight: bold; color: #667eea; margin-bottom: 8px;">🇬🇧 English Summary</div>
                    <div>{summary.get('en', '').replace(chr(10), '<br>')}</div>
                </div>
            """
        else:
            summary_text = summary if isinstance(summary, str) else summary.get(language, '')
            summary_html = summary_text.replace(chr(10), '<br>')
        
        html += f"""
        <div class="{paper_class}">
            <div class="paper-title">{i}. {paper['title']}{blp_badge}{pinned_badge}{repush_badge}{date_badge}{quality_badge}</div>
            <div class="meta">
                <div class="meta-item">
                    <strong>👥 {txt['authors']}:</strong> {paper['authors'][:200]}{'...' if len(paper['authors']) > 200 else ''}
                </div>
                <div class="meta-item">
                    <strong>📅 {txt['published']}:</strong> {paper['published'].strftime('%Y-%m-%d %H:%M')}
                </div>
                <div class="meta-item">
                    <strong>🏷️ {txt['categories']}:</strong>
                    <div class="categories">{categories_html}</div>
                </div>
                <div class="meta-item">
                    <strong>📊 {txt['quality_score']}:</strong> {paper.get('quality_score', 0):.1f}
                </div>
            </div>
            
            <div class="summary">
                <div class="summary-title">🤖 {txt['ai_summary']}</div>
                <div class="summary-body">{summary_html}</div>
            </div>
            
            <div class="links">
                <a href="{paper['pdf_url']}" class="link-button">📄 {txt['view_pdf']}</a>
            </div>
        </div>
        """
    
    html += f"""
        <div class="footer">
            <p>{txt['footer_auto']}</p>
            <p>{txt['footer_powered']}</p>
        </div>
    </body>
    </html>
    """
    
    return html


def generate_pushplus_content(papers_with_summaries, language='zh'):
    """HTML digest for PushPlus (WeChat webview).

    Uses inline-styled divs only (no <html>/<head>/<style>) so PushPlus's
    HTML sanitizer doesn't reject it — the original full email HTML got a
    code=999 服务端验证错误 because of its <style> block, not raw length.
    Summaries are rendered in full; newlines are preserved as <br>.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    parts = [
        '<div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;line-height:1.6;font-size:16px;">',
        f'<h3 style="color:#667eea;margin:0 0 10px;font-size:20px;">📚 arXiv 每日推送 · {today}</h3>',
        f'<p style="color:#888;font-size:14px;margin:0 0 15px;">共 {len(papers_with_summaries)} 篇 · 点标题跳 arXiv</p>',
    ]

    for i, item in enumerate(papers_with_summaries, 1):
        paper = item['paper']
        summary = item['summary']

        # Normalize summary and preserve paragraph breaks as <br> so the
        # WeChat webview keeps the structure DeepSeek generated.
        if isinstance(summary, dict):
            summary_text = summary.get('zh') or summary.get('en') or ''
        else:
            summary_text = summary or ''
        summary_html = summary_text.strip().replace('\n', '<br>')

        badges = []
        if paper.get('is_blp_recommend'):
            badges.append('<span style="color:#e91e63;font-size:13px;">⭐BLP</span>')
        if paper.get('is_pinned'):
            badges.append('<span style="color:#ff9800;font-size:13px;">📌</span>')
        if paper.get('is_repush'):
            badges.append('<span style="color:#4caf50;font-size:13px;">♻️</span>')
        score = paper.get('quality_score', 0)
        badges.append(f'<span style="color:#888;font-size:13px;">[{score:.1f}]</span>')
        badge_html = ' '.join(badges)

        # entry_id is the arxiv abs page (e.g. https://arxiv.org/abs/2401.12345v1).
        # abs page is nicer than pdf for a quick tap-through — has title/abstract too.
        link = paper.get('entry_id') or paper.get('pdf_url', '')

        # Category chips: up to 3 categories, primary first. Translate to Chinese
        # via CATEGORY_LABELS, fall back to raw arxiv code (e.g. "q-fin.MF") for
        # anything not mapped.
        cats = list(paper.get('categories') or [])[:3]
        cat_html = ' '.join(
            f'<span style="display:inline-block;background:#eef1ff;color:#4a5aa8;'
            f'font-size:13px;padding:1px 6px;border-radius:3px;margin-right:4px;">'
            f'{CATEGORY_LABELS.get(c, c)}</span>'
            for c in cats
        )

        parts.append(
            f'<div style="margin:0 0 16px;padding:12px;background:#f7f8fa;border-left:3px solid #667eea;border-radius:4px;">'
            f'<div style="font-weight:600;font-size:17px;margin-bottom:6px;">'
            f'{i}. <a href="{link}" style="color:#333;text-decoration:none;">{paper["title"]}</a> {badge_html}'
            f'</div>'
            f'<div style="margin:4px 0 8px;">{cat_html}</div>'
            f'<div style="color:#555;font-size:15px;">{summary_html}</div>'
            f'</div>'
        )

    parts.append('</div>')
    return ''.join(parts)


def send_pushplus(subject, html_content):
    """Push the digest to WeChat via PushPlus (https://www.pushplus.plus).

    Returns True on success, False on failure. PushPlus renders the HTML in
    a WeChat webview when the user taps the notification.
    """
    print(f"\n📱 Pushing to WeChat via PushPlus ({len(html_content)} chars)...")

    payload = json.dumps({
        'token': PUSHPLUS_TOKEN,
        'title': subject,
        'content': html_content,
        'template': 'html',
    }).encode('utf-8')

    max_attempts = 3
    backoff = 5

    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(
                PUSHPLUS_URL,
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode('utf-8', errors='replace')
            data = json.loads(body)
            if data.get('code') == 200:
                print(f"✅ Pushed successfully (msg_id={data.get('data')})")
                return True
            # code != 200 is an application-level failure (bad token, quota,
            # etc.) and won't self-heal on retry.
            print(f"❌ PushPlus rejected the request: code={data.get('code')} msg={data.get('msg')}")
            return False
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"⚠️ Push attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                time.sleep(backoff)
                backoff *= 2

    print("❌ PushPlus push failed after all retries.")
    return False


def main():
    """Main execution function"""
    print("=" * 60)
    print("🚀 arXiv Daily Paper Digest - Starting")
    print("=" * 60)
    
    # Check required environment variables
    required_vars = ['DEEPSEEK_API_KEY', 'PUSHPLUS_TOKEN']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these environment variables")
        sys.exit(1)
    
    try:
        # Step 1: Fetch latest papers with quality filtering
        papers = get_latest_papers()

        # Step 1.5: Load push history and inject any papers that are due for a
        # scheduled repush (D+7 / D+30). Those papers usually aren't in the
        # natural recent-window fetch anymore, so we look them up by arXiv id.
        print("\n" + "=" * 60)
        print("📜 Applying push-history filter (memory-curve schedule)")
        print("=" * 60)
        history = push_history.load()
        history = push_history.prune(history)

        have = {push_history.paper_key(p.get('entry_id')) for p in papers}
        due_ids = [pid for pid in push_history.due_repush_ids(history) if pid not in have]
        if due_ids:
            print(f"  ♻️ {len(due_ids)} paper(s) due for scheduled repush, fetching by id...")
            repush_papers = fetch_papers_by_ids(due_ids)
            papers.extend(repush_papers)
            # Re-rank so newly injected papers get the same priority treatment
            for p in repush_papers:
                arxiv_primary = (list(p.get('categories', [])) or [None])[0]
                p['is_pinned'] = arxiv_primary == PRIMARY_CATEGORY
                p['is_blp_recommend'] = matches_blp(p)
            papers.sort(key=lambda x: (
                _priority_tier(x),
                -x.get('quality_score', 0),
                -x['published'].timestamp()
            ))

        # Step 1.6: Apply the schedule filter (drops papers that already pushed
        # today's slot or are retired after TOTAL_PUSHES_CAP pushes).
        papers, skipped = push_history.filter_papers(papers, history)
        print(f"  Kept {len(papers)} paper(s), skipped {len(skipped)} per schedule")

        if not papers:
            print("\n❌ No papers to push. Either arXiv fetch returned an empty pool "
                  "or every candidate is in the memory-curve cooldown. Failing the job "
                  "so the miss is visible instead of silently succeeding.")
            push_history.save(history)
            sys.exit(1)

        # Step 1.7: Cap the surviving pool to MAX_RESULTS. The pool was built
        # oversized (CANDIDATE_POOL_SIZE) precisely so that when the recent
        # top-scored papers are all in the memory-curve cooldown, backfill from
        # unpushed older-but-still-good papers happens automatically here.
        if len(papers) > MAX_RESULTS:
            trimmed = len(papers) - MAX_RESULTS
            papers = papers[:MAX_RESULTS]
            print(f"  Capped to top {MAX_RESULTS} (trimmed {trimmed} lower-ranked candidates)")

        # Step 2: Analyze paper dates and output statistics
        date_stats = analyze_paper_dates(papers)
        print(f"\n📊 Paper Date Statistics:")
        print(f"   Today: {date_stats['today']} papers")
        print(f"   Yesterday: {date_stats['yesterday']} papers")
        print(f"   Older: {date_stats['older']} papers")
        
        # Step 3: Generate AI summaries for each paper
        print("\n" + "=" * 60)
        print("🤖 Generating AI Summaries")
        print("=" * 60)
        
        papers_with_summaries = []
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}]")
            summary = summarize_paper(paper, EMAIL_LANGUAGE)
            papers_with_summaries.append({
                'paper': paper,
                'summary': summary
            })
        
        # Step 4: Generate compact HTML digest for PushPlus (must fit under
        # the account tier's content limit; free tier caps out around ~5KB).
        print("\n" + "=" * 60)
        print("📧 Generating Digest Content")
        print("=" * 60)
        html_content = generate_pushplus_content(papers_with_summaries, EMAIL_LANGUAGE)

        # Step 5: Push via PushPlus
        today = datetime.now().strftime('%Y-%m-%d')
        subject = f"📚 arXiv {today} · {len(papers_with_summaries)}篇"
        sent_ok = send_pushplus(subject, html_content)

        # Step 6: On successful send, record today's push so the schedule can
        # decide when (and whether) each paper should be pushed again.
        if not sent_ok:
            print("\n❌ send_pushplus() returned False. "
                  "Not updating push history. Failing the job so the miss is visible.")
            sys.exit(1)

        history = push_history.update(history, papers)
        push_history.save(history)

        print("\n" + "=" * 60)
        print("✅ Execution completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n❌ Execution error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

import os
import smtplib
import arxiv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from openai import OpenAI
from collections import Counter, defaultdict
import re
from difflib import SequenceMatcher

# ========== Configuration ==========

# arXiv search configuration
CATEGORIES = ['econ.EM', 'math.OC', 'stat.ML']  # Econometrics + non-convex opt + statistical ML (for BLP research)
PRIMARY_CATEGORY = 'econ.EM'  # Dominant field; other categories only fill leftover slots
MAX_RESULTS = 10  # Daily cap
MIN_PAPERS_PER_CATEGORY = 1  # (kept for backward compat; not used by the primary-first selector below)

# Language configuration
# Supported values: 'zh' (Chinese), 'en' (English), 'both' (Bilingual)
EMAIL_LANGUAGE = os.environ.get('EMAIL_LANGUAGE', 'zh')  # Default to Chinese

# DeepSeek API configuration (official DeepSeek platform)
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = 'https://api.deepseek.com/v1'
DEEPSEEK_MODEL = 'deepseek-chat'  # points to the latest DeepSeek-V3

# Email configuration
SENDER_EMAIL = os.environ.get('SENDER_EMAIL')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD')
RECEIVER_EMAIL = os.environ.get('RECEIVER_EMAIL')
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))

# Quality filtering thresholds
MIN_ABSTRACT_LENGTH = 100  # Minimum abstract length (characters)
SIMILARITY_THRESHOLD = 0.85  # Title similarity threshold for duplicate detection

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
        # 'published_older_single': '<strong>{count} 篇</strong>是 {days} 天前发布（可能已读过）',
        'published_older_multi': '<strong>{count} 篇</strong>是 2 天及更早前发布（可能已读过）',
        'notice_text': '本次推送的 {total} 篇论文中，{parts}。',
        'new_today': '今日新发布',
        'yesterday_label': '昨日发布',
        'days_ago_label': '{days} 天前',
        'high_quality': '⭐ 高质量',
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
        # 'published_older_single': '<strong>{count} paper</strong> published {days} days ago (may have been read)',
        'published_older_multi': '<strong>{count} papers</strong> published 2+ days ago (may have been read)',
        'notice_text': 'Of the {total} papers in this digest, {parts}.',
        'new_today': 'NEW TODAY',
        'yesterday_label': 'YESTERDAY',
        'days_ago_label': '{days} DAYS AGO',
        'high_quality': '⭐ HIGH QUALITY',
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
    
    client = arxiv.Client()
    papers_by_category = defaultdict(list)
    seen_ids = set()
    
    # Step 1: Fetch papers from each category separately
    for category in CATEGORIES:
        print(f"\n🔎 Searching category: {category}")
        
        try:
            # Fetch a wider window so that if today has few/zero new papers in the
            # primary category, we can fall back to high-quality papers from
            # earlier days (~10 recent days for econ.EM at ~3 papers/day).
            search = arxiv.Search(
                query=f'cat:{category}',
                max_results=MAX_RESULTS * 3,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            results = list(client.results(search))
            print(f"  API returned {len(results)} papers")
            
            for result in results:
                if result.entry_id not in seen_ids:
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
    
    # Step 2: Primary category first — econ.EM dominates the digest.
    # Take up to MAX_RESULTS from PRIMARY_CATEGORY, ordered by quality within the
    # recent-fetch window. If today has zero new papers, this naturally rolls back
    # to the best of the past few days (SortCriterion.SubmittedDate DESC + wider window).
    print(f"\n⚖️ Selecting papers (primary: {PRIMARY_CATEGORY})...")
    selected_papers = []
    primary_papers = papers_by_category.get(PRIMARY_CATEGORY, [])
    take_from_primary = min(len(primary_papers), MAX_RESULTS)
    selected_papers.extend(primary_papers[:take_from_primary])
    print(f"  Selected {take_from_primary} papers from {PRIMARY_CATEGORY} (primary)")

    # Step 3: Fill any remaining slots with top-quality papers from the other
    # categories (math.OC, stat.ML, ...). If econ.EM already fills all MAX_RESULTS
    # slots, no other-category papers are shown.
    remaining_slots = MAX_RESULTS - len(selected_papers)

    if remaining_slots > 0:
        print(f"\n📊 Filling {remaining_slots} remaining slots from other categories...")
        others = []
        seen_entry_ids = {p['entry_id'] for p in selected_papers}
        for category in CATEGORIES:
            if category == PRIMARY_CATEGORY:
                continue
            for paper in papers_by_category.get(category, []):
                if paper['entry_id'] not in seen_entry_ids:
                    others.append(paper)
        others.sort(key=lambda x: x['quality_score'], reverse=True)
        selected_papers.extend(others[:remaining_slots])
    
    # Step 4: Remove duplicates using intelligent similarity detection
    print(f"\n🔍 Checking for duplicate/similar papers...")
    selected_papers = remove_duplicate_papers(selected_papers)
    
    # Step 5: Final sort by publish date (newest first)
    selected_papers.sort(key=lambda x: x['published'], reverse=True)
    
    print(f"\n✅ Total papers collected: {len(selected_papers)}")
    print(f"📄 Papers to send: {len(selected_papers)}")
    
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
    return text


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
        
        # Format category tags (join with a space so tags don't run together visually)
        categories_html = ' '.join([
            f'<span class="category-tag">{cat}</span>'
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
        <div class="paper">
            <div class="paper-title">{i}. {paper['title']}{date_badge}{quality_badge}</div>
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


def send_email(subject, html_content):
    """
    Send email via SMTP
    
    Args:
        subject: Email subject line
        html_content: HTML formatted email content
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n📧 Sending email to {RECEIVER_EMAIL}...")
    
    try:
        message = MIMEMultipart('alternative')
        message['Subject'] = subject
        message['From'] = SENDER_EMAIL
        message['To'] = RECEIVER_EMAIL
        
        html_part = MIMEText(html_content, 'html', 'utf-8')
        message.attach(html_part)
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
        
        print(f"✅ Email sent successfully!")
        return True
    
    except Exception as e:
        print(f"❌ Email sending failed: {str(e)}")
        return False


def main():
    """Main execution function"""
    print("=" * 60)
    print("🚀 arXiv Daily Paper Digest - Starting")
    print("=" * 60)
    
    # Check required environment variables
    required_vars = ['DEEPSEEK_API_KEY', 'SENDER_EMAIL', 'SENDER_PASSWORD', 'RECEIVER_EMAIL']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these environment variables")
        return
    
    try:
        # Step 1: Fetch latest papers with quality filtering
        papers = get_latest_papers()
        
        if not papers:
            print("\n⚠️ No papers found, exiting")
            return
        
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
        
        # Step 4: Generate email content
        print("\n" + "=" * 60)
        print("📧 Generating Email Content")
        print("=" * 60)
        html_content = generate_email_content(papers_with_summaries, EMAIL_LANGUAGE)
        
        # Step 5: Send email
        today = datetime.now().strftime('%Y-%m-%d')
        subject = f"📚 arXiv Daily Paper Digest - {today}"
        send_email(subject, html_content)
        
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

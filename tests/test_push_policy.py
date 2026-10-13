"""离线验证推送次数、旧记录迁移与中文提示词，不发送任何推送。"""
import ast
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import econ_backlog
import push_history


class PushPolicyTests(unittest.TestCase):
    def setUp(self):
        self.start = date(2026, 9, 1)
        self.paper = {
            'entry_id': 'https://arxiv.org/abs/2609.00001v2',
            'title': 'Example',
            'categories': ['econ.EM'],
        }

    def test_category_limits(self):
        for category in ('econ.EM', 'math.ST', 'stat.ME', 'stat.ML', 'stat.AP', 'stat.CO', 'stat.TH'):
            with self.subTest(category=category):
                self.assertEqual(push_history.push_cap({'categories': ['cs.LG', category]}), 3)
        for category in ('math.OC', 'cs.LG', 'econ.TH', 'econ.GN', 'math.PR', 'q-fin.ST'):
            with self.subTest(category=category):
                self.assertEqual(push_history.push_cap({'categories': [category]}), 1)
        self.assertEqual(push_history.push_cap({'primary_category': 'econ.EM'}), 3)
        self.assertEqual(push_history.push_cap({}), 1)

    def test_core_schedule_and_retirement(self):
        history = {}
        for offset, eligible in ((0, True), (0, False), (6, False), (7, True), (29, False), (30, True), (60, False)):
            today = self.start + timedelta(days=offset)
            kept, _ = push_history.filter_papers([self.paper], history, today)
            self.assertEqual(bool(kept), eligible)
            if kept:
                push_history.update(history, kept, today)
        self.assertEqual(len(history['2609.00001']['push_dates']), 3)
        self.assertEqual(push_history.due_repush_ids(history, self.start + timedelta(days=90)), [])

    def test_other_paper_only_once(self):
        self.paper['categories'] = ['math.OC']
        history = push_history.update({}, [self.paper], self.start)
        today = self.start + timedelta(days=40)
        self.assertEqual(push_history.due_repush_ids(history, today), [])
        self.assertEqual(push_history.filter_papers([self.paper], history, today)[0], [])

    def test_legacy_history_is_classified_before_repush(self):
        for category, allowed in (('stat.ME', True), ('math.OC', False)):
            with self.subTest(category=category):
                history = {'2609.00001': {'push_dates': [self.start.isoformat()]}}
                today = self.start + timedelta(days=40)
                self.assertEqual(push_history.due_repush_ids(history, today), ['2609.00001'])
                self.paper['categories'] = [category]
                kept, _ = push_history.filter_papers([self.paper], history, today)
                self.assertEqual(bool(kept), allowed)
                self.assertEqual(history['2609.00001']['categories'], [category])
                self.assertEqual(history['2609.00001']['push_dates'], [self.start.isoformat()])
                if not allowed:
                    self.assertEqual(push_history.due_repush_ids(history, today), [])

    def test_backlog_uses_same_cap(self):
        self.paper['categories'] = ['math.OC']
        self.paper['primary_category'] = 'math.OC'
        history = push_history.update({}, [self.paper], self.start)
        backlog = econ_backlog.upsert({}, [self.paper], self.start)
        self.assertEqual(econ_backlog.pick_fillers(backlog, history, 3, today=self.start + timedelta(days=40)), [])

    def test_global_cap_cannot_allow_four_pushes(self):
        with patch.object(push_history, 'TOTAL_PUSHES_CAP', 5):
            self.assertEqual(push_history.push_cap(self.paper), 3)


class ChinesePromptTests(unittest.TestCase):
    def test_both_summary_styles_request_chinese_prose(self):
        # 仅加载目标函数，并用假的模型响应检查实际发出的提示词；不需要联网或密钥。
        source = Path(__file__).resolve().parents[1] / 'fetch_papers.py'
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'summarize_paper')
        for style in ('plain', 'academic'):
            with self.subTest(style=style):
                client = Mock()
                client.chat.completions.create.return_value = [SimpleNamespace(
                    usage=None,
                    choices=[SimpleNamespace(delta=SimpleNamespace(content='作者提出了新的工具变量估计方法。'))],
                )]
                namespace = {
                    'SUMMARY_STYLE': style, 'OpenAI': Mock(return_value=client),
                    'DEEPSEEK_BASE_URL': 'unused', 'DEEPSEEK_API_KEY': 'unused',
                    'DEEPSEEK_MODEL': 'unused', 'strip_markdown': lambda text: text,
                }
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
                result = namespace['summarize_paper']({'title': 'Test', 'abstract': 'Test abstract'}, 'zh')
                self.assertEqual(result, '作者提出了新的工具变量估计方法。')
                prompt = client.chat.completions.create.call_args.kwargs['messages'][0]['content']
                self.assertIn('正文全部用自然、完整的中文叙述', prompt)
                self.assertIn('不写英文句子', prompt)
                self.assertIn('自助法', prompt)
                self.assertNotIn('这类术语直接用', prompt)


if __name__ == '__main__':
    unittest.main()

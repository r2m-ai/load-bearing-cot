"""Release checks: migrated paths, CLI artifacts and regression boundaries."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluate
from load_bearing.datasets import load_examples
from load_bearing.protocol import correct, extract_answer, interventions, perturb

ROOT = Path(__file__).resolve().parents[1]


class RegressionTests(unittest.TestCase):
    def test_bbh_bracket_sequences_remain_distinct(self):
        self.assertEqual(extract_answer('Answer: ))', 'bbh'), '))')
        self.assertFalse(correct('))', ')', 'bbh'))
        self.assertFalse(correct(']', '])', 'bbh'))

    def test_control_is_exact_baseline_prefix(self):
        trace = '\n  1. Take 4 items.\r\n\r\n  2. Add 6 items.\r\n  3. Sum is 10.\r\n  4. Done.\r\nAnswer: 10'
        for edit in interventions(trace, 'arithmetic'):
            self.assertTrue(trace.startswith(edit['control_prefix']))
            self.assertTrue(edit['prefix'].startswith('\n  1.'))
            self.assertIn('\r\n', edit['prefix'])

    def test_empty_answer_does_not_eat_next_line(self):
        self.assertIsNone(extract_answer('Answer:\n42', 'gsm8k'))
        self.assertIsNone(extract_answer('Answer: 42\nAnswer: ', 'gsm8k'))

    def test_bold_answer_is_excluded_from_steps(self):
        self.assertEqual(extract_answer('**Answer:** 24', 'gsm8k'), '24')
        trace = 'First 2.\nSecond 3.\n**Answer:** 24\nMore text 4.\nMore text 5.'
        self.assertEqual(interventions(trace, 'arithmetic'), [])

    def test_thousands_separator_is_one_value(self):
        self.assertEqual(perturb('2. There are 1,200 items.', 'arithmetic'), '2. There are 2403 items.')

    def test_invalid_input_is_rejected_before_sampling(self):
        cases = [('mmlu', dict(id='q', question='Q', reference_answer='AB', choices=['a','b','c','d'])),
                 ('gsm8k', dict(id='q', question='Q', reference_answer='NaN')),
                 ('gsm8k', dict(id=[], question='Q', reference_answer='1'))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.jsonl'
            for dataset, row in cases:
                path.write_text(json.dumps(row) + '\n')
                with self.subTest(dataset=dataset, row=row), self.assertRaises(ValueError):
                    load_examples(dataset, 1, 42, input_file=path)


class LayoutTests(unittest.TestCase):
    def test_research_directories_are_present(self):
        for topic in ('pipeline', 'behavior', 'controls', 'validation', 'models', 'probes', 'steering', 'figures'):
            self.assertTrue((ROOT / 'experiments' / topic).is_dir())
        self.assertFalse((ROOT / 'revision').exists())
        self.assertFalse((ROOT / 'scripts/legacy').exists())

    def test_dynamic_import_targets_exist(self):
        for path in (ROOT / 'experiments').rglob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('_load_module', '_load'):
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        with self.subTest(script=str(path), module=node.args[0].value):
                            self.assertTrue((ROOT / 'experiments' / node.args[0].value).is_file())

    def test_root_discovery_functions_after_moves(self):
        for path in (ROOT / 'experiments').rglob('*.py'):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == '_find_repo_root':
                    # Run only the pure discovery function, never model imports
                    # or experiment-level code that writes outputs at import time.
                    namespace = {'Path': Path}
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
                    with self.subTest(script=str(path)):
                        self.assertEqual(namespace['_find_repo_root'](path), ROOT)


class CLITests(unittest.TestCase):
    def invoke(self, directory, backend):
        source = Path(directory) / 'input.jsonl'
        source.write_text(json.dumps(dict(id='q', question='Q', reference_answer='24')) + '\n')
        output = Path(directory) / 'run'
        args = ['evaluate.py', '--model', 'fake', '--dataset', 'gsm8k', '--input-file', str(source), '--output-dir', str(output)]
        with patch('sys.argv', args), patch.object(evaluate, 'VLLMBackend', return_value=backend), \
             patch.object(evaluate.importlib.metadata, 'version', return_value='test'), \
             contextlib.redirect_stdout(io.StringIO()):
            evaluate.main()
        return output

    def test_empty_interventions_are_still_an_artifact(self):
        backend = SimpleNamespace(tokenizer=SimpleNamespace(chat_template='test'), generate=lambda requests: ['Answer: 99'])
        with tempfile.TemporaryDirectory() as directory:
            output = self.invoke(directory, backend)
            self.assertEqual((output / 'interventions.jsonl').read_text(), '')
            self.assertEqual(json.loads((output / 'summary.json').read_text())['status'], 'insufficient_data')
            self.assertIn('finished_at', json.loads((output / 'manifest.json').read_text()))

    def test_interrupt_is_recorded(self):
        def interrupt(requests):
            raise KeyboardInterrupt
        backend = SimpleNamespace(tokenizer=SimpleNamespace(chat_template='test'), generate=interrupt)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyboardInterrupt):
                self.invoke(directory, backend)
            manifest = json.loads((Path(directory) / 'run/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'interrupted')

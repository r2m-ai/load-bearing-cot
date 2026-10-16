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
from load_bearing.datasets import DATASETS, adapter, load_examples
from load_bearing.protocol import STRATEGIES, correct, extract_answer, interventions, perturb

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


class RegistryTests(unittest.TestCase):
    def test_every_dataset_row_is_complete(self):
        for name, dataset in DATASETS.items():
            with self.subTest(dataset=name):
                for field in ('source', 'split', 'answer_type', 'default_strategy'):
                    self.assertTrue(getattr(dataset, field), field)
                for field in ('convert', 'validate', 'question', 'extract', 'correct'):
                    self.assertTrue(callable(getattr(dataset, field)), field)
                self.assertIn(dataset.default_strategy, STRATEGIES)
                self.assertTrue(callable(dataset.decoy))
                self.assertTrue(dataset.configs is None or len(dataset.configs) > 0)

    def test_unknown_dataset_raises(self):
        with self.assertRaisesRegex(ValueError, 'Unknown dataset'):
            adapter('tsp')
        with self.assertRaisesRegex(ValueError, 'Unknown dataset'):
            extract_answer('Answer: 1', 'tsp')
        with self.assertRaisesRegex(ValueError, 'Unknown dataset'):
            load_examples('tsp', 1, 42, input_file='unused.jsonl')

    def test_cli_choices_come_from_the_registry(self):
        parser_source = (ROOT / 'evaluate.py').read_text()
        self.assertIn('choices=list(DATASETS)', parser_source)
        self.assertNotIn('"arithmetic" if args.dataset == "gsm8k"', parser_source)


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
    def invoke(self, directory, backend, extra=()):
        source = Path(directory) / 'input.jsonl'
        source.write_text(json.dumps(dict(id='q', question='Q', reference_answer='24')) + '\n')
        output = Path(directory) / 'run'
        args = ['evaluate.py', '--model', 'fake', '--dataset', 'gsm8k', '--input-file', str(source), '--output-dir', str(output), *extra]
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
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertIn('finished_at', manifest)
            self.assertIsNone(manifest['environment'])

    def test_environment_lands_in_the_manifest(self):
        environment = {'gpu': [{'name': 'H100', 'compute_capability': '9.0'}], 'device_count': 1,
                       'cuda': '12.4', 'driver': None, 'torch': '2.t', 'dtype': 'torch.bfloat16', 'max_model_len': 8192}
        backend = SimpleNamespace(tokenizer=SimpleNamespace(chat_template='test'),
                                  generate=lambda requests: ['Answer: 99'], environment=lambda: environment)
        with tempfile.TemporaryDirectory() as directory:
            output = self.invoke(directory, backend)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['environment'], environment)
            self.assertIn('tensor_parallel_size', manifest['arguments'])
            self.assertIn('gpu_memory_utilization', manifest['arguments'])

    def test_strategy_alias_is_recorded_under_its_canonical_name(self):
        backend = SimpleNamespace(tokenizer=SimpleNamespace(chat_template='test'),
                                  generate=lambda requests: ['Answer: 24'] * len(requests) if requests[0][1] else
                                  ['1. Four rows.\n2. Six items each.\n3. Four times six is 24.\n4. Done.\nAnswer: 24'])
        with tempfile.TemporaryDirectory() as directory:
            output = self.invoke(directory, backend, ['--strategy', 'contradiction'])
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['arguments']['strategy'], 'premise_contradiction')
            rows = [json.loads(line) for line in (output / 'interventions.jsonl').read_text().splitlines()]
            self.assertTrue(rows)
            self.assertTrue(all(row['strategy'] == 'premise_contradiction' for row in rows))

    def test_reasoning_tags_are_recorded_and_validated(self):
        backend = SimpleNamespace(tokenizer=SimpleNamespace(chat_template='test'), generate=lambda requests: ['Answer: 99'])
        with tempfile.TemporaryDirectory() as directory:
            output = self.invoke(directory, backend)
            self.assertEqual(json.loads((output / 'manifest.json').read_text())['reasoning_tags'], ['<think>', '</think>'])
        with tempfile.TemporaryDirectory() as directory:
            output = self.invoke(directory, backend, ['--reasoning-tags', '<r>,</r>'])
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['reasoning_tags'], ['<r>', '</r>'])
            self.assertEqual(manifest['arguments']['reasoning_tags'], '<r>,</r>')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                self.invoke(directory, backend, ['--reasoning-tags', '<r>'])
            self.assertFalse((Path(directory) / 'run').exists())

    def test_existing_output_dir_is_a_usage_error(self):
        backend = SimpleNamespace(tokenizer=SimpleNamespace(chat_template='test'), generate=lambda requests: ['Answer: 99'])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            output.mkdir()
            (output / 'keep.txt').write_text('x')
            stderr = io.StringIO()
            with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(stderr):
                self.invoke(directory, backend)
            self.assertEqual(caught.exception.code, 2)
            self.assertIn('--output-dir already exists', stderr.getvalue())
            self.assertEqual(sorted(p.name for p in output.iterdir()), ['keep.txt'])

    def resumable_run(self, directory):
        """Interrupt a two-question run after the second baseline is on disk."""
        source = Path(directory) / 'input.jsonl'
        source.write_text(''.join(json.dumps(dict(id=f'q{i}', question=f'Q{i}', reference_answer='24')) + '\n' for i in (1, 2)))
        output = Path(directory) / 'run'
        trace = '1. There are 4 rows.\n2. Each row has 6 items.\n3. 4 times 6 is 24.\n4. The total is 24.\n5. So 24 items.\nAnswer: 24'
        base = ['evaluate.py', '--model', 'fake', '--dataset', 'gsm8k', '--input-file', str(source), '--seed', '3']

        class Backend:
            def __init__(self, interrupt_second):
                self.baseline_prompts, self.continuation_prefixes, self.interrupt_second = [], [], interrupt_second
                self.tokenizer = SimpleNamespace(chat_template='test')
            def generate(self, requests):
                if requests[0][1] == '':
                    self.baseline_prompts.append(requests[0][0])
                    return [trace]
                if self.interrupt_second and len(self.baseline_prompts) == 2:
                    raise KeyboardInterrupt
                self.continuation_prefixes.extend(prefix for _, prefix in requests)
                return ['Answer: 99' if i % 2 == 0 else 'Answer: 24' for i in range(len(requests))]

        def invoke(argv, backend):
            with patch('sys.argv', argv), patch.object(evaluate, 'VLLMBackend', return_value=backend), \
                 patch.object(evaluate.importlib.metadata, 'version', return_value='test'), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                evaluate.main()
        first = Backend(interrupt_second=True)
        with self.assertRaises(KeyboardInterrupt):
            invoke(base + ['--output-dir', str(output)], first)
        self.assertEqual(json.loads((output / 'manifest.json').read_text())['status'], 'interrupted')
        self.assertEqual(len((output / 'baselines.jsonl').read_text().splitlines()), 2)
        self.assertEqual(len((output / 'interventions.jsonl').read_text().splitlines()), 3)
        return base, output, Backend, invoke

    def test_resume_regenerates_only_the_missing_continuations(self):
        with tempfile.TemporaryDirectory() as directory:
            base, output, Backend, invoke = self.resumable_run(directory)
            second = Backend(interrupt_second=False)
            invoke(base + ['--resume', str(output)], second)
            self.assertEqual(second.baseline_prompts, [])
            self.assertEqual(len(second.continuation_prefixes), 6)
            rows = [json.loads(line) for line in (output / 'interventions.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual(sorted({row['id'] for row in rows}), ['q1', 'q2'])
            self.assertEqual(len((output / 'baselines.jsonl').read_text().splitlines()), 2)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['n_questions'], 2)
            self.assertEqual(summary['n_interventions'], 6)
            self.assertEqual(summary['n_eligible_questions'], 2)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'ok')
            self.assertEqual(len(manifest['resumed_from']), 1)
            self.assertNotEqual(manifest['resumed_from'][0], manifest['started_at'])
            self.assertNotIn('error', manifest)
            # A finished run resumes to the same artifacts without any generation.
            third = Backend(interrupt_second=False)
            invoke(base + ['--resume', str(output)], third)
            self.assertEqual((third.baseline_prompts, third.continuation_prefixes), ([], []))
            self.assertEqual(len((output / 'interventions.jsonl').read_text().splitlines()), 6)
            self.assertEqual(len(json.loads((output / 'manifest.json').read_text())['resumed_from']), 2)

    def test_resume_refuses_mismatched_settings_and_partial_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            base, output, Backend, invoke = self.resumable_run(directory)
            for extra in (['--strategy', 'contradiction'], ['--seed', '4'], ['--max-new-tokens', '5'],
                          ['--reasoning-tags', '<r>,</r>'], ['--output-dir', str(Path(directory) / 'other')]):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    invoke(base + ['--resume', str(output)] + extra, Backend(interrupt_second=False))
            self.assertEqual(len((output / 'interventions.jsonl').read_text().splitlines()), 3)
            with (output / 'interventions.jsonl').open('a') as handle:
                row = json.loads((output / 'interventions.jsonl').read_text().splitlines()[0])
                handle.write(json.dumps(dict(row, id='q2')) + '\n')
            with self.assertRaises(SystemExit):
                invoke(base + ['--resume', str(output)], Backend(interrupt_second=False))
            with self.assertRaises(SystemExit):
                invoke(base + ['--resume', str(Path(directory) / 'missing')], Backend(interrupt_second=False))

    def test_interrupt_is_recorded(self):
        def interrupt(requests):
            raise KeyboardInterrupt
        backend = SimpleNamespace(tokenizer=SimpleNamespace(chat_template='test'), generate=interrupt)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyboardInterrupt):
                self.invoke(directory, backend)
            manifest = json.loads((Path(directory) / 'run/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'interrupted')

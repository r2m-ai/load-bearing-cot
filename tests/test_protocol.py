import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from evaluate import run
from load_bearing.datasets import load_examples
from load_bearing.protocol import correct, extract_answer, interventions, perturb, reasoning_steps, summarize

TRACE = '1. There are 4 rows.\n2. Each row has 6 items.\n3. Multiply 4 by 6 to obtain 24.\n4. The total is 24.\nAnswer: 24'


class ProtocolTests(unittest.TestCase):
    def test_last_explicit_answer(self):
        self.assertEqual(extract_answer('Answer: 4\nActually...\nAnswer: -1,200.50', 'gsm8k'), '-1200.50')

    def test_no_number_fallback(self):
        self.assertIsNone(extract_answer('There are 42 items.', 'gsm8k'))
        self.assertIsNone(extract_answer('Answer: 42 or 43', 'gsm8k'))

    def test_numeric_equivalence(self):
        self.assertTrue(correct('24.0', '24', 'gsm8k'))
        self.assertIsNone(correct(None, '24', 'gsm8k'))

    def test_choice_validation(self):
        self.assertEqual(extract_answer('Answer: (b).', 'mmlu'), 'B')
        self.assertIsNone(extract_answer('Answer: Because this...', 'mmlu'))

    def test_bbh_not_prefix_matching(self):
        self.assertFalse(correct(extract_answer('Answer: no', 'bbh'), 'not valid', 'bbh'))
        self.assertTrue(correct(extract_answer('Answer: (A).', 'bbh'), '(A)', 'bbh'))

    def test_step_number_is_not_perturbed(self):
        self.assertEqual(perturb('2. There are 6 items.', 'arithmetic'), '2. There are 15 items.')
        self.assertIsNone(perturb('2. Nothing numerical.', 'arithmetic'))

    def test_no_op_numeric_edit_avoided(self):
        self.assertEqual(perturb('Value is -3.', 'arithmetic'), 'Value is -2.')
        self.assertEqual(perturb('Value is 6.', 'arithmetic'), 'Value is 15.')
        self.assertEqual(perturb('Value is 1.5.', 'arithmetic'), 'Value is 6.0.')

    def test_truncation_and_control(self):
        edits = interventions(TRACE, 'arithmetic')
        self.assertEqual(len(edits), 3)
        for edit in edits:
            self.assertNotIn('Answer:', edit['prefix'])
            self.assertNotIn('The total', edit['prefix'])
            self.assertTrue(edit['control_prefix'].endswith(edit['original_step'] + '\n'))
            self.assertTrue(edit['prefix'].endswith(edit['perturbed_step'] + '\n'))

    def test_think_boundary(self):
        thinking = '<think>Answer: 999\nLong reasoning</think>\n'
        edits = interventions(thinking + TRACE, 'arithmetic')
        self.assertTrue(all(e['prefix'].startswith(thinking) for e in edits))
        self.assertEqual(extract_answer(thinking + 'Answer: 24', 'gsm8k'), '24')
        self.assertEqual(reasoning_steps('<think>unfinished\n' + TRACE)[1], [])
        self.assertIsNone(extract_answer('<think>unfinished\nAnswer: 24', 'gsm8k'))

    def test_short_trace_ineligible(self):
        self.assertEqual(interventions('One line 2\nAnswer: 2', 'arithmetic'), [])

    def test_denominators_and_unknowns(self):
        rows = [dict(perturbed_correct=False, control_correct=True),
                dict(perturbed_correct=True, control_correct=None),
                dict(perturbed_correct=None, control_correct=False)]
        result = summarize(rows)
        self.assertEqual(result['error_propagation_rate'], .5)
        self.assertEqual(result['answer_coverage'], 2/3)
        self.assertEqual(result['n_paired'], 1)
        self.assertEqual(result['paired_error_rate_difference'], 1)
        self.assertIsNone(summarize([])['error_propagation_rate'])

    def test_local_data_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.jsonl'
            row = dict(id='x', question='Q', reference_answer='1')
            path.write_text((json.dumps(row) + '\n') * 2)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_examples('gsm8k', 100, 42, input_file=path)

    def test_full_orchestration_with_fake_backend(self):
        class FakeBackend:
            def __init__(self):
                self.requests = []
            def generate(self, requests):
                self.requests.extend(requests)
                if requests[0][1] == '':
                    return [TRACE]
                return ['Answer: 99' if i % 2 == 0 else 'Answer: 24' for i in range(len(requests))]
        backend = FakeBackend()
        records = []
        with contextlib.redirect_stdout(io.StringIO()):
            result = run([dict(id='q1', question='Q', reference_answer='24')],
                         'gsm8k', backend, 'arithmetic', lambda name, row: records.append((name, row)))
        self.assertEqual(result['n_eligible_questions'], 1)
        self.assertEqual(result['error_propagation_rate'], 1)
        self.assertEqual(result['control_error_rate'], 0)
        self.assertEqual(len(backend.requests), 7)
        self.assertEqual(len(records), 4)

    def test_incorrect_baseline_never_intervened(self):
        class Backend:
            def generate(self, requests):
                self.assertions = requests
                return ['Answer: 99']
        with contextlib.redirect_stdout(io.StringIO()):
            result = run([dict(id='q1', question='Q', reference_answer='24')],
                         'gsm8k', Backend(), 'arithmetic', lambda *args: None)
        self.assertEqual(result['status'], 'insufficient_data')
        self.assertIsNone(result['error_propagation_rate'])


if __name__ == '__main__':
    unittest.main()

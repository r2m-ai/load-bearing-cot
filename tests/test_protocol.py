import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from evaluate import run
from load_bearing.datasets import load_examples
from load_bearing.protocol import (REASONING, correct, extract_answer, interventions, perturb, position_indices,
                                   reasoning_steps, summarize, unclosed_reasoning)

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
        # Four steps: early=1, middle=2, late would also be 2 and is skipped.
        self.assertEqual([e['position'] for e in edits], ['early', 'middle'])
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

    def test_non_default_reasoning_tags(self):
        tags = ('<reasoning>', '</reasoning>')
        thinking = '<reasoning>Answer: 999\nscratch</reasoning>\n'
        edits = interventions(thinking + TRACE, 'arithmetic', tags)
        self.assertEqual([e['position'] for e in edits], ['early', 'middle'])
        self.assertTrue(all(e['prefix'].startswith(thinking) for e in edits))
        self.assertEqual(extract_answer(thinking + 'Answer: 24', 'gsm8k', tags), '24')
        self.assertIsNone(extract_answer('<reasoning>unfinished\nAnswer: 24', 'gsm8k', tags))
        self.assertEqual(reasoning_steps('<reasoning>unfinished\n' + TRACE, tags)[1], [])
        # Under the default pair the preamble is not recognized: its lines count
        # as visible steps and the scratch answer is scored. Name the tags instead.
        self.assertEqual(extract_answer(thinking + 'Answer: 24', 'gsm8k'), '24')
        self.assertEqual(reasoning_steps(thinking + TRACE)[1][0], '<reasoning>Answer: 999')
        self.assertTrue(unclosed_reasoning('<reasoning>x', tags))
        self.assertFalse(unclosed_reasoning('<reasoning>x', REASONING))

    def test_run_fails_loudly_when_every_baseline_leaves_reasoning_open(self):
        class Backend:
            def generate(self, requests):
                return ['<think>still thinking ' + str(len(requests))]
        examples = [dict(id=f'q{i}', question='Q', reference_answer='24') for i in range(6)]
        with contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(ValueError, 'opened .* without closing'):
            run(examples, 'gsm8k', Backend(), 'arithmetic', lambda *args: None)
        with contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(ValueError, 'opened .* without closing'):
            run(examples[:2], 'gsm8k', Backend(), 'arithmetic', lambda *args: None)

    def test_occasional_unclosed_reasoning_is_counted_not_fatal(self):
        class Backend:
            def __init__(self):
                self.calls = 0
            def generate(self, requests):
                if requests[0][1] != '':
                    return ['Answer: 24'] * len(requests)
                self.calls += 1
                return ['<think>truncated'] if self.calls % 2 else ['<think>t</think>\n' + TRACE]
        examples = [dict(id=f'q{i}', question='Q', reference_answer='24') for i in range(4)]
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(examples, 'gsm8k', Backend(), 'arithmetic', lambda *args: None)
        self.assertEqual(result['n_baseline_unclosed_reasoning'], 2)
        self.assertEqual(result['n_baseline_unparsed'], 2)
        self.assertEqual(result['n_eligible_questions'], 2)

    def test_short_trace_ineligible(self):
        self.assertEqual(interventions('One line 2\nAnswer: 2', 'arithmetic'), [])

    def test_coincident_positions_are_not_duplicated(self):
        def trace(n):
            return ''.join(f'{i+1}. Step number {i+1} uses {i+1} items.\n' for i in range(n)) + 'Answer: 24'
        self.assertEqual(position_indices(3), [('early', 0), ('middle', 1)])
        self.assertEqual(position_indices(4), [('early', 1), ('middle', 2)])
        self.assertEqual(position_indices(5), [('early', 1), ('middle', 2), ('late', 3)])
        for n in (3, 4, 5, 8):
            edits = interventions(trace(n), 'arithmetic')
            keys = {(e['step_index'], e['prefix'], e['control_prefix']) for e in edits}
            with self.subTest(n=n):
                self.assertEqual(len(keys), len(edits))
                for edit in edits:
                    self.assertEqual(edit['n_steps_total'], n)
                    self.assertEqual(edit['n_steps_after_cut'], n - edit['step_index'] - 1)
                    self.assertGreaterEqual(edit['n_steps_after_cut'], 1)

    def test_mid_chain_answer_line_does_not_end_the_rationale(self):
        trace = ('1. There are 4 rows.\n2. Each row has 6 items.\n'
                 'Answer: not sure yet, let me redo step 3.\n'
                 '3. Multiply 4 by 6 to obtain 24.\n4. The total is 24.\nAnswer: 24')
        self.assertEqual(extract_answer(trace, 'gsm8k'), '24')
        _, steps = reasoning_steps(trace)
        # Scoring and step finding now agree on where the rationale ends; the
        # tentative answer line stays in the prefix but is not an editable step.
        self.assertEqual(len(steps), 4)
        self.assertNotIn('Answer: not sure', steps)
        edits = interventions(trace, 'arithmetic')
        self.assertEqual(len(edits), 2)
        self.assertTrue(all('Answer: 24' not in e['prefix'] for e in edits))
        self.assertIn('Answer: not sure yet', edits[1]['prefix'])

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
        # One baseline plus two distinct interventions x two conditions.
        self.assertEqual(len(backend.requests), 5)
        self.assertEqual(len(records), 3)
        self.assertEqual(result['n_coincident_positions_skipped'], 1)
        self.assertEqual(result['by_position']['late']['n_interventions'], 0)
        self.assertEqual(sorted(result['by_remaining_steps']), [1, 2])
        self.assertEqual(result['by_remaining_steps'][2]['n_interventions'], 1)

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

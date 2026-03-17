# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A programs database that implements the evolutionary algorithm."""
from __future__ import annotations

import profile
from collections import deque
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any, Tuple

from absl import logging
import numpy as np
import scipy

from implementation import code_manipulation
from implementation import config as config_lib

# RZ: I change the original code "tuple[float, ...]" to "Tuple[float, ...]"
Signature = Tuple[float, ...]

'''
Adrian: Updated ScoresPerTest Mapping
The evaluation function no longer just returns a float value,
but a dictionary that contains multiple metrics. See more in the main program.
'''
ScoresPerTest = Mapping[Any, float | Mapping[str, Any]]

'''
Adrian: Since the score provides more metrics, a function was developed
to extract the correct score. For backwards compatibility, it will
return the float value as is if it is not a dictionary.
'''
def _extract_primary_score(score_value: float | Mapping[str, Any]) -> float:
    if isinstance(score_value, Mapping):
        if 'primary' in score_value:
            return float(score_value['primary'])
        return 0.0
    return float(score_value)

'''
Adrian: Following from the function above, this function was developed
to extract the other metrics that were captured during the evaluation.
The function takes in the metric_name, as defined in the the evalute function.
'''
def _extract_numeric_metric(score_value: float | Mapping[str, Any], metric_name: str) -> float | None:
    """Extracts a numeric metric from a score payload, if present."""
    if not isinstance(score_value, Mapping):
        return None
    metric = score_value.get(metric_name, None)
    if isinstance(metric, (int, float)):
        return float(metric)
    return None

'''
Adrian: Function to summarize the scores/metrics gathered at each evaluation.
This will be stored in the ProgramDatabase later for evaluation.
'''
def _summarize_scores(scores_per_test: ScoresPerTest) -> dict[str, float]:
    """Builds aggregate metrics over tests for prompt feedback."""

    # Variables below align with the keys in the metrics payload
    primary_scores: list[float] = []
    avg_bins_used_values: list[float] = []
    avg_fullness_values: list[float] = []
    avg_wasted_space_values: list[float] = []
    wasted_space_std_values: list[float] = []
    pct_nearly_full_values: list[float] = []

    for test_id in scores_per_test:
        test_score = scores_per_test[test_id]
        primary = _extract_primary_score(test_score)
        primary_scores.append(primary)

        # Extract the metrics based on the keys below
        # then populate the list values defined above for all instances.
        for key, target in [
            ('avg_bins_used', avg_bins_used_values),
            ('avg_fullness', avg_fullness_values),
            ('avg_wasted_space', avg_wasted_space_values),
            ('wasted_space_std', wasted_space_std_values),
            ('pct_nearly_full', pct_nearly_full_values),
        ]:
            val = _extract_numeric_metric(test_score, key)
            if val is not None:
                target.append(val)

    summary = {
        'primary_score': sum(primary_scores) / len(primary_scores),
        'num_tests': float(len(primary_scores)),
    }

    if avg_bins_used_values:
        summary['avg_bins_used'] = sum(avg_bins_used_values) / len(avg_bins_used_values)
    else:
        # Backward-compatible proxy if only primary score exists.
        summary['avg_bins_used'] = -summary['primary_score']

    for key, values in [
        ('avg_fullness', avg_fullness_values),
        ('avg_wasted_space', avg_wasted_space_values),
        ('wasted_space_std', wasted_space_std_values),
        ('pct_nearly_full', pct_nearly_full_values),
    ]:
        if values:
            summary[key] = sum(values) / len(values)

    # This summary will contain the average score after running
    # evaluation across all instances.
    return summary


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    """Reduces per-test scores into a single score.
    """
    # Adrian: Changed the following to use _extract_primary_score such that it returns
    # the correct primary score value.
    test_scores = [_extract_primary_score(scores_per_test[k]) for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores)


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """Represents test scores as a canonical signature."""
    # Adrian: Same as the above.
    return tuple(_extract_primary_score(scores_per_test[k]) for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

    Attributes:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the implementations
         included in the prompt. Used to direct the newly generated implementation
         into the same island.
    """
    code: str
    version_generated: int
    island_id: int


class ProgramsDatabase:
    """A collection of programs, organized as islands."""

    def __init__(
            self,
            config: config_lib.ProgramsDatabaseConfig,
            template: code_manipulation.Program,
            function_to_evolve: str,
    ) -> None:
        self._config: config_lib.ProgramsDatabaseConfig = config
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve

        # Initialize empty islands.
        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(template, function_to_evolve, config.functions_per_prompt,
                       config.cluster_sampling_temperature_init,
                       config.cluster_sampling_temperature_period))
        self._best_score_per_island: list[float] = (
                [-float('inf')] * config.num_islands)
        self._best_program_per_island: list[code_manipulation.Function | None] = (
                [None] * config.num_islands)
        self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
                [None] * config.num_islands)
        # Adrian: In addition, I have included history per island to be used with _summarize_scores.
        # This will provide access to previous metrics during prompt generation.
        # This will also be a double-ended queue with a maximum length of 64, initialized to the length of the number of islands.
        # Once the 65th history appears for a certain island, the first one will be discarded to save memory.
        self._history_per_island: list[deque[dict[str, float]]] = [
            deque(maxlen=64) for _ in range(config.num_islands)
        ]

        self._last_reset_time: float = time.time()

    def get_prompt(self) -> Prompt:
        """Returns a prompt containing implementations from one chosen island."""
        # Adrian: Randomly gets an island for evolution
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)

    def _register_program_in_island(
            self,
            program: code_manipulation.Function,
            island_id: int,
            scores_per_test: ScoresPerTest,
            **kwargs  # RZ: add this for profiling
    ) -> None:
        """Registers `program` in the specified island."""
        self._islands[island_id].register_program(program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info('Best score of island %d increased to %s', island_id, score)

        '''
        Adrian: Mechanism for saving the historical metrics in a certain island
        In addition to the metrics defined in _summarize_scores, I have also included
        the sample_time (The time it was required to generate this heuristic) and
        evaluate_time (how long it took to run this heuritic for bin packing)
        '''
        summary = _summarize_scores(scores_per_test)
        sample_time = kwargs.get('sample_time', None)
        evaluate_time = kwargs.get('evaluate_time', None)
        if isinstance(sample_time, (int, float)):
            summary['sample_time'] = float(sample_time)
        if isinstance(evaluate_time, (int, float)):
            summary['evaluate_time'] = float(evaluate_time)
        # Finally append this dictionary of metrics to the history of this island
        self._history_per_island[island_id].append(summary)

        # ======== RZ: profiling ========
        profiler: profile.Profiler = kwargs.get('profiler', None)
        if profiler:
            global_sample_nums = kwargs.get('global_sample_nums', None)
            sample_time = kwargs.get('sample_time', None)
            evaluate_time = kwargs.get('evaluate_time', None)
            program.score = score
            program.global_sample_nums = global_sample_nums
            program.sample_time = sample_time
            program.evaluate_time = evaluate_time
            profiler.register_function(program)

    def register_program(
            self,
            program: code_manipulation.Function,
            island_id: int | None,
            scores_per_test: ScoresPerTest,
            **kwargs  # RZ: add this for profiling
    ) -> None:
        """Registers `program` in the database."""
        # In an asynchronous implementation we should consider the possibility of
        # registering a program on an island that had been reset after the prompt
        # was generated. Leaving that out here for simplicity.
        if island_id is None:
            # This is a program added at the beginning, so adding it to all islands.
            for island_id in range(len(self._islands)):
                self._register_program_in_island(program, island_id, scores_per_test, **kwargs)
        else:
            self._register_program_in_island(program, island_id, scores_per_test, **kwargs)

        # Check whether it is time to reset an island.
        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()

    '''
    Adrian: This is the main implementation of the Thought-Augmented and Direction-Extraction Module
    This will provide narrative feedback to the LLM based on the historical performance of the island.
    This will then guide the LLM in generating the next heuristic, while factoring in the recent performance of the island.
    '''
    def get_island_prompt_feedback(self, island_id: int) -> str:
        """Returns concise natural-language feedback for the next prompt."""
        history = list(self._history_per_island[island_id])

        # Starting phrase for all prompts.
        starting_phrase: str = """
        You are an Artificial Intelligence Heuritistic Expert.
        You are tasked to work on improving the following heuristic function for the Online Bin Packing problem.
        This heuristic function will determine how to pack items into bins in an online manner, and your goal is to minimize the number of bins used while maximizing the fullness of the bins.
        The code blocks you are seeing is the previous version of the heuristic function.
        As a heuristic expert, you are expected to analyze the previous version(s) of the heuristic function
        and generate a more advanced and complex, yet accurate function that can achieve better performance.
        """

        # Return instruction
        # Adrian: For testing only. This will change as we introduce the Analyst LLM and Programmer LLM.
        # The following instruction_phrase will only work for a single LLM instance.
        instruction_phrase: str = """
        Please elaborate and put comments over the code to explain how your heuristic function works.
        However, do not output anything other than the code itself and the comments within in.
        Strictly output the code of the heuristic function.
        """

        # Going to return an empty feedback as the island is still very early in its evolution
        # Historical figures will only make sense after at least 2 evolutions.
        if len(history) < 2:
            # There is not enough historical figures to generate a meaningful narrative feedback.
            # I will just return a simple instruction telling it to be creative for the first evolution.
            return starting_phrase + """
                Since this is the first iteration, no historical performances are available.
                Please be creative and come up with a heuristic function to kick start this evolution.
            """ + instruction_phrase

        # Trend evaluation based on the last 8 historical figures.
        window = history[-8:]
        current = window[-1]
        best_in_window = max(window, key=lambda x: x['primary_score'])

        # Takes the middle index, later for use to determine
        # the first half (old_window) and the second half (new_window)
        split_index = max(1, len(window) // 2)
        old_window = window[:split_index]
        new_window = window[split_index:]

        old_mean = sum(s['primary_score'] for s in old_window) / len(old_window)
        new_mean = sum(s['primary_score'] for s in new_window) / len(new_window)
        
        # Compares the old and the new mean with a delta value
        # Essentially this will tell LLM whether the performance is degrading or improving.
        delta = new_mean - old_mean

        if delta > 0.15:
            trend = 'improving'
        elif delta < -0.15:
            trend = 'regressing'
        else:
            trend = 'stable'

        advice_line = 'Try a structurally different heuristic while preserving strong cases.'
        if trend == 'improving':
            advice_line = 'Keep beneficial patterns and search for small gains in bin usage.'
        elif trend == 'regressing':
            advice_line = 'Recent changes hurt performance; prefer safer, less disruptive edits.'

        current_bins_used = current.get('avg_bins_used', -current['primary_score'])
        best_bins_used = best_in_window.get('avg_bins_used', -best_in_window['primary_score'])

        feedback_lines = [
            '# Evolution feedback from previous evaluated programs:',
            f"# - Island trend is {trend} (recent score delta {delta:+.3f}, higher is better).",
            f"# - Current average bins used is about {current_bins_used:.3f}; recent best is {best_bins_used:.3f}. (We want to minimize this number)",
            f"# - Current primary score is {current['primary_score']:.3f}; recent best is {best_in_window['primary_score']:.3f}. (We want to maximize this number)",
        ]

        if 'avg_fullness' in current and 'avg_fullness' in best_in_window:
            feedback_lines.append(
                f"# - Average bin fullness is {current['avg_fullness']:.4f * 100:.2f}%; recent best is {best_in_window['avg_fullness']:.4f * 100:.2f}%."
            )
        else:
            feedback_lines.append('# - Fullness metric is unavailable in current score payload; optimize bin usage signal first.')

        # Wasted space distribution
        if 'avg_wasted_space' in current:
            wasted = current['avg_wasted_space']
            std = current.get('wasted_space_std', 0.0)
            pct_full = current.get('pct_nearly_full', 0.0)
            best_wasted = best_in_window.get('avg_wasted_space', wasted)

            if std < 10:
                spread_desc = 'waste is consistent across bins'
            elif std < 25:
                spread_desc = 'waste varies moderately across bins'
            else:
                spread_desc = 'waste is highly uneven — some bins are nearly full, others are sparse'

            if pct_full > 0.6:
                tight_desc = f'{pct_full*100:.0f}% of bins are nearly full (under 10% capacity remaining) — good tightness'
            elif pct_full > 0.3:
                tight_desc = f'only {pct_full*100:.0f}% of bins are nearly full — room to improve packing density'
            else:
                tight_desc = f'only {pct_full*100:.0f}% of bins are nearly full — packing is loose, prioritize tighter fits'

            feedback_lines.append(
                f"# - Wasted space per bin: avg {wasted:.2f} units (best in window {best_wasted:.2f}), std {std:.2f} — {spread_desc}."
            )
            feedback_lines.append(f'# - {tight_desc}.')

        sample_times = [s['sample_time'] for s in window if 'sample_time' in s]
        eval_times = [s['evaluate_time'] for s in window if 'evaluate_time' in s]
        if sample_times and eval_times:
            feedback_lines.append(
                f"# - Average sample/evaluate time in this window: {sum(sample_times)/len(sample_times):.3f}s / {sum(eval_times)/len(eval_times):.3f}s."
            )

        feedback_lines.append(f'# - Guidance: {advice_line}')
        return starting_phrase + '\n'.join(feedback_lines) + instruction_phrase

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # We sort best scores after adding minor noise to break ties.
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6)
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period)
            self._best_score_per_island[island_id] = -float('inf')
            self._history_per_island[island_id].clear()
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            self._register_program_in_island(founder, island_id, founder_scores)


class Island:
    """A sub-population of the programs database."""

    def __init__(
            self,
            template: code_manipulation.Program,
            function_to_evolve: str,
            functions_per_prompt: int,
            cluster_sampling_temperature_init: float,
            cluster_sampling_temperature_period: int,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = (
            cluster_sampling_temperature_period)

        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0

    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        signature = _get_signature(scores_per_test)
        if signature not in self._clusters:
            score = _reduce_score(scores_per_test)
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1

    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing functions from this island."""
        signatures = list(self._clusters.keys())
        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures])

        # Convert scores to probabilities using softmax with temperature schedule.
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period)
        probabilities = _softmax(cluster_scores, temperature)

        # At the beginning of an experiment when we have few clusters, place fewer
        # programs into the prompt.
        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities)
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(
            self,
            implementations: Sequence[code_manipulation.Function]) -> str:
        """Creates a prompt containing a sequence of function `implementations`."""
        implementations = copy.deepcopy(implementations)  # We will mutate these.

        # Format the names and docstrings of functions to be included in the prompt.
        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            # Update the docstring for all subsequent functions after `_v0`.
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
            # If the function is recursive, replace calls to itself with its new name.
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(
                code_manipulation.text_to_function(implementation))

        # Create the header of the function to be generated by the LLM.
        next_version = len(implementations)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body='',
            docstring=('Improved version of '
                       f'`{self._function_to_evolve}_v{next_version - 1}`.'),
        )
        versioned_functions.append(header)

        # Replace functions in the template with the list constructed here.
        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        return str(prompt)


class Cluster:
    """A cluster of programs on the same island and with the same Signature."""

    def __init__(self, score: float, implementation: code_manipulation.Function):
        self._score = score
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        """Reduced score of the signature that this cluster represents."""
        return self._score

    def register_program(self, program: code_manipulation.Function) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> code_manipulation.Function:
        """Samples a program, giving higher probability to shorther programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
                max(self._lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)

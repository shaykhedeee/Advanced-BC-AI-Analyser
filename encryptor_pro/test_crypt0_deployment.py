import unittest

from crypt0_deployment import MultiAIOrchestrator


class StubOrchestrator(MultiAIOrchestrator):
    def __init__(self, config, local_models):
        super().__init__(config)
        self._local_models = local_models

    def discover_local_models(self):
        return self._local_models


class TestModelSelection(unittest.TestCase):
    def test_prefers_local_llama_model(self):
        orchestrator = StubOrchestrator(
            config={"max_analysis_models": 4},
            local_models=["mistral:7b", "llama3.1:70b-instruct", "qwen2.5:14b"],
        )

        selection = orchestrator.select_model(
            model="claude-3-opus-20240229",
            reasoning="reasoning",
        )

        self.assertEqual(selection.model, "llama3.1:70b-instruct")
        self.assertEqual(selection.source, "ollama-local")
        self.assertIn("llama3.1:70b-instruct", selection.analysis_models)

    def test_uses_requested_model_when_no_local_models(self):
        orchestrator = StubOrchestrator(config={}, local_models=[])

        selection = orchestrator.select_model(
            model="claude-3-opus-20240229",
            reasoning="reasoning",
        )

        self.assertEqual(selection.model, "claude-3-opus-20240229")
        self.assertEqual(selection.source, "requested-default")

    def test_weighted_majority_consensus(self):
        orchestrator = StubOrchestrator(config={}, local_models=[])
        outputs = {
            "llama3.1:70b-instruct": "Try strategy A.",
            "mistral:7b": "Try strategy B.",
            "qwen2.5:14b": "Try strategy B.",
        }

        result = orchestrator.merge_outputs(outputs)

        self.assertEqual(result.consensus_text, "Try strategy B.")
        self.assertEqual(set(result.winning_models), {"mistral:7b", "qwen2.5:14b"})

    def test_score_based_vote_can_override_raw_count(self):
        orchestrator = StubOrchestrator(config={}, local_models=[])
        outputs = {
            "llama3.1:70b-instruct": "Use high-confidence approach.",
            "tiny-rand:1b": "Use low-confidence approach.",
            "tiny-rand:1.3b": "Use low-confidence approach.",
        }

        result = orchestrator.merge_outputs(outputs)

        self.assertEqual(result.consensus_text, "Use high-confidence approach.")
        self.assertEqual(result.winning_models, ["llama3.1:70b-instruct"])

    def test_paraphrased_outputs_cluster_and_beat_unrelated_output(self):
        orchestrator = StubOrchestrator(config={}, local_models=[])
        outputs = {
            "tiny-rand:1b": "Rotate API keys regularly and monitor access logs.",
            "tiny-rand:1.3b": "Regularly rotate API keys while monitoring access logs.",
            "qwen2.5:14b": "Disable logging and keep one static key forever.",
        }

        result = orchestrator.merge_outputs(outputs)

        self.assertEqual(
            result.consensus_text,
            "Regularly rotate API keys while monitoring access logs.",
        )
        self.assertEqual(set(result.winning_models), {"tiny-rand:1b", "tiny-rand:1.3b"})

    def test_high_capability_model_breaks_close_semantic_cluster_tie_by_weight(self):
        orchestrator = StubOrchestrator(config={"semantic_similarity_threshold": 0.9}, local_models=[])
        outputs = {
            "llama3:8b": "Rotate API keys every 30 days and monitor access logs.",
            "tiny-rand:1b": "Rotate credentials every month and monitor audit logs.",
            "tiny-rand:1.3b": "Rotate credentials every month and monitor audit logs.",
        }

        result = orchestrator.merge_outputs(outputs)

        self.assertEqual(result.consensus_text, "Rotate API keys every 30 days and monitor access logs.")
        self.assertEqual(result.winning_models, ["llama3:8b"])


if __name__ == "__main__":
    unittest.main()

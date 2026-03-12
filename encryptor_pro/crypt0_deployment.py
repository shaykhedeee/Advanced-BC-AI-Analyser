import json
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class ModelSelection:
    model: str
    source: str
    reasoning: str
    analysis_models: List[str]


@dataclass
class EnsembleResult:
    consensus_text: str
    winning_models: List[str]
    vote_weights: Dict[str, float]
    raw_outputs: Dict[str, str]


class MultiAIOrchestrator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.ollama_base_url = str(config.get("ollama_base_url", "http://localhost:11434")).rstrip("/")
        self.max_analysis_models = int(config.get("max_analysis_models", 4))
        threshold = config.get("semantic_similarity_threshold", 0.7)
        self.semantic_similarity_threshold = self._clamp_threshold(threshold)

    def discover_local_models(self) -> List[str]:
        """Read installed models from local Ollama (`/api/tags`)."""
        endpoint = f"{self.ollama_base_url}/api/tags"
        try:
            with urlopen(endpoint, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError, OSError):
            return []

        models = payload.get("models", []) if isinstance(payload, dict) else []
        names: List[str] = []
        for model in models:
            if isinstance(model, dict) and isinstance(model.get("name"), str):
                names.append(model["name"])

        return sorted(set(names))

    def generate_with_model(self, model: str, prompt: str, system: str = "") -> str:
        """Generate output from a local Ollama model. Returns empty string on failure."""
        endpoint = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        request = Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=30) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError, OSError):
            return ""

        output = parsed.get("response", "") if isinstance(parsed, dict) else ""
        return output.strip() if isinstance(output, str) else ""

    def run_ensemble(self, prompt: str, selection: ModelSelection, system: str = "") -> EnsembleResult:
        """Run prompt across multiple models and merge outputs with weighted voting."""
        outputs: Dict[str, str] = {}
        for model_name in selection.analysis_models:
            text = self.generate_with_model(model_name, prompt=prompt, system=system)
            if text:
                outputs[model_name] = text

        if not outputs:
            outputs[selection.model] = "No model response available from local runtime."

        return self.merge_outputs(outputs)

    def merge_outputs(self, outputs: Dict[str, str]) -> EnsembleResult:
        """Merge model outputs using semantic clustering followed by weighted voting."""
        semantic_clusters: List[Dict[str, Any]] = []
        vote_weights: Dict[str, float] = {}

        for model_name, text in outputs.items():
            normalized = self._normalize_text(text)
            if not normalized:
                continue

            model_weight = self._model_weight(model_name)
            vote_weights[normalized] = vote_weights.get(normalized, 0.0) + model_weight

            best_cluster_index = -1
            best_similarity = 0.0
            for index, cluster in enumerate(semantic_clusters):
                similarity = self._cluster_similarity(normalized, cluster["normalized_texts"])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster_index = index

            if best_cluster_index >= 0 and best_similarity >= self.semantic_similarity_threshold:
                cluster = semantic_clusters[best_cluster_index]
            else:
                cluster = {
                    "models": [],
                    "normalized_texts": [],
                    "variant_models": {},
                    "variant_weights": {},
                    "variant_text": {},
                    "total_weight": 0.0,
                }
                semantic_clusters.append(cluster)

            cluster["models"].append(model_name)
            cluster["normalized_texts"].append(normalized)
            cluster["total_weight"] += model_weight

            variant_models = cluster["variant_models"]
            variant_weights = cluster["variant_weights"]
            variant_text = cluster["variant_text"]
            variant_models.setdefault(normalized, []).append(model_name)
            variant_weights[normalized] = variant_weights.get(normalized, 0.0) + model_weight
            variant_text.setdefault(normalized, text)

        if not semantic_clusters:
            return EnsembleResult(
                consensus_text="",
                winning_models=[],
                vote_weights={},
                raw_outputs=outputs,
            )

        winning_cluster = max(
            semantic_clusters,
            key=lambda cluster: (
                float(cluster["total_weight"]),
                len(cluster["models"]),
                max((self._model_weight(name) for name in cluster["models"]), default=0.0),
                max((float(weight) for weight in cluster["variant_weights"].values()), default=0.0),
            ),
        )

        winner_key = self._pick_consensus_variant(winning_cluster)

        return EnsembleResult(
            consensus_text=winning_cluster["variant_text"][winner_key],
            winning_models=list(winning_cluster["models"]),
            vote_weights=vote_weights,
            raw_outputs=outputs,
        )

    def select_model(self, model: str, reasoning: str) -> ModelSelection:
        """Select the best model, preferring local Llama models when available."""
        local_models = self.discover_local_models()
        best_local = self._pick_best_model(local_models)

        configured_pool = self._normalize_model_list(self.config.get("model_pool", []))
        analysis_pool = self._normalize_model_list(self.config.get("analysis_models", []))

        if best_local:
            selected = best_local
            source = "ollama-local"
        elif model:
            selected = model
            source = "requested-default"
        elif configured_pool:
            selected = configured_pool[0]
            source = "config-fallback"
        else:
            selected = "llama3:8b"
            source = "built-in-fallback"

        combined = local_models + analysis_pool + configured_pool + [selected]
        analysis_models = self._dedupe(combined)[: self.max_analysis_models]

        return ModelSelection(
            model=selected,
            source=source,
            reasoning=reasoning,
            analysis_models=analysis_models,
        )

    def _pick_best_model(self, models: List[str]) -> str:
        if not models:
            return ""

        def score(name: str) -> float:
            lowered = name.lower()
            base = 0.0

            if "llama" in lowered:
                base += 100
            if "qwen" in lowered:
                base += 60
            if "mistral" in lowered:
                base += 50

            match = re.search(r"(\d+(?:\.\d+)?)\s*b", lowered)
            if match:
                base += float(match.group(1))

            if "instruct" in lowered:
                base += 5
            if "coder" in lowered:
                base += 3

            return base

        return max(models, key=score)

    @staticmethod
    def _normalize_model_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str) and item.strip()]

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        return normalized

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
        normalized_tokens: List[str] = []
        for token in raw_tokens:
            if len(token) > 4 and token.endswith("ing"):
                token = token[:-3]
            elif len(token) > 3 and token.endswith("ed"):
                token = token[:-2]
            elif len(token) > 3 and token.endswith("s"):
                token = token[:-1]
            normalized_tokens.append(token)
        return normalized_tokens

    @staticmethod
    def _semantic_similarity(left: str, right: str) -> float:
        if not left or not right:
            return 0.0

        left_tokens = set(MultiAIOrchestrator._tokenize_text(left))
        right_tokens = set(MultiAIOrchestrator._tokenize_text(right))
        shared = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)

        jaccard = (shared / union) if union else 0.0
        min_size = min(len(left_tokens), len(right_tokens))
        overlap = (shared / min_size) if min_size else 0.0
        sequence_ratio = SequenceMatcher(None, left, right).ratio()

        token_score = (jaccard + overlap) / 2.0
        similarity = (0.7 * token_score) + (0.3 * sequence_ratio)

        symmetric_difference = len(left_tokens ^ right_tokens)
        if min_size <= 4 and shared >= (min_size - 1) and symmetric_difference == 2:
            similarity *= 0.7

        return similarity

    @staticmethod
    def _cluster_similarity(candidate: str, members: List[str]) -> float:
        if not members:
            return 0.0
        return max(MultiAIOrchestrator._semantic_similarity(candidate, member) for member in members)

    def _pick_consensus_variant(self, cluster: Dict[str, Any]) -> str:
        variant_weights: Dict[str, float] = cluster["variant_weights"]
        variant_models: Dict[str, List[str]] = cluster["variant_models"]

        return max(
            variant_weights.keys(),
            key=lambda key: (
                variant_weights.get(key, 0.0),
                len(variant_models.get(key, [])),
                max((self._model_weight(name) for name in variant_models.get(key, [])), default=0.0),
            ),
        )

    @staticmethod
    def _clamp_threshold(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.7
        return max(0.0, min(1.0, numeric))

    def _model_weight(self, model_name: str) -> float:
        """Convert model quality score to a positive voting weight."""
        return 1.0 + (self._score_model_name(model_name) / 100.0)

    def _score_model_name(self, name: str) -> float:
        lowered = name.lower()
        base = 0.0

        if "llama" in lowered:
            base += 100
        if "qwen" in lowered:
            base += 60
        if "mistral" in lowered:
            base += 50

        match = re.search(r"(\d+(?:\.\d+)?)\s*b", lowered)
        if match:
            base += float(match.group(1))

        if "instruct" in lowered:
            base += 5
        if "coder" in lowered:
            base += 3

        return base

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        seen = set()
        output: List[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                output.append(value)
        return output


class CRYPT0Agent:
    def __init__(
        self,
        config: Dict[str, Any],
        skillset: str,
        selection: ModelSelection,
        orchestrator: MultiAIOrchestrator,
    ) -> None:
        self.config = config
        self.skillset = skillset
        self.selection = selection
        self.orchestrator = orchestrator

    def start_workspace_scan(self) -> None:
        print(f"[CRYPT0] Workspace scan started with model: {self.selection.model}")

    def begin_learning_cycle(self) -> None:
        print("[CRYPT0] Learning cycle started.")
        print(f"[CRYPT0] Analysis model pool: {', '.join(self.selection.analysis_models)}")

    def analyze_with_weighted_voting(self, prompt: str, system: str = "") -> EnsembleResult:
        """Run analysis across model pool and return weighted-voting consensus."""
        return self.orchestrator.run_ensemble(prompt=prompt, selection=self.selection, system=system)


class CRYPT0Deployment:
    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent
        self.config_path = base_dir / "crypt0_agent_config.json"
        self.skillset_path = base_dir / "CRYPT0_SKILLSET.md"

    def deploy(self) -> CRYPT0Agent:
        """Deploy CRYPT0 agent with full capabilities."""
        print("Initializing CRYPT0 deployment...")

        config = self._load_config()
        skillset = self._load_skillset()

        orchestrator = MultiAIOrchestrator(config)
        selection = orchestrator.select_model(
            model="claude-3-opus-20240229",
            reasoning="Advanced cryptographic analysis and tool development requires maximum reasoning capabilities for complex code synthesis and pattern recognition",
        )

        config["resolved_model"] = selection.model
        config["analysis_models"] = selection.analysis_models
        config["model_selection_source"] = selection.source
        config["model_selection_reasoning"] = selection.reasoning
        config["voting_strategy"] = "weighted-majority-score"

        print(f"[CRYPT0] Selected model: {selection.model} ({selection.source})")

        agent = self.initialize_agent(config, skillset, selection, orchestrator)
        agent.start_workspace_scan()
        agent.begin_learning_cycle()

        return agent

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Missing config file: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Config must be a JSON object.")

        return data

    def _load_skillset(self) -> str:
        if not self.skillset_path.exists():
            raise FileNotFoundError(f"Missing skillset file: {self.skillset_path}")

        skillset = self.skillset_path.read_text(encoding="utf-8").strip()
        if not skillset:
            raise ValueError("Skillset file is empty.")

        return skillset

    def initialize_agent(
        self,
        config: Dict[str, Any],
        skillset: str,
        selection: ModelSelection,
        orchestrator: MultiAIOrchestrator,
    ) -> CRYPT0Agent:
        """Initialize and return a CRYPT0 agent instance."""
        return CRYPT0Agent(
            config=config,
            skillset=skillset,
            selection=selection,
            orchestrator=orchestrator,
        )


if __name__ == "__main__":
    deployment = CRYPT0Deployment()
    deployment.deploy()


"""
rt_detr_analyzer.py — RT-DETR Visual Game Screen Analyzer for Edge Tracker
===========================================================================
Uses the RT-DETR (Real-Time DEtection TRansformer) model to analyze game
screenshots in real time — detecting UI elements, reading multipliers,
identifying game states, and feeding structured data into the AI pipeline.

Architecture:
  Screenshot → RT-DETR → Detected Regions → OCR/Classification → Structured Data
                                                                       ↓
                                                              AI Brain Pipeline

Capabilities:
  - Detect multiplier displays, buttons, graphs, bet panels
  - Read crash multiplier values from game UI via OCR
  - Track game state transitions (waiting → playing → crashed)
  - Feed visual data into the AI Brain for richer predictions
  - Fine-tune RT-DETR on custom game UI elements

Requires: pip install transformers torch torchvision pillow
Optional: pip install pytesseract easyocr mss (for screen capture + OCR)
"""

import os
import time
import json
import threading
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from collections import deque
from io import BytesIO

# Core dependencies
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# RT-DETR from HuggingFace transformers
RTDETR_AVAILABLE = False
try:
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    RTDETR_AVAILABLE = True
except ImportError:
    pass

# OCR engines (optional — for reading multiplier text)
EASYOCR_AVAILABLE = False
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass

TESSERACT_AVAILABLE = False
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pass

# Screen capture (optional — for live game monitoring)
MSS_AVAILABLE = False
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    pass


# =====================================================================
# Game UI Element Labels (for fine-tuning or post-processing)
# =====================================================================
GAME_UI_LABELS = {
    0: "multiplier_display",   # The big crash multiplier number
    1: "bet_panel",            # Where you place bets
    2: "cashout_button",       # Cash out button
    3: "graph_area",           # The crash curve graph
    4: "chat_area",            # Chat panel
    5: "history_bar",          # Recent crash history
    6: "balance_display",      # Player balance
    7: "timer_countdown",      # Pre-game countdown
    8: "player_list",          # Active players panel
    9: "payout_text",          # Payout multiplier text
}


class RTDetrGameAnalyzer:
    """
    Real-Time DEtection TRansformer for game screen analysis.

    Uses PekingU/rtdetr_r50vd (or fine-tuned variant) to detect
    objects in game screenshots, then extracts structured data
    from detected regions.
    """

    def __init__(
        self,
        model_name: str = "PekingU/rtdetr_r50vd",
        custom_model_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.custom_model_path = custom_model_path
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")

        self.model = None
        self.processor = None
        self.is_loaded = False
        self._lock = threading.Lock()

        # Analysis history
        self.detection_history = deque(maxlen=500)
        self.multiplier_history = deque(maxlen=1000)
        self.game_state_log = deque(maxlen=200)

        # OCR engine
        self.ocr_reader = None

        # Performance stats
        self.stats = {
            "frames_analyzed": 0,
            "avg_inference_ms": 0,
            "detections_total": 0,
            "multipliers_read": 0,
        }

    # -----------------------------------------------------------------
    # Model Loading
    # -----------------------------------------------------------------
    def load_model(self):
        """Load the RT-DETR model and image processor."""
        if not RTDETR_AVAILABLE:
            print("[RT-DETR] transformers library not available.")
            print("         Install: pip install transformers torch torchvision")
            return False

        try:
            path = self.custom_model_path or self.model_name
            print(f"[RT-DETR] Loading model: {path} ...")

            self.processor = RTDetrImageProcessor.from_pretrained(path)
            self.model = RTDetrForObjectDetection.from_pretrained(path)
            self.model = self.model.to(self.device)
            self.model.eval()

            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"[RT-DETR] Model loaded on {self.device} ({num_params/1e6:.1f}M params)")
            self.is_loaded = True
            return True

        except Exception as e:
            print(f"[RT-DETR] Failed to load model: {e}")
            return False

    def _ensure_loaded(self):
        if not self.is_loaded:
            self.load_model()

    # -----------------------------------------------------------------
    # OCR Setup
    # -----------------------------------------------------------------
    def setup_ocr(self):
        """Initialize OCR engine for reading text from detected regions."""
        if EASYOCR_AVAILABLE:
            print("[RT-DETR] Setting up EasyOCR ...")
            self.ocr_reader = easyocr.Reader(["en"], gpu=self.device == "cuda")
            print("[RT-DETR] EasyOCR ready")
        elif TESSERACT_AVAILABLE:
            print("[RT-DETR] Using Tesseract OCR")
            self.ocr_reader = "tesseract"
        else:
            print("[RT-DETR] No OCR engine available.")
            print("         Install: pip install easyocr  OR  pip install pytesseract")

    def _read_text_from_region(self, image: Image.Image, box: list) -> str:
        """Extract text from a detected bounding box region."""
        if self.ocr_reader is None:
            return ""

        x1, y1, x2, y2 = [int(c) for c in box]
        # Clamp to image bounds
        w, h = image.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return ""

        crop = image.crop((x1, y1, x2, y2))

        if EASYOCR_AVAILABLE and self.ocr_reader != "tesseract":
            arr = np.array(crop)
            results = self.ocr_reader.readtext(arr, detail=0)
            return " ".join(results).strip()
        elif TESSERACT_AVAILABLE:
            return pytesseract.image_to_string(crop).strip()
        return ""

    # -----------------------------------------------------------------
    # Core Detection
    # -----------------------------------------------------------------
    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run RT-DETR object detection on a game screenshot.

        Args:
            image: PIL Image of the game screen

        Returns:
            Dict with detected objects, boxes, scores, labels
        """
        self._ensure_loaded()
        if not self.is_loaded:
            return {"error": "Model not loaded", "detections": []}

        start_time = time.perf_counter()

        with self._lock, torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            target_sizes = torch.tensor(
                [(image.height, image.width)], device=self.device
            )
            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold,
            )[0]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Build structured detections list
        detections = []
        id2label = self.model.config.id2label

        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            score_val = score.item()
            label_idx = label_id.item()
            box_coords = [round(c, 2) for c in box.tolist()]
            label_name = id2label.get(label_idx, f"class_{label_idx}")

            detections.append({
                "label": label_name,
                "label_id": label_idx,
                "score": round(score_val, 4),
                "box": box_coords,  # [x1, y1, x2, y2]
            })

        # Update stats
        self.stats["frames_analyzed"] += 1
        self.stats["detections_total"] += len(detections)
        n = self.stats["frames_analyzed"]
        self.stats["avg_inference_ms"] = (
            self.stats["avg_inference_ms"] * (n - 1) + elapsed_ms
        ) / n

        result = {
            "detections": detections,
            "count": len(detections),
            "inference_ms": round(elapsed_ms, 2),
            "image_size": (image.width, image.height),
            "threshold": self.confidence_threshold,
        }

        self.detection_history.append(result)
        return result

    def detect_from_path(self, image_path: str) -> Dict[str, Any]:
        """Detect objects from a file path."""
        image = Image.open(image_path).convert("RGB")
        return self.detect(image)

    def detect_from_screenshot(self, monitor_idx: int = 1) -> Dict[str, Any]:
        """Capture screen and run detection (requires mss)."""
        if not MSS_AVAILABLE:
            return {"error": "mss not installed. pip install mss"}

        with mss.mss() as sct:
            monitor = sct.monitors[monitor_idx]
            screenshot = sct.grab(monitor)
            image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        return self.detect(image)

    # -----------------------------------------------------------------
    # Game-Specific Analysis
    # -----------------------------------------------------------------
    def analyze_game_screen(self, image: Image.Image) -> Dict[str, Any]:
        """
        Full game screen analysis pipeline:
        1. Run RT-DETR detection
        2. Read text from key regions (OCR)
        3. Map detections to game UI elements
        4. Extract multiplier value
        5. Determine game state
        """
        raw = self.detect(image)
        if "error" in raw:
            return raw

        analysis = {
            "detections": raw["detections"],
            "inference_ms": raw["inference_ms"],
            "multiplier": None,
            "game_state": "unknown",
            "balance": None,
            "ui_elements": {},
        }

        # Try to read text from each detected region
        for det in raw["detections"]:
            region_text = self._read_text_from_region(image, det["box"])
            det["text"] = region_text

            label = det["label"].lower()

            # Map common COCO labels to game UI concepts heuristically
            # (until fine-tuned on game screenshots)
            if any(kw in label for kw in ["tv", "monitor", "screen", "clock"]):
                # Could be a multiplier display or timer
                multiplier = self._extract_multiplier(region_text)
                if multiplier is not None:
                    analysis["multiplier"] = multiplier
                    analysis["ui_elements"]["multiplier_display"] = det

            elif any(kw in label for kw in ["keyboard", "laptop", "cell phone"]):
                analysis["ui_elements"]["bet_panel"] = det

            elif "book" in label or "dining table" in label:
                analysis["ui_elements"]["history_bar"] = det

        # If we found a multiplier, log it
        if analysis["multiplier"] is not None:
            self.multiplier_history.append({
                "time": time.time(),
                "value": analysis["multiplier"],
            })
            self.stats["multipliers_read"] += 1

        # Determine game state from visual cues
        analysis["game_state"] = self._infer_game_state(analysis)
        self.game_state_log.append({
            "time": time.time(),
            "state": analysis["game_state"],
            "multiplier": analysis["multiplier"],
        })

        return analysis

    def _extract_multiplier(self, text: str) -> Optional[float]:
        """Parse a multiplier value from OCR text (e.g., '2.45x', '12.3')."""
        if not text:
            return None
        import re
        # Match patterns like "2.45x", "2.45X", "2.45", "x2.45"
        patterns = [
            r'(\d+\.\d+)\s*[xX]',  # 2.45x
            r'[xX]\s*(\d+\.\d+)',   # x2.45
            r'(\d+\.\d{1,2})',       # 2.45
            r'(\d+)[xX]',           # 2x
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    val = float(match.group(1))
                    if 1.0 <= val <= 10000.0:  # reasonable range
                        return round(val, 2)
                except ValueError:
                    continue
        return None

    def _infer_game_state(self, analysis: Dict) -> str:
        """Infer game state from visual analysis."""
        mult = analysis.get("multiplier")
        detections = analysis.get("detections", [])
        num_objects = len(detections)

        if mult is not None:
            if mult >= 1.0:
                return "playing"
            return "crashed"

        if num_objects == 0:
            return "loading"

        # Check recent history for state transitions
        if len(self.game_state_log) >= 2:
            prev = self.game_state_log[-1].get("state", "unknown")
            if prev == "playing" and mult is None:
                return "crashed"

        return "waiting"

    # -----------------------------------------------------------------
    # Annotated Visualization
    # -----------------------------------------------------------------
    def draw_detections(self, image: Image.Image, detections: list) -> Image.Image:
        """Draw bounding boxes on the image for visualization."""
        draw = ImageDraw.Draw(image)

        colors = {
            "multiplier_display": "#00ff00",
            "bet_panel": "#ff6600",
            "graph_area": "#0066ff",
            "default": "#ff0000",
        }

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            score = det["score"]
            color = colors.get(label, colors["default"])

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            text = f"{label} {score:.0%}"
            draw.text((x1, y1 - 12), text, fill=color)

        return image

    # -----------------------------------------------------------------
    # Pipeline Data Output (for AI Brain integration)
    # -----------------------------------------------------------------
    def get_pipeline_data(self) -> Dict[str, Any]:
        """
        Get structured data suitable for the AI Brain pipeline.
        Returns recent multipliers, game states, detection stats.
        """
        recent_mults = [
            entry["value"] for entry in self.multiplier_history
        ][-50:]

        recent_states = [
            entry["state"] for entry in self.game_state_log
        ][-20:]

        state_counts = {}
        for s in recent_states:
            state_counts[s] = state_counts.get(s, 0) + 1

        return {
            "source": "rt_detr_vision",
            "multipliers_detected": recent_mults,
            "multiplier_count": len(recent_mults),
            "current_state": recent_states[-1] if recent_states else "unknown",
            "state_distribution": state_counts,
            "avg_multiplier": float(np.mean(recent_mults)) if recent_mults else None,
            "stats": dict(self.stats),
        }

    def get_visual_features(self, n_recent: int = 20) -> Optional[np.ndarray]:
        """
        Convert recent visual detections into a numerical feature vector
        that can be fed into ML models alongside game data.
        """
        recent_mults = [
            entry["value"] for entry in self.multiplier_history
        ][-n_recent:]

        if len(recent_mults) < 5:
            return None

        arr = np.array(recent_mults, dtype=np.float32)
        features = np.array([
            np.mean(arr),
            np.std(arr),
            np.median(arr),
            np.min(arr),
            np.max(arr),
            np.percentile(arr, 25),
            np.percentile(arr, 75),
            np.sum(arr < 2.0) / len(arr),
            np.sum(arr >= 5.0) / len(arr),
            arr[-1] - arr[-2] if len(arr) >= 2 else 0,
        ], dtype=np.float32)

        return features

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        return {
            "model_loaded": self.is_loaded,
            "model_name": self.custom_model_path or self.model_name,
            "device": self.device,
            "ocr_available": self.ocr_reader is not None,
            "rtdetr_available": RTDETR_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "screen_capture_available": MSS_AVAILABLE,
            "stats": dict(self.stats),
            "multipliers_in_history": len(self.multiplier_history),
        }


# =====================================================================
# Fine-Tuning RT-DETR on Game Screenshots
# =====================================================================
class RTDetrGameFineTuner:
    """
    Fine-tune RT-DETR on custom game UI screenshots.

    Workflow:
    1. Collect screenshots with labeled bounding boxes (COCO format)
    2. Run this fine-tuner to adapt RT-DETR to your game's UI
    3. Save the model for use by RTDetrGameAnalyzer

    Label format (COCO JSON):
    {
        "images": [{"id": 1, "file_name": "screenshot_001.png", "width": 1920, "height": 1080}],
        "annotations": [{"image_id": 1, "category_id": 0, "bbox": [x, y, w, h], "area": ...}],
        "categories": [{"id": 0, "name": "multiplier_display"}, ...]
    }
    """

    def __init__(
        self,
        base_model: str = "PekingU/rtdetr_r50vd",
        output_dir: str = "./rtdetr_game_model",
        num_labels: int = 10,
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.num_labels = num_labels

    def prepare_dataset(self, annotations_path: str, images_dir: str):
        """Load COCO-format annotations for fine-tuning."""
        if not RTDETR_AVAILABLE:
            print("[RT-DETR Fine-Tune] transformers not available")
            return None

        with open(annotations_path) as f:
            coco_data = json.load(f)

        print(f"[RT-DETR Fine-Tune] {len(coco_data.get('images', []))} images, "
              f"{len(coco_data.get('annotations', []))} annotations, "
              f"{len(coco_data.get('categories', []))} categories")

        return coco_data

    def finetune(
        self,
        annotations_path: str,
        images_dir: str,
        epochs: int = 20,
        batch_size: int = 2,
        learning_rate: float = 1e-4,
    ):
        """
        Fine-tune RT-DETR on game screenshots.

        This uses the HuggingFace Trainer API under the hood.
        Requires: pip install transformers[torch] accelerate
        """
        if not RTDETR_AVAILABLE or not TORCH_AVAILABLE:
            print("[RT-DETR Fine-Tune] Missing dependencies")
            return

        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        print(f"[RT-DETR Fine-Tune] Loading base model: {self.base_model}")
        processor = RTDetrImageProcessor.from_pretrained(self.base_model)

        # Load model with updated number of labels for game UI
        id2label = {v: k for k, v in GAME_UI_LABELS.items()}
        label2id = dict(GAME_UI_LABELS)

        model = RTDetrForObjectDetection.from_pretrained(
            self.base_model,
            num_labels=self.num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        # Load COCO annotations
        coco_data = self.prepare_dataset(annotations_path, images_dir)
        if coco_data is None:
            return

        # Build image -> annotations mapping
        img_map = {img["id"]: img for img in coco_data["images"]}
        ann_by_image = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in ann_by_image:
                ann_by_image[img_id] = []
            ann_by_image[img_id].append(ann)

        # Simple training loop (alternative to full Trainer for small datasets)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        print(f"[RT-DETR Fine-Tune] Training for {epochs} epochs on {device}")
        for epoch in range(epochs):
            total_loss = 0
            count = 0

            for img_id, img_info in img_map.items():
                img_path = os.path.join(images_dir, img_info["file_name"])
                if not os.path.exists(img_path):
                    continue

                image = Image.open(img_path).convert("RGB")
                anns = ann_by_image.get(img_id, [])
                if not anns:
                    continue

                # Prepare labels
                boxes = []
                class_labels = []
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    # Convert COCO [x,y,w,h] to [cx,cy,w,h] normalized
                    cx = (x + w / 2) / image.width
                    cy = (y + h / 2) / image.height
                    nw = w / image.width
                    nh = h / image.height
                    boxes.append([cx, cy, nw, nh])
                    class_labels.append(ann["category_id"])

                labels = [{
                    "class_labels": torch.tensor(class_labels, dtype=torch.long).to(device),
                    "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                }]

                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs, labels=labels)

                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / max(count, 1)
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Images: {count}")

        # Save fine-tuned model
        os.makedirs(self.output_dir, exist_ok=True)
        model.save_pretrained(self.output_dir)
        processor.save_pretrained(self.output_dir)
        print(f"[RT-DETR Fine-Tune] Model saved to {self.output_dir}/")

    def create_sample_annotations(self, output_path: str = "game_annotations.json"):
        """Create a template COCO annotation file for labeling."""
        template = {
            "images": [
                {"id": 1, "file_name": "screenshot_001.png", "width": 1920, "height": 1080}
            ],
            "annotations": [
                {
                    "id": 1, "image_id": 1, "category_id": 0,
                    "bbox": [800, 200, 320, 100],
                    "area": 32000, "iscrowd": 0,
                }
            ],
            "categories": [
                {"id": idx, "name": name}
                for idx, name in GAME_UI_LABELS.items()
            ]
        }

        with open(output_path, "w") as f:
            json.dump(template, f, indent=2)

        print(f"[RT-DETR] Template annotations saved to {output_path}")
        print("  Edit this file with your actual bounding boxes.")
        print(f"  Categories: {list(GAME_UI_LABELS.values())}")


# =====================================================================
# Live Game Monitor (combines screen capture + RT-DETR + OCR)
# =====================================================================
class LiveGameMonitor:
    """
    Continuously captures game screen and feeds analysis
    into the Edge Tracker AI pipeline.
    """

    def __init__(
        self,
        analyzer: Optional[RTDetrGameAnalyzer] = None,
        capture_interval: float = 1.0,
        monitor_idx: int = 1,
    ):
        self.analyzer = analyzer or RTDetrGameAnalyzer()
        self.capture_interval = capture_interval
        self.monitor_idx = monitor_idx
        self._running = False
        self._thread = None
        self.callbacks = []

    def add_callback(self, fn):
        """Add a callback that receives analysis results."""
        self.callbacks.append(fn)

    def start(self):
        """Start live monitoring in background thread."""
        if not MSS_AVAILABLE:
            print("[LiveGameMonitor] mss not installed. pip install mss")
            return

        if not self.analyzer.is_loaded:
            self.analyzer.load_model()
            self.analyzer.setup_ocr()

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"[LiveGameMonitor] Started (interval: {self.capture_interval}s)")

    def stop(self):
        """Stop live monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[LiveGameMonitor] Stopped")

    def _monitor_loop(self):
        while self._running:
            try:
                with mss.mss() as sct:
                    monitor = sct.monitors[self.monitor_idx]
                    screenshot = sct.grab(monitor)
                    image = Image.frombytes(
                        "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
                    )

                analysis = self.analyzer.analyze_game_screen(image)

                for cb in self.callbacks:
                    try:
                        cb(analysis)
                    except Exception:
                        pass

            except Exception as e:
                print(f"[LiveGameMonitor] Error: {e}")

            time.sleep(self.capture_interval)


# =====================================================================
# Module-level convenience
# =====================================================================
_analyzer_instance: Optional[RTDetrGameAnalyzer] = None


def get_analyzer() -> RTDetrGameAnalyzer:
    """Get or create the global RT-DETR analyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = RTDetrGameAnalyzer()
    return _analyzer_instance


# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  RT-DETR Visual Game Screen Analyzer")
    print("=" * 65)

    analyzer = RTDetrGameAnalyzer()
    print(f"\nStatus: {json.dumps(analyzer.get_status(), indent=2)}")

    if RTDETR_AVAILABLE:
        print("\nLoading RT-DETR model ...")
        analyzer.load_model()

        # Demo: create fine-tune template
        finetuner = RTDetrGameFineTuner()
        finetuner.create_sample_annotations("game_annotations_template.json")

        print("\nTo analyze a screenshot:")
        print("  result = analyzer.detect_from_path('screenshot.png')")
        print("  result = analyzer.analyze_game_screen(pil_image)")
        print("\nTo start live monitoring:")
        print("  monitor = LiveGameMonitor(analyzer)")
        print("  monitor.start()")
    else:
        print("\nRT-DETR not available. Install:")
        print("  pip install transformers torch torchvision")

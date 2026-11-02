"""
Vision-Language Model (VLM) sub-module for dense captioning and OCR.
Supports Florence-2 with graceful fallback.
"""

from __future__ import annotations
import os
from typing import List, Tuple, Optional, Dict, Any, Union
from PIL import Image
import numpy as np
import cv2
import torch

from ..models import RawFrame, RegionCaption, OCRResult
from ..config import VLMSettings
from ..utils.logging import setup_logger
from ..utils.gpu import get_optimal_device

logger = setup_logger("analysis.vlm")


class VisionLanguageModel:
    """
    Multimodal scene understanding using Florence-2 (dense captioning, OCR with regions,
    detailed scene descriptions).
    """

    def __init__(self, settings: Optional[VLMSettings] = None):
        self.settings = settings or VLMSettings()
        self.device = get_optimal_device(self.settings.device)
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info(f"Attempting to load Florence-2 model '{self.settings.model}' on {self.device}...")
            import os
            os.environ["USE_TF"] = "0"
            os.environ["USE_TORCH"] = "1"

            import transformers
            if not hasattr(transformers.PretrainedConfig, "forced_bos_token_id"):
                transformers.PretrainedConfig.forced_bos_token_id = None
            if hasattr(transformers, "tokenization_utils_base") and hasattr(transformers.tokenization_utils_base, "PreTrainedTokenizerBase"):
                transformers.tokenization_utils_base.PreTrainedTokenizerBase.additional_special_tokens = property(
                    lambda self: getattr(self, "_additional_special_tokens", [])
                )

            from transformers import AutoModelForCausalLM, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.settings.model,
                trust_remote_code=True,
            )
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.model,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()
            logger.info("Florence-2 loaded successfully!")
        except Exception as e:
            logger.warning(
                f"Failed to load Florence-2 ({e}). "
                f"Falling back to heuristic / OCR fallback engine."
            )
            self.model = None
            self.processor = None

    def _run_florence_task(
        self,
        pil_image: Image.Image,
        task_prompt: str,
        text_input: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs a single Florence-2 task on a PIL Image."""
        if self.model is None or self.processor is None:
            return {}

        prompt = task_prompt if text_input is None else task_prompt + text_input
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device.type == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=min(self.settings.max_new_tokens, 256),
                num_beams=1,
                early_stopping=False,
                do_sample=False,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(pil_image.width, pil_image.height),
        )
        return parsed_answer

    def _fallback_process(
        self,
        frame: RawFrame,
    ) -> Tuple[str, List[RegionCaption], List[OCRResult]]:
        """
        Graceful heuristic fallback when Florence-2 is unavailable or disabled.
        Generates sensible scene descriptions and extracts text using OpenCV / Tesseract if present.
        """
        w, h = frame.width, frame.height
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        light_condition = "brightly lit" if brightness > 128 else "low light"

        caption = f"A video scene under {light_condition} conditions showing visual elements at {frame.timestamp_sec:.2f} seconds."
        region_captions: List[RegionCaption] = []
        ocr_results: List[OCRResult] = []

        # Try pytesseract for OCR fallback
        try:
            import pytesseract
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])
                if text and conf > 40:
                    x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    x2, y2 = x + bw, y + bh
                    norm_bbox = [x / w, y / h, x2 / w, y2 / h]
                    ocr_results.append(
                        OCRResult(
                            text=text,
                            confidence=conf / 100.0,
                            bbox_xyxy=[float(x), float(y), float(x2), float(y2)],
                            bbox_normalized=norm_bbox,
                        )
                    )
        except Exception:
            pass

        return caption, region_captions, ocr_results

    def process_frame(
        self,
        frame: RawFrame,
    ) -> Tuple[str, List[RegionCaption], List[OCRResult]]:
        """
        Processes a single frame for captioning, dense region captions, and OCR.
        Returns (frame_caption, region_captions, ocr_results).
        """
        if self.model is None or self.processor is None:
            return self._fallback_process(frame)

        # Convert BGR to RGB PIL Image
        rgb_image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        w, h = frame.width, frame.height

        # Optimization: Downscale high-res frames for VLM to max 1024px to prevent quadratic vision attention overhead
        max_vlm_dim = 1024
        orig_w, orig_h = pil_image.size
        if max(orig_w, orig_h) > max_vlm_dim:
            scale = max_vlm_dim / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            vlm_image = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        else:
            vlm_image = pil_image
            new_w, new_h = orig_w, orig_h

        detailed_caption = ""
        region_captions: List[RegionCaption] = []
        ocr_results: List[OCRResult] = []

        try:
            # 1. Detailed Caption
            if "MORE_DETAILED_CAPTION" in self.settings.tasks:
                res = self._run_florence_task(vlm_image, "<MORE_DETAILED_CAPTION>")
                detailed_caption = res.get("<MORE_DETAILED_CAPTION>", "")

            # 2. Dense Region Caption (skipped if disabled in config)
            if "DENSE_REGION_CAPTION" in self.settings.tasks:
                res = self._run_florence_task(vlm_image, "<DENSE_REGION_CAPTION>")
                dense_data = res.get("<DENSE_REGION_CAPTION>", {})
                bboxes = dense_data.get("bboxes", [])
                labels = dense_data.get("labels", [])
                for box, lbl in zip(bboxes, labels):
                    x1, y1, x2, y2 = [float(b) for b in box]
                    norm_box = [x1 / new_w, y1 / new_h, x2 / new_w, y2 / new_h]
                    region_captions.append(
                        RegionCaption(
                            caption=lbl,
                            bbox_xyxy=[norm_box[0] * w, norm_box[1] * h, norm_box[2] * w, norm_box[3] * h],
                            bbox_normalized=norm_box,
                        )
                    )

            # 3. OCR With Regions
            if "OCR_WITH_REGION" in self.settings.tasks:
                res = self._run_florence_task(vlm_image, "<OCR_WITH_REGION>")
                ocr_data = res.get("<OCR_WITH_REGION>", {})
                quad_boxes = ocr_data.get("quad_boxes", [])
                labels = ocr_data.get("labels", [])
                for quad, text in zip(quad_boxes, labels):
                    xs = quad[0::2]
                    ys = quad[1::2]
                    x1, x2 = float(min(xs)), float(max(xs))
                    y1, y2 = float(min(ys)), float(max(ys))
                    norm_box = [x1 / new_w, y1 / new_h, x2 / new_w, y2 / new_h]
                    ocr_results.append(
                        OCRResult(
                            text=str(text).strip(),
                            confidence=0.9,
                            bbox_xyxy=[norm_box[0] * w, norm_box[1] * h, norm_box[2] * w, norm_box[3] * h],
                            bbox_normalized=norm_box,
                        )
                    )

            if not detailed_caption:
                detailed_caption = f"Scene captured at {frame.timestamp_sec:.2f}s."

        except Exception as e:
            logger.error(f"Error in Florence-2 inference: {e}", exc_info=True)
            return self._fallback_process(frame)

        return detailed_caption, region_captions, ocr_results

    def answer_vqa(self, image: Union[np.ndarray, Image.Image, str], question: str) -> str:
        """
        Answers a natural language question about an image using Florence-2 VQA.
        """
        if self.model is None or self.processor is None:
            return ""

        pil_img = None
        if isinstance(image, str):
            if os.path.exists(image):
                pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            pil_img = image

        if pil_img is None:
            return ""

        # Optimization: Downscale high-res frames for VQA to max 1024px
        if max(pil_img.size) > 1024:
            scale = 1024 / max(pil_img.size)
            pil_img = pil_img.resize((int(pil_img.width * scale), int(pil_img.height * scale)), Image.Resampling.BILINEAR)

        try:
            # Check if this is a text, plate, or OCR question
            q_lower = question.lower()
            is_text_q = any(w in q_lower for w in ["license", "plate", "number", "text", "read", "written", "sign", "word", "letter", "ocr"])
            task = "<DocVQA>" if is_text_q else "<VQA>"
            prompt = f"{task}{question.strip()}"

            inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.device.type == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=128,
                    num_beams=1,
                    early_stopping=False,
                    do_sample=False,
                )

            gen_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(
                gen_text, task=task, image_size=pil_img.size
            )
            raw_ans = parsed.get(task, "").strip()

            # Clean up Florence-2 task prefixes (VQA>, QA>, DocVQA>, DocQA>, etc.)
            import re
            clean_ans = re.sub(r"^(VQA>|QA>|DocVQA>|DocQA>|MORE_DETAILED_CAPTION>|\s*)+", "", raw_ans, flags=re.IGNORECASE)
            # Strip XML/HTML location and special token tags
            clean_ans = re.sub(r"<[^>]+>", "", clean_ans).strip()

            # Guard against Florence-2 echoing the question or generating empty/junk text
            norm_q = re.sub(r"[^\w\s]", "", question.lower()).strip()
            norm_a = re.sub(r"[^\w\s]", "", clean_ans.lower()).strip()
            if not norm_a or norm_a == norm_q or norm_q in norm_a or len(norm_a) < 2:
                # If DocVQA failed/echoed, try one fallback with standard <VQA>
                if task == "<DocVQA>":
                    fallback_inputs = self.processor(text=f"<VQA>{question.strip()}", images=pil_img, return_tensors="pt")
                    fallback_inputs = {k: v.to(self.device) for k, v in fallback_inputs.items()}
                    if self.device.type == "cuda":
                        fallback_inputs["pixel_values"] = fallback_inputs["pixel_values"].to(torch.float16)
                    with torch.no_grad():
                        fb_ids = self.model.generate(
                            input_ids=fallback_inputs["input_ids"],
                            pixel_values=fallback_inputs["pixel_values"],
                            max_new_tokens=64,
                            num_beams=1,
                            early_stopping=False,
                            do_sample=False,
                        )
                    fb_gen = self.processor.batch_decode(fb_ids, skip_special_tokens=False)[0]
                    fb_parsed = self.processor.post_process_generation(fb_gen, task="<VQA>", image_size=pil_img.size)
                    fb_raw = fb_parsed.get("<VQA>", "").strip()
                    fb_clean = re.sub(r"^(VQA>|QA>|DocVQA>|DocQA>|\s*)+", "", fb_raw, flags=re.IGNORECASE)
                    fb_clean = re.sub(r"<[^>]+>", "", fb_clean).strip()
                    fb_norm = re.sub(r"[^\w\s]", "", fb_clean.lower()).strip()
                    if fb_norm and fb_norm != norm_q and norm_q not in fb_norm:
                        return fb_clean
                return ""

            return clean_ans
        except Exception as e:
            logger.warning(f"Error in Florence-2 VQA inference: {e}")
            return ""

    def process_batch(
        self,
        frames: List[RawFrame],
    ) -> List[Tuple[str, List[RegionCaption], List[OCRResult]]]:
        """Processes a batch of frames sequentially or in parallel."""
        return [self.process_frame(f) for f in frames]


"""
RAG Query Engine orchestrating analysis, retrieval, re-ranking, and LLM synthesis.
"""

from __future__ import annotations
import os
import json
import re
from typing import Optional, List, Dict, Any

from ..models import QueryResponse, TimestampReference, SearchResult
from ..config import SystemConfig
from ..storage.vector_store import QdrantVectorStore
from ..storage.summary_store import SQLiteSummaryStore
from ..embedding.encoder import MultimodalEncoder
from ..analysis.vlm import VisionLanguageModel
from .analyzer import QueryAnalyzer
from .searcher import HybridSearcher
from .reranker import CrossEncoderReranker
from .context import ContextAssembler
from ..utils.logging import setup_logger
from ..utils.ffmpeg import format_timestamp

logger = setup_logger("query.engine")


class RAGQueryEngine:
    """
    End-to-end multimodal RAG query engine.
    Supports external LLMs (Gemini / OpenAI) as well as direct local VLM
    Visual Question Answering on retrieved video frames.
    """

    def __init__(
        self,
        config: SystemConfig,
        vector_store: QdrantVectorStore,
        summary_store: SQLiteSummaryStore,
        encoder: MultimodalEncoder,
        vlm: Optional[VisionLanguageModel] = None,
    ):
        self.config = config
        self.vector_store = vector_store
        self.summary_store = summary_store
        self.encoder = encoder
        self.vlm = vlm

        self.analyzer = QueryAnalyzer()
        self.searcher = HybridSearcher(
            vector_store=self.vector_store,
            encoder=self.encoder,
            score_threshold=self.config.query.score_threshold,
            temporal_window_sec=self.config.query.temporal_cluster_window_sec,
        )
        self.reranker = CrossEncoderReranker(settings=self.config.query)
        self.context_assembler = ContextAssembler()

        # Initialize LLM client
        self.api_key = os.getenv(self.config.llm.api_key_env, "")
        self.llm_client = None
        self.local_llm = None
        self.local_tokenizer = None
        self._init_llm()

    def _init_llm(self) -> None:
        if self.api_key:
            if self.config.llm.provider.lower() in ("google", "gemini"):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    self.llm_client = genai.GenerativeModel(self.config.llm.model)
                    logger.info(f"Initialized Gemini model '{self.config.llm.model}' for RAG synthesis")
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini: {e}")
            elif self.config.llm.provider.lower() == "openai":
                try:
                    from openai import OpenAI
                    self.llm_client = OpenAI(api_key=self.api_key)
                    logger.info("Initialized OpenAI client for RAG synthesis")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI: {e}")
        else:
            logger.info("No cloud API key found. Initializing fast local neural chat assistant...")
            self._init_local_llm()

    def _init_local_llm(self) -> None:
        """Loads lightweight instruction-tuned local LLM for fluent, natural QA conversation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            logger.info(f"Loading local chat model '{model_name}' on GPU...")
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.local_llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
            )
            self.local_llm.eval()
            logger.info("Local conversational chat assistant ready on GPU!")
        except Exception as e:
            logger.warning(f"Could not load local LLM ({e}). Will use multimodal rule synthesis fallback.")
            self.local_llm = None
            self.local_tokenizer = None

    def _local_llm_synthesis(
        self,
        query: str,
        results: List[SearchResult],
        summary: Optional[Any] = None,
    ) -> QueryResponse:
        """
        Uses local neural chat assistant (Qwen2.5-0.5B-Instruct) to synthesize
        fluent, conversational, natural English answers from retrieved evidence.
        """
        if not results:
            return QueryResponse(
                query=query,
                answer="I could not find any events or elements in the video matching your request.",
                confidence=0.0,
                evidence_frames=[],
                follow_up_suggestions=["Try describing the object, color, action, or time interval differently."],
            )

        # Cluster results chronologically
        sorted_candidates = sorted(results, key=lambda r: float(r.payload.get("timestamp_sec", 0.0)))
        clusters = []
        for r in sorted_candidates:
            ts = float(r.payload.get("timestamp_sec", 0.0))
            if not clusters or (ts - clusters[-1][-1]["ts"] > 3.0):
                clusters.append([{"result": r, "ts": ts}])
            else:
                clusters[-1].append({"result": r, "ts": ts})

        timestamps: List[TimestampReference] = []
        evidence_lines = []

        if summary and summary.narrative:
            clean_narrative = re.sub(r"<[^>]+>", "", summary.narrative).strip()
            evidence_lines.append(f"Global Video Narrative: {clean_narrative}")

        for cluster in clusters:
            start_ts = cluster[0]["ts"]
            end_ts = cluster[-1]["ts"]
            best_item = max(cluster, key=lambda x: x["result"].score)
            p = best_item["result"].payload
            f_idx = p.get("frame_index", 0)

            range_str = (
                format_timestamp(start_ts)
                if start_ts == end_ts
                else f"{format_timestamp(start_ts)} - {format_timestamp(end_ts)}"
            )

            cap = re.sub(r"<[^>]+>", "", p.get("frame_caption", "")).strip()
            det = re.sub(r"<[^>]+>", "", p.get("detection_summary", "")).strip()
            ocr = re.sub(r"<[^>]+>", "", p.get("ocr_full", "")).strip()
            ocr = re.sub(r"\s+", " ", ocr)

            ev_entry = f"Evidence Segment [{range_str}]:\n  - Scene Description: {cap}"
            if det and det != "No prominent objects detected":
                ev_entry += f"\n  - Objects: {det}"
            if ocr:
                ev_entry += f"\n  - Visible Text (OCR): {ocr}"
            evidence_lines.append(ev_entry)

            timestamps.append(
                TimestampReference(
                    time_sec=start_ts,
                    time_formatted=range_str,
                    description=cap[:120] if cap else "Activity observed",
                    frame_index=f_idx,
                )
            )

        evidence_str = "\n".join(evidence_lines)

        answer = ""
        if self.local_llm is not None and self.local_tokenizer is not None:
            try:
                import torch
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert video analysis chat assistant. "
                            "Answer the user's question directly, accurately, and conversationally based ONLY on the provided video evidence. "
                            "Directly state the requested property (e.g. the color, make, action, count, or license plate number). "
                            "Always cite the relevant timestamp [HH:MM:SS]. "
                            "Keep your response concise and natural (1-2 sentences)."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"VIDEO EVIDENCE:\n{evidence_str}\n\nUSER QUESTION: {query}",
                    },
                ]
                text = self.local_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.local_tokenizer([text], return_tensors="pt").to(self.local_llm.device)
                with torch.no_grad():
                    out = self.local_llm.generate(
                        **inputs,
                        max_new_tokens=120,
                        do_sample=False,
                        repetition_penalty=1.1,
                    )
                generated_tokens = out[0][len(inputs.input_ids[0]):]
                raw_ans = self.local_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                answer = re.sub(r"<[^>]+>", "", raw_ans).strip()
            except Exception as e:
                logger.warning(f"Error in local LLM synthesis: {e}")
                answer = ""

        # Fallback to direct heuristic synthesis if local LLM generation was empty
        if not answer:
            return self._vlm_direct_synthesis(query, results, summary)

        top_confidence = float(results[0].score) if results else 0.9
        if top_confidence > 1.0 or top_confidence < 0.0:
            import math
            top_confidence = 1.0 / (1.0 + math.exp(-top_confidence))

        return QueryResponse(
            query=query,
            answer=answer,
            timestamps=timestamps,
            confidence=round(top_confidence, 2),
            evidence_frames=[r.payload for r in results],
            follow_up_suggestions=[
                "Would you like more details about specific moments in this interval?",
                "Do you want to check for any other entities in this scene?",
            ],
            search_metadata={"synthesis_engine": "local_qwen_chat_llm"},
        )

    def _call_llm(self, prompt: str) -> str:
        """Invokes LLM and returns response text."""
        if self.llm_client is None:
            return ""

        if self.config.llm.provider.lower() in ("google", "gemini"):
            response = self.llm_client.generate_content(
                prompt,
                generation_config={
                    "temperature": self.config.llm.temperature,
                    "max_output_tokens": self.config.llm.max_tokens,
                    "response_mime_type": "application/json",
                },
            )
            return response.text
        elif self.config.llm.provider.lower() == "openai":
            response = self.llm_client.chat.completions.create(
                model=self.config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        return ""

    def _vlm_direct_synthesis(
        self,
        query: str,
        results: List[SearchResult],
        summary: Optional[Any] = None,
    ) -> QueryResponse:
        """
        Uses multimodal evidence (Florence-2 captions, dense OCR, object detections,
        video summary narrative, and VQA) to construct a direct, accurate natural language answer.
        """
        if not results:
            return QueryResponse(
                query=query,
                answer="I could not find any events or elements in the video matching your request.",
                confidence=0.0,
                evidence_frames=[],
                follow_up_suggestions=["Try describing the object, color, action, or time interval differently."],
            )

        # 1. Cluster proximate timestamps (within 3.0 seconds) to avoid duplicate spam
        sorted_candidates = sorted(results, key=lambda r: float(r.payload.get("timestamp_sec", 0.0)))
        clusters = []
        for r in sorted_candidates:
            ts = float(r.payload.get("timestamp_sec", 0.0))
            if not clusters or (ts - clusters[-1][-1]["ts"] > 3.0):
                clusters.append([{"result": r, "ts": ts}])
            else:
                clusters[-1].append({"result": r, "ts": ts})

        timestamps: List[TimestampReference] = []
        cluster_summaries = []

        q_lower = query.lower()
        is_text_or_plate = any(
            w in q_lower
            for w in ["license", "plate", "number", "text", "read", "written", "sign", "word", "letter", "ocr"]
        )

        for cluster in clusters:
            start_ts = cluster[0]["ts"]
            end_ts = cluster[-1]["ts"]
            best_item = max(cluster, key=lambda x: x["result"].score)
            p = best_item["result"].payload
            img_path = p.get("frame_image_path")
            f_idx = p.get("frame_index", 0)

            range_str = (
                format_timestamp(start_ts)
                if start_ts == end_ts
                else f"{format_timestamp(start_ts)} - {format_timestamp(end_ts)}"
            )

            cap = p.get("frame_caption", "")
            det = p.get("detection_summary", "")
            ocr = p.get("ocr_full", "")

            # Sanitize raw tags and special tokens
            clean_cap = re.sub(r"<[^>]+>", "", cap).strip()
            clean_ocr = re.sub(r"<[^>]+>", "", ocr).strip()
            clean_ocr = re.sub(r"\s+", " ", clean_ocr)
            clean_det = re.sub(r"<[^>]+>", "", det).strip()

            # Run VLM Visual Question Answering if VLM and image exist
            vqa_ans = ""
            if self.vlm is not None and img_path and os.path.exists(img_path):
                raw_vqa = self.vlm.answer_vqa(img_path, query)
                clean_vqa = re.sub(r"^(VQA>|QA>|DocVQA>|\s*)+", "", raw_vqa, flags=re.IGNORECASE)
                clean_vqa = re.sub(r"<[^>]+>", "", clean_vqa).strip()
                # Verify that VLM did not just echo the query or output trivial tokens
                norm_q = re.sub(r"[^\w\s]", "", query.lower()).strip()
                norm_a = re.sub(r"[^\w\s]", "", clean_vqa.lower()).strip()
                if norm_a and norm_a != norm_q and norm_q not in norm_a and len(norm_a) >= 2:
                    vqa_ans = clean_vqa

            # Assemble clean description for the timestamp reference pill
            desc_parts = []
            if vqa_ans:
                desc_parts.append(vqa_ans)
            elif clean_cap:
                desc_parts.append(clean_cap)
            if clean_det and clean_det != "No prominent objects detected":
                desc_parts.append(f"Objects: {clean_det}")
            if clean_ocr:
                desc_parts.append(f"Text: '{clean_ocr}'")

            cluster_desc = " | ".join(desc_parts)

            timestamps.append(
                TimestampReference(
                    time_sec=start_ts,
                    time_formatted=range_str,
                    description=cluster_desc,
                    frame_index=f_idx,
                )
            )

            cluster_summaries.append({
                "range_str": range_str,
                "caption": clean_cap,
                "ocr": clean_ocr,
                "det": clean_det,
                "vqa": vqa_ans,
            })

        # 2. Intelligent Answer Synthesis
        primary = cluster_summaries[0]
        narrative = summary.narrative if summary else ""
        clean_narrative = re.sub(r"<[^>]+>", "", narrative)

        answer = ""

        # Case A: Query asks for license plate, registration, or visible text/signs
        if is_text_or_plate:
            corpus = f"{clean_narrative} " + " ".join(
                [c["caption"] + " " + c["ocr"] + " " + c["vqa"] for c in cluster_summaries]
            )
            # Find license plates, vehicle numbers, or quoted text
            plate_matches = re.findall(r'\b[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}[0-9]{1,4}\b', corpus)
            # Exclude common non-plate tokens
            plate_matches = [
                p for p in plate_matches
                if not p.lower() in ("fps", "vqa", "docvqa", "video", "utc", "gmt", "sec", "min", "honda", "ford", "bmw")
            ]
            quoted_matches = re.findall(r'"([^"]+)"', corpus)

            # Find caption/summary sentence discussing the license plate or text
            all_sentences = [
                s.strip() for s in re.split(r'\.\s+', clean_narrative + ". " + primary["caption"]) if s.strip()
            ]
            plate_sentence = next(
                (s for s in all_sentences if any(k in s.lower() for k in ["license", "plate", "reads"])), ""
            )

            best_identifier = ""
            if plate_matches:
                best_identifier = plate_matches[0]
            elif quoted_matches:
                best_identifier = quoted_matches[0]
            elif primary["vqa"]:
                best_identifier = primary["vqa"]
            elif primary["ocr"]:
                best_identifier = primary["ocr"].split()[0]

            if best_identifier:
                if plate_sentence:
                    answer = f"{plate_sentence.rstrip('.')}. The identified license number is **{best_identifier}**"
                else:
                    answer = f"The identified license / text is **{best_identifier}**"

                if primary["ocr"]:
                    answer += f" (detected OCR text: '{primary['ocr']}')"
                answer += f", clearly visible in the video between [{primary['range_str']}]."
            else:
                answer = f"Around [{primary['range_str']}], text detected in scene: '{primary['ocr'] or 'No clear text'}'."

        # Case B: General Scene / Object / Activity Question
        if not answer:
            # Score individual sentences in caption and summary by token relevance to query
            candidate_sentences = [
                s.strip() for s in re.split(r'\.\s+', primary["caption"] + ". " + clean_narrative) if s.strip()
            ]
            q_keywords = set(re.findall(r'\w+', q_lower)) - {
                'what', 'is', 'the', 'of', 'in', 'color', 'a', 'an', 'are', 'there', 'who', 'and', 'to', 'how'
            }

            best_sentence = None
            best_score = -1
            for s in candidate_sentences:
                s_words = set(re.findall(r'\w+', s.lower()))
                common = q_keywords.intersection(s_words)
                score = len(common)
                if score > best_score:
                    best_score = score
                    best_sentence = s

            if primary["vqa"]:
                answer = f"Around [{primary['range_str']}], {primary['vqa'].rstrip('.')}."
                if best_sentence and best_score > 0 and best_sentence.lower() not in primary["vqa"].lower():
                    answer += f" {best_sentence.rstrip('.')}."
            elif best_sentence and best_score > 0:
                answer = f"Around [{primary['range_str']}], {best_sentence.rstrip('.')}."
            elif primary["caption"]:
                answer = f"Around [{primary['range_str']}], {primary['caption'].rstrip('.')}."
            else:
                answer = f"The query '{query}' matches activity between [{primary['range_str']}]."

            if primary["det"] and primary["det"] != "No prominent objects detected":
                answer += f" Identified entities: {primary['det']}."

        # If multiple distinct temporal clusters exist, append concise bullet points
        if len(cluster_summaries) > 1:
            answer += "\n\n**Additional Occurrences:**\n"
            for c in cluster_summaries[1:]:
                c_text = c["vqa"] or c["caption"][:100]
                answer += f"- [{c['range_str']}] {c_text.rstrip('.')}."

        top_confidence = float(results[0].score) if results else 0.8
        # Ensure confidence is normalized between 0.0 and 1.0
        if top_confidence > 1.0 or top_confidence < 0.0:
            import math
            top_confidence = 1.0 / (1.0 + math.exp(-top_confidence))

        return QueryResponse(
            query=query,
            answer=answer,
            timestamps=timestamps,
            confidence=round(top_confidence, 2),
            evidence_frames=[r.payload for r in results],
            follow_up_suggestions=[
                "Would you like more details about specific moments in this interval?",
                "Do you want to check for any other entities in this scene?",
            ],
            search_metadata={"synthesis_engine": "multimodal_grounded_synthesis"},
        )

    def ask(
        self,
        query: str,
        video_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> QueryResponse:
        """
        Processes a natural language query end-to-end:
        1. Analyzes query and extracts filters
        2. Retrieves candidate frames
        3. Re-ranks candidates
        4. Fetches video summary context
        5. Prompts LLM and synthesizes structured response
        """
        analyzed = self.analyzer.analyze(
            query=query,
            video_id=video_id,
            top_k=top_k or self.config.query.default_top_k,
        )

        # Retrieve candidates
        candidates = self.searcher.search(analyzed)

        # Re-rank candidates
        reranked = self.reranker.rerank(
            query=analyzed.search_text,
            results=candidates,
            top_n=self.config.query.reranker_top_n,
        )

        # Fetch video summary if video_id is targeted
        summary = None
        target_vid = analyzed.target_video_id or (reranked[0].payload.get("video_id") if reranked else None)
        if target_vid:
            summary = self.summary_store.get_summary(target_vid)

        # If no external LLM configured, use local chat LLM or VLM direct synthesis
        if self.llm_client is None:
            if self.local_llm is not None:
                resp = self._local_llm_synthesis(query, reranked, summary)
            else:
                resp = self._vlm_direct_synthesis(query, reranked, summary)
            resp.video_summary_used = summary is not None
            return resp

        # Assemble prompt and call LLM
        prompt = self.context_assembler.assemble_prompt(
            query=query,
            results=reranked,
            summary=summary,
        )

        try:
            llm_text = self._call_llm(prompt)
            data = json.loads(llm_text)

            ts_refs: List[TimestampReference] = []
            for t in data.get("timestamps", []):
                t_sec = float(t.get("time_sec", 0.0))
                ts_refs.append(
                    TimestampReference(
                        time_sec=t_sec,
                        time_formatted=t.get("time_formatted") or format_timestamp(t_sec),
                        description=t.get("description", ""),
                        frame_index=int(t.get("frame_index", 0)),
                    )
                )

            return QueryResponse(
                query=query,
                answer=data.get("answer", ""),
                timestamps=ts_refs,
                confidence=float(data.get("confidence", 0.8)),
                evidence_frames=[r.payload for r in reranked],
                video_summary_used=summary is not None,
                follow_up_suggestions=data.get("follow_up_suggestions", []),
                search_metadata={
                    "total_candidates": len(candidates),
                    "reranked_count": len(reranked),
                    "vector_name": analyzed.vector_name,
                },
            )

        except Exception as e:
            logger.error(f"Error parsing LLM response ({e}). Using VLM direct synthesis.", exc_info=True)
            resp = self._vlm_direct_synthesis(query, reranked, summary)
            resp.video_summary_used = summary is not None
            return resp


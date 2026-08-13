"""
LLM Adapters - Provider-agnostic interfaces for different language models.

This module provides a unified interface for working with different LLM providers
including OpenAI, Anthropic Claude, and Google Gemini.
"""

import os
import base64
import re
from typing import Optional, List, Dict
from datetime import datetime
from dotenv import load_dotenv
import io
import torch
from PIL import Image

# =========================
# Base LLM Adapter
# =========================
class BaseLLMAdapter:
    """
    Minimal interface the app relies on:
      - build_user_content(init_canvas_b64, text) -> provider-specific "content" array
      - call(system_message, messages, additional_args) -> raw response
      - extract_text(raw_response) -> str content
      - request_has_image(messages) -> bool
    """
    def __init__(self, model: str, cache: bool = False, max_tokens: int = 3000):
        self.model = model
        self.cache = cache
        self.max_tokens = max_tokens

    def build_user_content(self, init_canvas_b64: Optional[str], text: str):
        raise NotImplementedError

    def call(self, system_message, messages, additional_args):
        raise NotImplementedError

    def extract_text(self, raw_response) -> str:
        raise NotImplementedError

    def request_has_image(self, messages: List[Dict]) -> bool:
        """Return True iff any message uses an image (image or image_url)."""
        for m in messages:
            for part in m.get("content", []):
                t = part.get("type")
                if t in ("image", "image_url"):
                    return True
        return False
    
    def response_metadata(self, raw_response) -> dict:
        """Provider-specific metadata (finish_reason, usage, safety, etc.)."""
        return {}

    def debug_dump(self, raw_response) -> dict:
        """Small, JSON-serializable snapshot of the raw provider response."""
        try:
            return {"repr": repr(raw_response)[:8000]}
        except Exception:
            return {}


# =========================
# Gemini Adapter (2.5 Pro/Flash)
# =========================
class GeminiAdapter(BaseLLMAdapter):
    """
    Works with Gemini 2.5 Pro (and Flash). Accepts your existing 'messages'
    shape (OpenAI/Claude style) and converts to Gemini {role, parts} format.
    If the response has no .text, we aggregate all text parts.
    """

    def __init__(self, model: str, cache: bool = False, max_tokens: int = 8192):
        super().__init__(model, cache, max_tokens)
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = None
        self._last_response = None
        self._last_sys = None

        # permissive safety so evals don't silently block
        self.safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT,         "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,        "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,             "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,  "threshold": HarmBlockThreshold.BLOCK_NONE},
        ]

    # ---------- helpers ----------
    def _ensure_model(self, system_message: str):
        if self._model is None or self._last_sys != system_message:
            self._model = self._genai.GenerativeModel(
                self.model,
                system_instruction=system_message,
                safety_settings=self.safety_settings
            )
            self._last_sys = system_message

    @staticmethod
    def _decode_data_url(url: str):
        """
        Accepts data URLs like 'data:image/png;base64,AAAA...'
        Returns (mime_type:str, raw_bytes:bytes) or (None, None) if not a data URL.
        """
        m = re.match(r"^data:(.*?);base64,(.+)$", url)
        if not m:
            return None, None
        mime = m.group(1)
        raw = base64.b64decode(m.group(2))
        return mime, raw

    def _parts_from_our_content(self, content):
        """
        Convert your OpenAI/Claude-style 'content' list into Gemini 'parts'.
        """
        parts = []
        for item in content or []:
            # If a stray string sneaks in, treat it as text
            if isinstance(item, str):
                parts.append({"text": item})
                continue

            t = item.get("type")
            if t == "text":
                parts.append({"text": item.get("text", "")})

            elif t == "image":
                src = item.get("source", {})
                if src.get("type") == "base64":
                    mt = src.get("media_type", "image/png")
                    data_b = base64.b64decode(src.get("data", "") or b"")
                    parts.append({"inline_data": {"mime_type": mt, "data": data_b}})

            elif t == "image_url":
                # Expected to be a *data URL* for local canvas
                url = (item.get("image_url") or {}).get("url", "")
                mt, data_b = self._decode_data_url(url)
                if mt and data_b:
                    parts.append({"inline_data": {"mime_type": mt, "data": data_b}})

            # ignore unknown types quietly
        return parts

    def _to_gemini_messages(self, messages):
        """
        Turn your messages into a list of {role, parts}.
        'assistant' -> 'model' role for Gemini.
        """
        out = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")

            # if content is already a string, make it a single text part
            if isinstance(content, str):
                parts = [{"text": content}]
            else:
                parts = self._parts_from_our_content(content)

            g_role = "model" if role == "assistant" else "user"
            out.append({"role": g_role, "parts": parts})
        return out

    # ---------- interface ----------
    def build_user_content(self, init_canvas_b64: Optional[str], text: str):
        """
        Keep your app's contract: returns a list of parts in OpenAI/Claude shape.
        """
        content = []
        if init_canvas_b64:
            mime = "image/jpeg" if init_canvas_b64.startswith("/9j") else "image/png"
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": init_canvas_b64}
            })
        content.append({"type": "text", "text": text})
        return content

    def call(self, system_message, messages, additional_args):
        self._ensure_model(system_message)

        gen_cfg = {"max_output_tokens": self.max_tokens}
        if "temperature" in additional_args:
            gen_cfg["temperature"] = additional_args["temperature"]

        # force single candidate
        gen_cfg["candidate_count"] = 1
        
        # Gemini supports stop_sequences in generation_config
        ss = additional_args.get("stop_sequences")
        if ss:
            gen_cfg["stop_sequences"] = ss if isinstance(ss, list) else [ss]

        # Prefer structured {role, parts}. If something looks off, fallback to flat text.
        try:
            contents = self._to_gemini_messages(messages)
            resp = self._model.generate_content(contents=contents, generation_config=gen_cfg)
        except Exception:
            # Fallback: flatten all text into a single prompt string
            text_blocks = []
            for m in messages:
                c = m.get("content")
                if isinstance(c, str):
                    text_blocks.append(c)
                elif isinstance(c, list):
                    for p in c:
                        if isinstance(p, dict) and p.get("type") == "text":
                            text_blocks.append(p.get("text", ""))
            prompt = "\n\n".join(x for x in text_blocks if x)
            resp = self._model.generate_content(prompt, generation_config=gen_cfg)

        self._last_response = resp
        return resp

    def extract_text(self, raw_response) -> str:
        # 1) try the convenience accessor
        try:
            if raw_response.text:
                return raw_response.text
        except Exception:
            pass

        # 2) aggregate parts
        try:
            chunks = []
            for cand in getattr(raw_response, "candidates", []) or []:
                cnt = getattr(cand, "content", None)
                for part in getattr(cnt, "parts", []) or []:
                    txt = getattr(part, "text", None)
                    if isinstance(txt, str) and txt:
                        chunks.append(txt)
            return "".join(chunks)
        except Exception:
            return ""

    def request_has_image(self, messages: List[Dict]) -> bool:
        for m in messages:
            for part in m.get("content", []):
                t = isinstance(part, dict) and part.get("type")
                if t in ("image", "image_url"):
                    return True
        return False

    def debug_dump(self, raw_response) -> dict:
        out = {
            "provider": "gemini",
            "model": self.model,
            "combined_text": "",
            "finish_reason": None,
            "has_candidates": False,
            "prompt_block_reason": None,
            "safety": [],
        }
        try:
            out["combined_text"] = self.extract_text(raw_response) or ""
            cands = getattr(raw_response, "candidates", None)
            out["has_candidates"] = bool(cands)
            if cands:
                fr = getattr(cands[0], "finish_reason", None)
                out["finish_reason"] = str(fr)
                sr = getattr(cands[0], "safety_ratings", None)
                if sr:
                    out["safety"] = [str(x) for x in sr]
            pf = getattr(raw_response, "prompt_feedback", None)
            if pf:
                out["prompt_block_reason"] = str(getattr(pf, "block_reason", None))
        except Exception as e:
            out["debug_error"] = str(e)
        return out


# =========================
# Claude Adapter
# =========================
class ClaudeAdapter(BaseLLMAdapter):
    """
    Uses anthropic.Anthropic messages API.
    - Images sent as base64 parts with media_type detection.
    - stop_sequences -> stop_sequences (as list)
    """
    def __init__(self, model: str, cache: bool = False, max_tokens: int = 3000):
        super().__init__(model, cache, max_tokens)
        import anthropic
        load_dotenv()
        key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key)
        self._anthropic = anthropic

    def build_user_content(self, init_canvas_b64: Optional[str], text: str):
        content: List[Dict] = []
        if init_canvas_b64:
            media_type = "image/jpeg" if init_canvas_b64[:3] == "/9j" else "image/png"
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": init_canvas_b64
                }
            })
        content.append({"type": "text", "text": text})
        if self.cache:
            content[-1]["cache_control"] = {"type": "ephemeral"}
        return content

    def call(self, system_message, messages, additional_args):
        args = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_message,
            messages=messages,
        )
        if "stop_sequences" in additional_args:
            ss = additional_args["stop_sequences"]
            args["stop_sequences"] = ss if isinstance(ss, list) else [ss]
        if "temperature" in additional_args:
            args["temperature"] = additional_args["temperature"]
        if "top_k" in additional_args:
            args["top_k"] = additional_args["top_k"]

        if self.cache:
            return self.client.beta.prompt_caching.messages.create(**args)
        return self.client.messages.create(**args)

    def extract_text(self, raw_response) -> str:
        return raw_response.content[0].text


# =========================
# OpenAI Adapter
# =========================
class OpenAIAdapter(BaseLLMAdapter):
    """
    Uses openai.chat.completions API.
    - Images must be sent as image_url (data URI) in message content.
    - stop_sequences -> stop (but we skip 'stop' if images are present).
    """
    def __init__(self, model: str, cache: bool = False, max_tokens: int = 3000):
        super().__init__(model, cache, max_tokens)
        import openai
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self._client = openai.OpenAI()
        
        # Use Responses API for reasoning models (GPT-5, o3, o4-mini etc.)
        m = (model or "").lower()
        self._use_responses_api = m.startswith(("gpt-5", "o3", "o4"))

    def build_user_content(self, init_canvas_b64: Optional[str], text: str):
        content: List[Dict] = []
        if init_canvas_b64:
            hdr = "data:image/jpeg;base64," if init_canvas_b64[:3] == "/9j" else "data:image/png;base64,"
            content.append({
                "type": "image_url",
                "image_url": {"url": hdr + init_canvas_b64}
            })
        content.append({"type": "text", "text": text})
        return content
    
    @staticmethod
    def get_text_from_response(resp):
        """
        Supports:
        - Chat Completions: resp.choices[0].message.content
        - Responses API:    resp.output_text  (or iterate blocks)
        - dicts:            best-effort extraction
        """
        # Responses API (SDK object)
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()

        # Chat Completions (SDK object)
        if hasattr(resp, "choices") and resp.choices:
            ch0 = resp.choices[0]
            if hasattr(ch0, "message") and getattr(ch0.message, "content", None):
                return ch0.message.content.strip()
            if getattr(ch0, "text", None):
                return ch0.text.strip()

        # Plain dicts
        if isinstance(resp, dict):
            if "choices" in resp and resp["choices"]:
                ch0 = resp["choices"][0]
                if isinstance(ch0, dict):
                    if "message" in ch0 and "content" in ch0["message"]:
                        return (ch0["message"]["content"] or "").strip()
                    if "text" in ch0:
                        return (ch0["text"] or "").strip()
            if "output_text" in resp and resp["output_text"]:
                return str(resp["output_text"]).strip()
            if "output" in resp and isinstance(resp["output"], list):
                for item in resp["output"]:
                    for block in item.get("content", []):
                        if block.get("type") in ("output_text", "text"):
                            t = block.get("text") or block.get("content")
                            if t:
                                return t.strip()
        return ""

    def response_metadata(self, raw_response) -> dict:
        try:
            rid = getattr(raw_response, "id", None) or (isinstance(raw_response, dict) and raw_response.get("id"))
            return {"response_id": rid}
        except Exception:
            return {}


    def call(self, system_message, messages, additional_args):
        add_args = additional_args or {}
        temperature = add_args.get("temperature")
        stop_sequences = add_args.get("stop_sequences")
        has_image = self.request_has_image(messages)

        # ---------------- Responses API (minimal + correct types) ----------------
        if self._use_responses_api:
            # Flatten all incoming chat turns into ONE user turn of input parts.
            # Only 'input_text' and 'input_image' are used (per your SDK error).
            def _norm_image(part: dict):
                if "image_data" in part and isinstance(part["image_data"], dict):
                    return {
                        "type": "input_image",
                        "image_data": {
                            "mime_type": part["image_data"].get("mime_type", "image/png"),
                            "data": part["image_data"].get("data", "")
                        }
                    }
                u = part.get("image_url")
                if isinstance(u, dict):
                    u = u.get("url")
                if not u:
                    u = part.get("url")
                if isinstance(u, str) and u.strip():
                    return {"type": "input_image", "image_url": u.strip()}
                return None

            parts = []
            for m in messages or []:
                c = m.get("content")
                if isinstance(c, str):
                    if c.strip():
                        parts.append({"type": "input_text", "text": c})
                elif isinstance(c, list):
                    for p in c:
                        if isinstance(p, str):
                            if p.strip():
                                parts.append({"type": "input_text", "text": p})
                            continue
                        if isinstance(p, dict):
                            pt = p.get("type")
                            if pt in ("text", "input_text"):
                                t = p.get("text", "")
                                if isinstance(t, str) and t.strip():
                                    parts.append({"type": "input_text", "text": t})
                            elif pt in ("image_url", "input_image"):
                                img = _norm_image(p)
                                if img:
                                    parts.append(img)

            if not parts:
                parts = [{"type": "input_text", "text": ""}]

            rargs = {
                "model": self.model,
                "input": [{"role": "user", "content": parts}],
                "max_output_tokens": self.max_tokens,
                "store": True, #keep reasoning items server-side between turns
            }
            
            
            prev_id = additional_args.get("previous_response_id")
            if prev_id:
                rargs["previous_response_id"] = prev_id
            
            if "reasoning_effort" in additional_args:
                rargs["reasoning"] = {"effort": additional_args["reasoning_effort"]}
            
            # <- minimal: attach your system prompt here
            if isinstance(system_message, str) and system_message.strip():
                rargs["instructions"] = system_message
            if temperature is not None:
                rargs["temperature"] = float(temperature)

            resp = self._client.responses.create(**rargs)
            self._last_response_id = getattr(resp, "id", None)
            return resp

        # ---------------- Chat Completions (minimal fallback) ----------------
        chat_messages = messages or []
        if isinstance(system_message, str) and system_message.strip():
            chat_messages = [{"role": "system", "content": system_message}] + chat_messages

        args = dict(
            model=self.model,
            messages=chat_messages,
            max_completion_tokens=self.max_tokens,
        )
        if temperature is not None:
            args["temperature"] = float(temperature)
        if stop_sequences and not has_image:
            args["stop"] = stop_sequences if isinstance(stop_sequences, list) else [stop_sequences]

        return self._client.chat.completions.create(**args)




    def extract_text(self, raw_response) -> str:
        """
        Robustly extract text from either:
        - Responses API objects (preferred for GPT-5 / o3 / o4)
        - Chat Completions objects (classic .choices[0].message.content)
        - A dict wrapper (defensive)
        """
        # If we were handed our old dict wrapper by some caller, unwrap it
        if isinstance(raw_response, dict):
            if "text" in raw_response and isinstance(raw_response["text"], str):
                return raw_response["text"]
            raw = raw_response.get("raw")
            if raw is not None:
                return OpenAIAdapter.get_text_from_response(raw)
            return ""

        # Normal path: use the helper that understands both schemas
        return OpenAIAdapter.get_text_from_response(raw_response)



# =========================
# Local Hugging Face Qwen3.5 Adapter
# =========================
class HuggingFaceQwenAdapter(BaseLLMAdapter):
    """
    Run Qwen3.5 locally using Hugging Face Transformers.

    Supports:
      - text prompts
      - base64 PNG/JPEG images
      - system messages
      - multi-turn message history

    No vLLM or API server is required.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-9B",
        cache: bool = False,
        max_tokens: int = 3000,
    ):
        super().__init__(model, cache, max_tokens)

        from transformers import AutoProcessor, AutoModelForMultimodalLM

        print(f"[HuggingFace] Loading processor: {model}", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            model,
            trust_remote_code=True,
        )

        print(f"[HuggingFace] Loading model: {model}", flush=True)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self._model.eval()

        try:
            self.device = next(self._model.parameters()).device
        except StopIteration:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        print(
            f"[HuggingFace] Qwen model loaded. Primary device: {self.device}",
            flush=True,
        )

    @staticmethod
    def _decode_base64_image(data: str) -> Image.Image:
        """
        Decode either:
          - plain base64
          - data:image/png;base64,...
        """
        if not data:
            raise ValueError("Empty base64 image")

        if data.startswith("data:"):
            _, data = data.split(",", 1)

        image_bytes = base64.b64decode(data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def build_user_content(
        self,
        init_canvas_b64: Optional[str],
        text: str,
    ):
        """
        Keep the same intermediate message structure used by SketchApp.
        Conversion to Hugging Face format happens inside call().
        """
        content: List[Dict] = []

        if init_canvas_b64:
            media_type = (
                "image/jpeg"
                if init_canvas_b64.startswith("/9j")
                else "image/png"
            )

            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": init_canvas_b64,
                },
            })

        content.append({
            "type": "text",
            "text": text,
        })

        return content

    def _convert_content(self, content):
        """
        Convert the app's provider-independent content into the message
        format expected by AutoProcessor.apply_chat_template().
        """
        if isinstance(content, str):
            return [{
                "type": "text",
                "text": content,
            }]

        converted = []

        for part in content or []:
            if isinstance(part, str):
                converted.append({
                    "type": "text",
                    "text": part,
                })
                continue

            if not isinstance(part, dict):
                continue

            part_type = part.get("type")

            if part_type == "text":
                converted.append({
                    "type": "text",
                    "text": part.get("text", ""),
                })

            elif part_type == "image":
                source = part.get("source", {})

                if source.get("type") == "base64":
                    image = self._decode_base64_image(
                        source.get("data", "")
                    )
                    converted.append({
                        "type": "image",
                        "image": image,
                    })

            elif part_type == "image_url":
                image_url = part.get("image_url", {})
                url = (
                    image_url.get("url", "")
                    if isinstance(image_url, dict)
                    else str(image_url)
                )

                if url.startswith("data:image/"):
                    image = self._decode_base64_image(url)
                    converted.append({
                        "type": "image",
                        "image": image,
                    })
                elif url:
                    converted.append({
                        "type": "image",
                        "url": url,
                    })

        return converted

    def _convert_messages(self, system_message, messages):
        converted_messages = []

        if isinstance(system_message, str) and system_message.strip():
            converted_messages.append({
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": system_message,
                }],
            })

        for message in messages or []:
            role = message.get("role", "user")

            # Hugging Face/Qwen uses assistant, not Gemini's "model".
            if role not in ("system", "user", "assistant"):
                role = "user"

            converted_messages.append({
                "role": role,
                "content": self._convert_content(
                    message.get("content", "")
                ),
            })

        return converted_messages

    @staticmethod
    def _move_inputs_to_device(inputs, device):
        """
        BatchFeature normally supports .to(device), but keep a fallback
        for compatibility with different Transformers versions.
        """
        try:
            return inputs.to(device)
        except Exception:
            output = {}

            for key, value in inputs.items():
                if hasattr(value, "to"):
                    output[key] = value.to(device)
                else:
                    output[key] = value

            return output

    def call(self, system_message, messages, additional_args):
        additional_args = additional_args or {}

        hf_messages = self._convert_messages(
            system_message,
            messages,
        )

        inputs = self.processor.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = self._move_inputs_to_device(
            inputs,
            self.device,
        )

        temperature = additional_args.get("temperature")
        top_k = additional_args.get("top_k")
        stop_sequences = additional_args.get("stop_sequences")

        generation_args = {
            "max_new_tokens": self.max_tokens,
            "use_cache": True,
        }

        # Deterministic mode in your main app sets temperature=0.
        if temperature is not None and float(temperature) <= 0:
            generation_args["do_sample"] = False
        else:
            generation_args["do_sample"] = True
            generation_args["temperature"] = (
                float(temperature)
                if temperature is not None
                else 0.7
            )
            generation_args["top_p"] = float(
                additional_args.get("top_p", 0.8)
            )

            if top_k is not None:
                generation_args["top_k"] = int(top_k)
            else:
                generation_args["top_k"] = 20

        pad_token_id = getattr(
            self.processor.tokenizer,
            "pad_token_id",
            None,
        )
        eos_token_id = getattr(
            self.processor.tokenizer,
            "eos_token_id",
            None,
        )

        if pad_token_id is not None:
            generation_args["pad_token_id"] = pad_token_id

        if eos_token_id is not None:
            generation_args["eos_token_id"] = eos_token_id

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                **generation_args,
            )

        input_length = inputs["input_ids"].shape[-1]
        new_token_ids = generated_ids[:, input_length:]

        texts = self.processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        output_text = texts[0] if texts else ""

        # Transformers generate() does not directly use arbitrary textual
        # stop sequences here, so trim after decoding.
        if stop_sequences:
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]

            earliest_position = None

            for stop_text in stop_sequences:
                if not stop_text:
                    continue

                position = output_text.find(stop_text)

                if position >= 0:
                    if (
                        earliest_position is None
                        or position < earliest_position
                    ):
                        earliest_position = position

            if earliest_position is not None:
                output_text = output_text[:earliest_position]

        return {
            "text": output_text.strip(),
            "model": self.model,
            "input_tokens": int(input_length),
            "output_tokens": int(new_token_ids.shape[-1]),
        }

    def extract_text(self, raw_response) -> str:
        if isinstance(raw_response, str):
            return raw_response.strip()

        if isinstance(raw_response, dict):
            return str(raw_response.get("text", "")).strip()

        return ""

    def response_metadata(self, raw_response) -> dict:
        if not isinstance(raw_response, dict):
            return {}

        return {
            "model": raw_response.get("model"),
            "input_tokens": raw_response.get("input_tokens"),
            "output_tokens": raw_response.get("output_tokens"),
        }

    def debug_dump(self, raw_response) -> dict:
        if not isinstance(raw_response, dict):
            return {
                "provider": "huggingface",
                "model": self.model,
                "repr": repr(raw_response)[:8000],
            }

        return {
            "provider": "huggingface",
            "model": self.model,
            "combined_text": raw_response.get("text", ""),
            "input_tokens": raw_response.get("input_tokens"),
            "output_tokens": raw_response.get("output_tokens"),
            "finish_reason": "generated",
        }



# =========================
# OpenRouter Adapter 
# =========================
class OpenRouterAdapter(BaseLLMAdapter):
    """
    Uses OpenRouter API 
    - Images sent as image_url (data URI) in message content.
    - Provider preference set to Alibaba only.
    - Supports Gemini image generation model with modalities.
    """
    def __init__(self, model: str, cache: bool = False, max_tokens: int = 3000):
        super().__init__(model, cache, max_tokens)
        import openai
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        self._client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self._is_image_gen_model = (model == "google/gemini-3-pro-image-preview")
        # Storage for two-turn image generation results
        self._last_image_gen_images = []
        self._last_image_gen_text = None
        self._last_image_gen_reasoning_turn1 = None  # Reasoning for image generation
        self._last_image_gen_reasoning_turn2 = None  # Reasoning for text answer

    def build_user_content(self, init_canvas_b64: Optional[str], text: str):
        content: List[Dict] = []
        if init_canvas_b64:
            hdr = "data:image/jpeg;base64," if init_canvas_b64[:3] == "/9j" else "data:image/png;base64,"
            content.append({
                "type": "image_url",
                "image_url": {"url": hdr + init_canvas_b64}
            })
        content.append({"type": "text", "text": text})
        return content

    def call(self, system_message, messages, additional_args):
        add_args = additional_args or {}
        temperature = add_args.get("temperature")
        stop_sequences = add_args.get("stop_sequences")

        # Build messages with system message
        chat_messages = list(messages or [])  # Make a copy

        # extra_body = {
        #     "provider": {
        #         "only": ["google-ai-studio"],      # enforce single provider
        #         "allow_fallbacks": False  # never fall back
        #     }
        # }

        extra_body = {"provider": {"allow_fallbacks": False}, "usage": {"include": True}}

        # Only lock to google-ai-studio when using Gemini models
        # if self.model.startswith("google/"):
        #     extra_body["provider"]["only"] = ["google-ai-studio"]
        # else:
        # extra_body["provider"]["only"] = ["parasail/fp8"]
        # extra_body["provider"]["only"] = ["deepinfra"]
        # extra_body["provider"]["only"] = ["moonshotai"]
        # extra_body["provider"]["only"] = ["fireworks"]
        extra_body["provider"]["only"] = ["fireworks/fast"]
        # extra_body["provider"]["only"] = ["wafer"]
        # extra_body["provider"]["only"] = ["modal"]
        # extra_body["provider"]["only"] = ["coreweave"]
        # extra_body["provider"]["only"] = ["modelrun"]
        extra_body["reasoning"] = {"effort": "high"}

        # Special handling for image generation model - two-turn approach
        # Only use it when explicitly requested via use_image_gen flag
        if self._is_image_gen_model and add_args.get("use_image_gen", False):
            return self._call_image_gen_two_turn(system_message, chat_messages, temperature)

        # Standard path for non-image models
        if isinstance(system_message, str) and system_message.strip():
            chat_messages = [{"role": "system", "content": system_message}] + chat_messages

        args = dict(
            model=self.model,
            messages=chat_messages,
            max_tokens=self.max_tokens,
            extra_body=extra_body
        )
        if temperature is not None:
            args["temperature"] = float(temperature)
        if stop_sequences:
            args["stop"] = stop_sequences if isinstance(stop_sequences, list) else [stop_sequences]

        reasoning_effort = add_args.get("reasoning_effort")
        if reasoning_effort:
            args["reasoning_effort"] = reasoning_effort  # OpenRouter expects this for GPT-5

        return self._client.chat.completions.create(**args)

    def _call_image_gen_two_turn(self, system_message, chat_messages, temperature):
        """
        Two-turn approach for Gemini image generation model:
        Turn 1: Generate image only (modalities: ["image"])
        Turn 2: Get text answer only (modalities: ["text"])

        Returns a combined response object with both image and text.
        """
        import copy

        # Clear any previous stored results
        self._last_image_gen_images = []
        self._last_image_gen_text = None
        self._last_image_gen_reasoning_turn1 = None
        self._last_image_gen_reasoning_turn2 = None

        def _extract_reasoning(resp):
            """Extract reasoning text from response message."""
            if not (hasattr(resp, 'choices') and resp.choices):
                return None
            msg = resp.choices[0].message
            # Check for reasoning attribute
            if hasattr(msg, 'reasoning') and msg.reasoning:
                return msg.reasoning
            # Check for reasoning_details
            if hasattr(msg, 'reasoning_details') and msg.reasoning_details:
                details = msg.reasoning_details
                if isinstance(details, list):
                    texts = []
                    for d in details:
                        if isinstance(d, dict) and d.get('type') == 'reasoning.text':
                            texts.append(d.get('text', ''))
                        elif hasattr(d, 'type') and d.type == 'reasoning.text':
                            texts.append(getattr(d, 'text', ''))
                    return '\n'.join(texts) if texts else None
                return str(details)
            return None

        extra_body_base = {
            "provider": {
                "only": ["google-ai-studio"],
                "allow_fallbacks": False
            }
        }

        # Prepare Turn 1 messages - add system message and image generation trigger
        turn1_messages = copy.deepcopy(chat_messages)
        if turn1_messages:
            first_msg = turn1_messages[0]
            content = first_msg.get("content", "")
            sys_prefix = f"{system_message}\n\n" if isinstance(system_message, str) and system_message.strip() else ""
            img_suffix = "\n\nGenerate the requested image."

            if isinstance(content, str):
                turn1_messages[0]["content"] = f"{sys_prefix}{content}{img_suffix}"
            elif isinstance(content, list):
                new_content = []
                if sys_prefix:
                    new_content.append({"type": "text", "text": system_message})
                new_content.extend(content)
                new_content.append({"type": "text", "text": img_suffix.strip()})
                turn1_messages[0] = {"role": first_msg.get("role", "user"), "content": new_content}

        # Turn 1: Image only
        extra_body_turn1 = {**extra_body_base, "modalities": ["image"]}
        args_turn1 = dict(
            model=self.model,
            messages=turn1_messages,
            max_tokens=self.max_tokens,
            extra_body=extra_body_turn1
        )
        if temperature is not None:
            args_turn1["temperature"] = float(temperature)

        print("[IMAGE GEN] Turn 1: Requesting image generation...")
        resp1 = self._client.chat.completions.create(**args_turn1)

        # Extract reasoning from Turn 1 (full)
        self._last_image_gen_reasoning_turn1 = _extract_reasoning(resp1)

        # Extract generated images from Turn 1
        generated_images = []
        if hasattr(resp1, 'choices') and resp1.choices:
            msg1 = resp1.choices[0].message
            if hasattr(msg1, 'images') and msg1.images:
                for img in msg1.images:
                    if isinstance(img, dict) and 'image_url' in img:
                        url_data = img['image_url']
                        if isinstance(url_data, dict) and 'url' in url_data:
                            generated_images.append(url_data['url'])
                        elif isinstance(url_data, str):
                            generated_images.append(url_data)

        print(f"[IMAGE GEN] Turn 1 complete: {len(generated_images)} image(s) generated")

        # Build Turn 2 messages - include the generated image in conversation
        turn2_messages = copy.deepcopy(turn1_messages)

        # Add assistant's response (the generated image)
        assistant_content = []
        if generated_images:
            for img_url in generated_images:
                assistant_content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url}
                })
        if not assistant_content:
            assistant_content = [{"type": "text", "text": "(image generated)"}]

        turn2_messages.append({
            "role": "assistant",
            "content": assistant_content
        })

        # Add follow-up question asking for final answer (generic - format is in original prompt)
        turn2_messages.append({
            "role": "user",
            "content": r"Now give your final answer in the format of  \"$\boxed{answer}$\."
        })

        # Turn 2: Text only
        extra_body_turn2 = {**extra_body_base, "modalities": ["text"]}
        args_turn2 = dict(
            model=self.model,
            messages=turn2_messages,
            max_tokens=self.max_tokens,
            extra_body=extra_body_turn2
        )
        if temperature is not None:
            args_turn2["temperature"] = float(temperature)

        print("[IMAGE GEN] Turn 2: Requesting text answer...")
        resp2 = self._client.chat.completions.create(**args_turn2)

        # Extract reasoning from Turn 2 (full)
        self._last_image_gen_reasoning_turn2 = _extract_reasoning(resp2)

        # Extract text from Turn 2
        text_answer = ""
        if hasattr(resp2, 'choices') and resp2.choices:
            msg2 = resp2.choices[0].message
            if isinstance(msg2.content, str):
                text_answer = msg2.content
            elif isinstance(msg2.content, list):
                for part in msg2.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_answer += part.get("text", "")
                    elif hasattr(part, "type") and part.type == "text":
                        text_answer += getattr(part, "text", "")

        print(f"[IMAGE GEN] Turn 2 complete: answer = {text_answer[:100]}...")

        # Store both responses for later extraction
        self._last_image_gen_images = generated_images
        self._last_image_gen_text = text_answer.strip()

        # Return Turn 2 response (text), but we'll use stored images in extract_images
        return resp2

    # ---------------------------------------------------------------
    # [KIMI-K2.6-EXPERIMENT] Diagnostics only.
    # Purely additive: does NOT change the request that is sent.
    # The base-class debug_dump() only stored repr(response)[:8000], which
    # truncates the reasoning stream and hides finish_reason for long
    # generations. We need the full picture to tell apart:
    #   (a) content=None because the model never emitted final content
    #   (b) content=None because generation was cut by a stop sequence
    #   (c) content=None because max_tokens was exhausted
    # ---------------------------------------------------------------
    @staticmethod
    def _openrouter_reasoning_text(msg):
        """Best-effort extraction of the provider-side reasoning stream."""
        if msg is None:
            return None
        r = getattr(msg, "reasoning", None)
        if isinstance(r, str) and r.strip():
            return r
        details = getattr(msg, "reasoning_details", None)
        if details:
            parts = []
            for d in details:
                if isinstance(d, dict):
                    t = d.get("text") or d.get("summary") or ""
                else:
                    t = getattr(d, "text", None) or getattr(d, "summary", None) or ""
                if t:
                    parts.append(t)
            if parts:
                return "\n".join(parts)
        return None

    def response_metadata(self, raw_response) -> dict:
        out = {
            "finish_reason": None,
            "native_finish_reason": None,
            "content_is_none": None,
            "content_len": None,
            "reasoning_len": None,
            "reasoning_tail": None,
            "reasoning_tokens": None,
            "provider": getattr(raw_response, "provider", None),
            "model_returned": getattr(raw_response, "model", None),
            "response_id": getattr(raw_response, "id", None),
        }
        try:
            choices = getattr(raw_response, "choices", None) or []
            if choices:
                ch = choices[0]
                out["finish_reason"] = getattr(ch, "finish_reason", None)
                out["native_finish_reason"] = getattr(ch, "native_finish_reason", None)
                msg = getattr(ch, "message", None)
                content = getattr(msg, "content", None)
                out["content_is_none"] = content is None
                out["content_len"] = len(content) if isinstance(content, str) else None
                reasoning = self._openrouter_reasoning_text(msg)
                if reasoning is not None:
                    out["reasoning_len"] = len(reasoning)
                    # the tail is where a stop-sequence cut becomes visible
                    out["reasoning_tail"] = reasoning[-800:]
                    out["reasoning_full"] = reasoning
            usage = getattr(raw_response, "usage", None)
            if usage is not None:
                ctd = getattr(usage, "completion_tokens_details", None)
                if ctd is not None:
                    out["reasoning_tokens"] = getattr(ctd, "reasoning_tokens", None)
        except Exception as e:
            out["metadata_error"] = str(e)
        return out

    def debug_dump(self, raw_response) -> dict:
        out = self.response_metadata(raw_response)
        try:
            out["repr"] = repr(raw_response)[:8000]
        except Exception:
            out["repr"] = None
        return out

    def extract_text(self, raw_response) -> str:
        """Extract text from OpenRouter response (same format as OpenAI)."""
        # For two-turn image gen, use stored text from Turn 2
        # Only if we actually have stored text (meaning image gen was used)
        if hasattr(self, '_last_image_gen_text') and self._last_image_gen_text is not None:
            text = self._last_image_gen_text
            # Clear the stored text after extraction
            self._last_image_gen_text = None
            return text

        if hasattr(raw_response, "choices") and raw_response.choices:
            msg = raw_response.choices[0].message
            content = msg.content

            if isinstance(content, str) and content:
                return content.strip()

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    # dict-style
                    if isinstance(part, dict):
                        pt = part.get("type")
                        if pt in ("text", "output_text"):
                            t = part.get("text") or part.get("content") or ""
                            if t.strip():
                                text_parts.append(t)
                        continue

                    # object-style (openai SDK models)
                    pt = getattr(part, "type", None)
                    if pt in ("text", "output_text"):
                        t = getattr(part, "text", None) or getattr(part, "content", None) or ""
                        if isinstance(t, str) and t.strip():
                            text_parts.append(t)

                if text_parts:
                    return "\n".join(text_parts).strip()

            # fallback some providers use
            if getattr(msg, "text", None):
                return msg.text.strip()

        return ""

    def extract_images(self, raw_response) -> List[str]:
        """
        Extract generated images from Gemini image model response.
        Returns list of base64 data URLs.

        For two-turn image gen, uses stored images from Turn 1.
        """
        # First check if we have stored images from two-turn approach
        if hasattr(self, '_last_image_gen_images') and self._last_image_gen_images:
            images = self._last_image_gen_images
            print(f"[IMAGE GEN] Returning {len(images)} stored image(s) from Turn 1")
            return images

        # Fallback: check response directly (for non-two-turn cases)
        images = []
        # Check response.images (OpenRouter format)
        if hasattr(raw_response, "images") and raw_response.images:
            for image in raw_response.images:
                if isinstance(image, dict) and "image_url" in image:
                    url = image["image_url"].get("url", "")
                    if url:
                        images.append(url)
        # Also check choices[0].message for images
        if hasattr(raw_response, "choices") and raw_response.choices:
            msg = raw_response.choices[0].message
            if hasattr(msg, "images") and msg.images:
                for image in msg.images:
                    if isinstance(image, dict) and "image_url" in image:
                        url = image["image_url"].get("url", "")
                        if url:
                            images.append(url)
        if images:
            print(f"[IMAGE GEN] Found {len(images)} generated images")
        return images


# # =========================
# # Adapter Factory
# # =========================
# def make_adapter(llm_name: str, model: str, cache: bool, max_tokens: int) -> BaseLLMAdapter:
#     """
#     Factory function to create the appropriate LLM adapter based on provider name.

#     Args:
#         llm_name: Provider name ('claude', 'gpt', 'gemini', 'openrouter', 'qwen3')
#         model: Model identifier
#         cache: Whether to enable caching
#         max_tokens: Maximum tokens for responses

#     Returns:
#         Configured LLM adapter instance

#     Raises:
#         ValueError: If unknown LLM provider is specified
#     """
#     name = (llm_name or "").lower()
#     if name in ("claude", "anthropic"):
#         return ClaudeAdapter(model=model, cache=cache, max_tokens=max_tokens)
#     if name in ("gpt", "openai"):
#         return OpenAIAdapter(model=model, cache=cache, max_tokens=max_tokens)
#     if name in ("gemini", "google"):
#         return GeminiAdapter(model=model, cache=cache, max_tokens=max_tokens)
#     if name in ("openrouter", "qwen3"):
#         return OpenRouterAdapter(model=model, cache=cache, max_tokens=max_tokens)
#     raise ValueError("Unknown --llm. Use 'claude', 'gpt', 'gemini', or 'qwen3'.")



def make_adapter(
    llm_name: str,
    model: str,
    cache: bool,
    max_tokens: int,
) -> BaseLLMAdapter:
    name = (llm_name or "").lower()

    if name in ("claude", "anthropic"):
        return ClaudeAdapter(
            model=model,
            cache=cache,
            max_tokens=max_tokens,
        )

    if name in ("gpt", "openai"):
        return OpenAIAdapter(
            model=model,
            cache=cache,
            max_tokens=max_tokens,
        )

    if name in ("gemini", "google"):
        return GeminiAdapter(
            model=model,
            cache=cache,
            max_tokens=max_tokens,
        )

    if name in ("huggingface", "hf", "qwen35"):
        return HuggingFaceQwenAdapter(
            model=model or "Qwen/Qwen3.5-9B",
            cache=cache,
            max_tokens=max_tokens,
        )

    # Keep the existing OpenRouter behavior separate.
    if name in ("openrouter", "qwen3"):
        return OpenRouterAdapter(
            model=model,
            cache=cache,
            max_tokens=max_tokens,
        )

    raise ValueError(
        "Unknown --llm. Use 'claude', 'gpt', 'gemini', "
        "'openrouter', 'qwen3', 'huggingface', or 'qwen35'."
    )
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
# OpenRouter Adapter (Qwen3)
# =========================
class OpenRouterAdapter(BaseLLMAdapter):
    """
    Uses OpenRouter API with Alibaba provider for Qwen3-VL models.
    - Images sent as image_url (data URI) in message content.
    - Provider preference set to Alibaba only.
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
        chat_messages = messages or []
        if isinstance(system_message, str) and system_message.strip():
            chat_messages = [{"role": "system", "content": system_message}] + chat_messages

        args = dict(
            model=self.model,
            messages=chat_messages,
            max_tokens=self.max_tokens,
            extra_body={
                "provider": {
                    "only": ["alibaba"],      # enforce single provider
                    "allow_fallbacks": False  # never fall back       
                }
            }
        )
        if temperature is not None:
            args["temperature"] = float(temperature)
        if stop_sequences:
            args["stop"] = stop_sequences if isinstance(stop_sequences, list) else [stop_sequences]

        return self._client.chat.completions.create(**args)

    def extract_text(self, raw_response) -> str:
        """Extract text from OpenRouter response (same format as OpenAI)."""
        if hasattr(raw_response, "choices") and raw_response.choices:
            return raw_response.choices[0].message.content.strip()
        return ""


# =========================
# Adapter Factory
# =========================
def make_adapter(llm_name: str, model: str, cache: bool, max_tokens: int) -> BaseLLMAdapter:
    """
    Factory function to create the appropriate LLM adapter based on provider name.

    Args:
        llm_name: Provider name ('claude', 'gpt', 'gemini', 'openrouter', 'qwen3')
        model: Model identifier
        cache: Whether to enable caching
        max_tokens: Maximum tokens for responses

    Returns:
        Configured LLM adapter instance

    Raises:
        ValueError: If unknown LLM provider is specified
    """
    name = (llm_name or "").lower()
    if name in ("claude", "anthropic"):
        return ClaudeAdapter(model=model, cache=cache, max_tokens=max_tokens)
    if name in ("gpt", "openai"):
        return OpenAIAdapter(model=model, cache=cache, max_tokens=max_tokens)
    if name in ("gemini", "google"):
        return GeminiAdapter(model=model, cache=cache, max_tokens=max_tokens)
    if name in ("openrouter", "qwen3"):
        return OpenRouterAdapter(model=model, cache=cache, max_tokens=max_tokens)
    raise ValueError("Unknown --llm. Use 'claude', 'gpt', 'gemini', or 'qwen3'.")

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for interacting with Gemini, Claude, OpenAI, Qwen APIs,
image processing, and PDF handling.
"""

import json
import asyncio
import base64
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any, Optional

import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import os

import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f) or {}

def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default

# Initialize clients lazily or with robust defaults
api_key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
if api_key:
    gemini_client = genai.Client(api_key=api_key)
    print("Initialized Gemini Client with API Key")
else:
    print("Warning: Could not initialize Gemini Client. Missing credentials.")
    gemini_client = None


anthropic_api_key = get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
if anthropic_api_key:
    anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
    print("Initialized Anthropic Client with API Key")
else:
    print("Warning: Could not initialize Anthropic Client. Missing credentials.")
    anthropic_client = None

openai_api_key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
if openai_api_key:
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    print("Initialized OpenAI Client with API Key")
else:
    print("Warning: Could not initialize OpenAI Client. Missing credentials.")
    openai_client = None

# Qwen / DashScope client (OpenAI-compatible endpoint)
dashscope_api_key = get_config_val("api_keys", "dashscope_api_key", "DASHSCOPE_API_KEY", "")
_openai_base_url = get_config_val("openai_base_url", None, "OPENAI_BASE_URL", "")
if dashscope_api_key:
    _qwen_base_url = _openai_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_client = AsyncOpenAI(api_key=dashscope_api_key, base_url=_qwen_base_url)
    print("Initialized Qwen/DashScope Client with API Key")
else:
    print("Warning: Could not initialize Qwen Client. Missing DASHSCOPE_API_KEY.")
    qwen_client = None



def _convert_to_gemini_parts(contents: List[Dict[str, Any]]) -> List[types.Part]:
    """
    Convert a generic content list to a list of Gemini's genai.types.Part objects.

    Supported image formats:
    - Standard: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
    - Legacy:   {"type": "image", "image_base64": "<base64>"}  (treated as image/jpeg)
    """
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
            elif item.get("image_base64"):
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(item["image_base64"]),
                        mime_type="image/jpeg",
                    )
                )
    return gemini_parts


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    ASYNC: Call Gemini API with asynchronous retry logic.
    """
    if gemini_client is None:
        raise RuntimeError(
            "Gemini client was not initialized: missing Google API key. "
            "Please set GOOGLE_API_KEY in environment, or configure api_keys.google_api_key in configs/model_config.yaml."
        )

    result_list = []
    target_candidate_count = config.candidate_count
    # Gemini API max candidate count is 8. We will call multiple times if needed.
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    for attempt in range(max_attempts):
        try:
            # Use global client
            client = gemini_client

            # Convert generic content list to Gemini's format right before the API call
            gemini_contents = _convert_to_gemini_parts(current_contents)
            response = await client.aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )

            # If we are using Image Generation models to generate images
            if (
                "nanoviz" in model_name
                or "image" in model_name
            ):
                raw_response_list = []
                if not response.candidates or not response.candidates[0].content.parts:
                    print(
                        f"[Warning]: Failed to generate image, retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                # In this mode, we can only have one candidate
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # Append base64 encoded image data to raw_response_list
                        raw_response_list.append(
                            base64.b64encode(part.inline_data.data).decode("utf-8")
                        )
                        break

            # Otherwise, for text generation models
            else:
                raw_response_list = [
                    part.text
                    for candidate in response.candidates
                    for part in candidate.content.parts
                ]
            result_list.extend([r for r in raw_response_list if r.strip() != ""])
            if len(result_list) >= target_candidate_count:
                result_list = result_list[:target_candidate_count]
                break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            
            # Exponential backoff (capped at 30s)
            current_delay = min(retry_delay * (2 ** attempt), 30)
            
            print(
                f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. Retrying in {current_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                result_list = ["Error"] * target_candidate_count

    if len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list

def _convert_to_claude_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list to Claude's API format.
    Currently, the formats are identical, so this acts as a pass-through
    for architectural consistency and future-proofing.

    Claude API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    """
    return contents


def _convert_to_openai_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list (Claude format) to OpenAI's API format.

    Supported image formats:
    - Standard: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
    - Legacy:   {"type": "image", "image_base64": "<base64>"}  (treated as image/jpeg)

    Claude format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]

    OpenAI format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ...
    ]
    """
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            elif item.get("image_base64"):
                data_url = f"data:image/jpeg;base64,{item['image_base64']}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
    return openai_contents


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Claude API with asynchronous retry logic.
    This version efficiently handles input size errors by validating and modifying
    the content list once before generating all candidates.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the Claude-specific format and perform an initial optimistic resize.
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    # Note that this check is required because Claude only has 128k / 256k context windows.
    # For Gemini series that support 1M, we do not need this step.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": valid_claude_contents}
                ],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list

async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI API with asynchronous retry logic.
    This follows the same pattern as Claude's implementation.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the OpenAI-specific format
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            # If we reach here, the input is valid.
            response_text_list.append(first_response.choices[0].message.content)
            is_input_valid = True
            break  # Exit the validation loop

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content)

    return response_text_list


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI Image Generation API (GPT-Image) with asynchronous retry logic.
    """
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")
    
    # Base parameters for all models
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    
    # Add GPT-Image specific parameters
    gen_params.update({
        "quality": quality,
        "background": background,
        "output_format": output_format,
    })

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.generate(**gen_params)
            
            # OpenAI images.generate returns a list of images in response.data
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(f"[Warning]: Failed to generate image via OpenAI, no data returned.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for OpenAI image generation model {model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_qwen_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Qwen (DashScope OpenAI-compatible) API with asynchronous retry logic.
    Follows the same pattern as call_openai_with_retry_async.
    """
    if qwen_client is None:
        raise RuntimeError(
            "Qwen client was not initialized: missing DashScope API key. "
            "Please set DASHSCOPE_API_KEY in environment, or configure "
            "api_keys.dashscope_api_key in configs/model_config.yaml."
        )

    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            qwen_contents = _convert_to_openai_format(contents)
            first_response = await qwen_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": qwen_contents},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            response_text_list.append(first_response.choices[0].message.content)
            is_input_valid = True
            break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    if not is_input_valid:
        context_msg = f" for {error_context}" if error_context else ""
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(f"Input validated. Now generating remaining {remaining_candidates} candidates...")
        valid_qwen_contents = _convert_to_openai_format(contents)
        tasks = [
            qwen_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_qwen_contents},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content)

    return response_text_list


async def call_wanxiang_image_generation_async(
    prompt: str,
    model_name: str = None,
    image_bytes: bytes = None,
    max_attempts: int = 5,
    retry_delay: int = 10,
    error_context: str = "",
) -> List[str]:
    """
    ASYNC: Call Tongyi Wanxiang (DashScope) text-to-image or image-editing API.

    Set ``image_bytes`` for image-editing mode (wanx2.1-imageedit), otherwise
    text-to-image mode is used.

    Returns a list with one base64-encoded PNG/JPEG string, or ["Error"] on failure.
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx is required for Wanxiang support. Install it with: pip install httpx")

    if not dashscope_api_key:
        raise RuntimeError(
            "DashScope API key not configured. Please set DASHSCOPE_API_KEY in environment "
            "or configure api_keys.dashscope_api_key in configs/model_config.yaml."
        )

    wanxiang_cfg = model_config.get("image_providers", {}).get("wanxiang", {})

    # Polling parameters
    _POLL_MAX_ATTEMPTS = 60
    _POLL_INTERVAL_SECONDS = 3

    if image_bytes:
        # Image-editing mode
        edit_model = wanxiang_cfg.get("edit_model", "wanx2.1-imageedit")
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation"
        payload = {
            "model": edit_model,
            "input": {
                "function": "description_edit",
                "prompt": prompt,
                "base_image_url": f"data:image/jpeg;base64,{img_b64}",
            },
        }
    else:
        # Text-to-image mode
        txt2img_model = model_name or wanxiang_cfg.get("model", "wanx2.1-t2i-plus")
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
        payload = {
            "model": txt2img_model,
            "input": {"prompt": prompt},
            "parameters": {"size": "1024*1024", "n": 1},
        }

    headers = {
        "Authorization": f"Bearer {dashscope_api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Submit task
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                result = resp.json()

                if result.get("code"):
                    raise RuntimeError(
                        f"Wanxiang API error {result.get('code')}: {result.get('message', 'Unknown error')}"
                    )

                task_id = result.get("output", {}).get("task_id")
                if not task_id:
                    raise RuntimeError(f"No task_id in Wanxiang response: {result}")

                # Poll for completion
                poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
                poll_headers = {"Authorization": f"Bearer {dashscope_api_key}"}
                for _ in range(_POLL_MAX_ATTEMPTS):
                    await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                    poll_resp = await client.get(poll_url, headers=poll_headers)
                    poll_resp.raise_for_status()
                    poll_result = poll_resp.json()
                    task_status = poll_result.get("output", {}).get("task_status")

                    if task_status == "SUCCEEDED":
                        results_list = poll_result.get("output", {}).get("results", [])
                        if results_list:
                            img_url = results_list[0].get("url")
                            if img_url:
                                img_resp = await client.get(img_url)
                                img_resp.raise_for_status()
                                img_b64 = base64.b64encode(img_resp.content).decode("utf-8")
                                return [img_b64]
                        raise RuntimeError("Wanxiang task succeeded but no image URL found.")

                    elif task_status == "FAILED":
                        raise RuntimeError(f"Wanxiang task failed: {poll_result}")

                raise RuntimeError("Wanxiang task timed out after polling.")

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for Wanxiang image generation failed{context_msg}: {e}."
                f" Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_sdxl_image_generation_async(
    prompt: str,
    image_bytes: bytes = None,
    width: int = 1024,
    height: int = 768,
    max_attempts: int = 5,
    retry_delay: int = 10,
    error_context: str = "",
) -> List[str]:
    """
    ASYNC: Call a self-hosted SDXL/Flux endpoint (e.g., Automatic1111 / ComfyUI).

    Set ``image_bytes`` for img2img (refinement) mode; omit for txt2img mode.
    Configure the endpoint via ``image_providers.sdxl.endpoint`` in model_config.yaml
    or the ``SDXL_ENDPOINT`` environment variable.

    Returns a list with one base64-encoded image string, or ["Error"] on failure.
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx is required for SDXL support. Install it with: pip install httpx")

    sdxl_cfg = model_config.get("image_providers", {}).get("sdxl", {})
    endpoint = sdxl_cfg.get("endpoint", "") or os.getenv("SDXL_ENDPOINT", "")

    if not endpoint:
        raise RuntimeError(
            "SDXL endpoint not configured. Please set image_providers.sdxl.endpoint in "
            "configs/model_config.yaml or the SDXL_ENDPOINT environment variable."
        )

    auth_header = sdxl_cfg.get("auth_header", "") or os.getenv("SDXL_AUTH_HEADER", "")
    headers = {"Content-Type": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header

    if image_bytes:
        path = sdxl_cfg.get("img2img_path", "/sdapi/v1/img2img")
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "prompt": prompt,
            "init_images": [img_b64],
            "denoising_strength": 0.7,
            "width": width,
            "height": height,
            "steps": 30,
        }
    else:
        path = sdxl_cfg.get("txt2img_path", "/sdapi/v1/txt2img")
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": 30,
        }

    url = f"{endpoint.rstrip('/')}{path}"

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                result = resp.json()
                images = result.get("images", [])
                if images:
                    return [images[0]]
                raise RuntimeError("No images returned in SDXL response.")

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for SDXL image generation failed{context_msg}: {e}."
                f" Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


def _aspect_ratio_to_wh(aspect_ratio: str, base_size: int = 1024) -> tuple:
    """Convert an aspect-ratio string like '16:9' to (width, height) integers."""
    try:
        w_str, h_str = aspect_ratio.split(":")
        w_ratio, h_ratio = float(w_str), float(h_str)
        if w_ratio >= h_ratio:
            width = base_size
            height = int(base_size * h_ratio / w_ratio)
        else:
            height = base_size
            width = int(base_size * w_ratio / h_ratio)
        # Round to multiples of 64 (common requirement for diffusion models)
        width = max(64, (width // 64) * 64)
        height = max(64, (height // 64) * 64)
        return width, height
    except Exception:
        return base_size, base_size


async def call_image_model_async(
    model_name: str,
    prompt: str,
    aspect_ratio: str = "1:1",
    image_bytes: Optional[bytes] = None,
    max_attempts: int = 5,
    retry_delay: int = 30,
    error_context: str = "",
) -> List[str]:
    """
    Route image generation to the appropriate backend based on ``model_name``.

    Supported backends (determined by substrings in ``model_name``):
    - ``"gemini"`` / ``"nanoviz"``  → Gemini image generation
    - ``"gpt-image"`` / ``"dall-e"`` → OpenAI image generation
    - ``"wanxiang"`` / ``"wanx"``    → Tongyi Wanxiang (DashScope)
    - ``"sdxl"`` / ``"flux"`` / ``"stable"`` → Self-hosted SDXL/Flux

    Set ``image_bytes`` for image-editing / refinement mode.
    Returns a list with one base64-encoded image string.
    """
    model_lower = model_name.lower()

    if "gemini" in model_lower or "nanoviz" in model_lower:
        content_list = [{"type": "text", "text": prompt}]
        if image_bytes:
            content_list.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                },
            })
        return await call_gemini_with_retry_async(
            model_name=model_name,
            contents=content_list,
            config=types.GenerateContentConfig(
                temperature=1.0,
                candidate_count=1,
                max_output_tokens=8192,
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size="1k",
                ),
            ),
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    elif "gpt-image" in model_lower or "dall-e" in model_lower:
        image_config = {
            "size": "1536x1024",
            "quality": "high",
            "background": "opaque",
            "output_format": "png",
        }
        return await call_openai_image_generation_with_retry_async(
            model_name=model_name,
            prompt=prompt,
            config=image_config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    elif "wanxiang" in model_lower or "wanx" in model_lower or model_lower.startswith("wanx"):
        return await call_wanxiang_image_generation_async(
            prompt=prompt,
            image_bytes=image_bytes,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    elif "sdxl" in model_lower or "flux" in model_lower or "stable" in model_lower:
        width, height = _aspect_ratio_to_wh(aspect_ratio)
        return await call_sdxl_image_generation_async(
            prompt=prompt,
            image_bytes=image_bytes,
            width=width,
            height=height,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    else:
        raise ValueError(
            f"Unsupported image model: '{model_name}'. "
            "Supported: gemini/nanoviz, gpt-image/dall-e, wanxiang/wanx, sdxl/flux/stable."
        )


async def call_llm_async(
    model_name: str,
    contents: List[Dict[str, Any]],
    system_prompt: str = "",
    temperature: float = 1.0,
    candidate_num: int = 1,
    max_output_tokens: int = 50000,
    max_attempts: int = 5,
    retry_delay: int = 5,
    error_context: str = "",
) -> List[str]:
    """
    Route LLM text/vision calls to the appropriate backend based on ``model_name``.

    Routing rules (first match wins):
    - ``"gemini"``              → Gemini API
    - ``"claude"`` / ``"anthropic"`` → Anthropic API
    - ``"qwen"``                → Qwen/DashScope OpenAI-compatible API
    - anything else             → OpenAI API

    Returns a list of response strings (length == ``candidate_num``).
    """
    model_lower = model_name.lower()

    if "gemini" in model_lower:
        return await call_gemini_with_retry_async(
            model_name=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                candidate_count=candidate_num,
                max_output_tokens=max_output_tokens,
            ),
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    elif "claude" in model_lower or "anthropic" in model_lower:
        return await call_claude_with_retry_async(
            model_name=model_name,
            contents=contents,
            config={
                "system_prompt": system_prompt,
                "temperature": temperature,
                "candidate_num": candidate_num,
                "max_output_tokens": max_output_tokens,
            },
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    elif "qwen" in model_lower:
        return await call_qwen_with_retry_async(
            model_name=model_name,
            contents=contents,
            config={
                "system_prompt": system_prompt,
                "temperature": temperature,
                "candidate_num": candidate_num,
                "max_completion_tokens": max_output_tokens,
            },
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    else:
        # Default: OpenAI-compatible
        return await call_openai_with_retry_async(
            model_name=model_name,
            contents=contents,
            config={
                "system_prompt": system_prompt,
                "temperature": temperature,
                "candidate_num": candidate_num,
                "max_completion_tokens": max_output_tokens,
            },
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

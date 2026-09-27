#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate offline tri-view sentiment pseudo-labels using GLM-4.6V (prompt v3).

Output JSONL fields:
  {
    "id": "1",
    "image_sentiment": "positive/neutral/negative",
    "text_sentiment": "positive/neutral/negative",
    "multimodal_sentiment": "positive/neutral/negative",
    "image_distribution": {"positive": 0.60, "neutral": 0.30, "negative": 0.10},
    "text_distribution": {"positive": 0.10, "neutral": 0.20, "negative": 0.70},
    "multimodal_distribution": {"positive": 0.15, "neutral": 0.20, "negative": 0.65}
  }

Design:
  Stage 1: image-only sentiment.
  Stage 2: text-only sentiment.
  Stage 3: joint image-text sentiment from raw multimodal semantics,
           with single-modality predictions used only as noisy auxiliary cues.

Prompt v3 focus:
  - Image: balance affective and neutral visual judgment.
  - Text: conservative treatment of weak/descriptive emotional language.
  - Multimodal: semantic-first joint reasoning rather than rule-based fusion.
  - Distribution: discourage overconfidence and fixed probability templates.

Important:
  - GLM-4.6V is used only offline.
  - SCRD training later only reads this JSONL file.
  - No explanation, no confidence, no relation field is written to output.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from zai import ZhipuAiClient
except ImportError:
    ZhipuAiClient = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


SENTIMENT_ORDER = ("positive", "neutral", "negative")
ALLOWED_SENTIMENT = set(SENTIMENT_ORDER)

DEFAULT_API_KEY_FILE = ""
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_MODEL = "glm-4.6v"


class QuotaExhaustedError(RuntimeError):
    pass


class RateLimitExceededError(RuntimeError):
    pass


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_api_key(api_key: Optional[str] = None, api_key_file: Optional[str] = None) -> str:
    if api_key:
        key = api_key.strip()
        if key:
            return key

    if api_key_file:
        path = Path(api_key_file)
        if not path.exists():
            raise FileNotFoundError(f"API key file not found: {api_key_file}")

        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            raise RuntimeError(f"API key file is empty: {api_key_file}")

        if raw.startswith("{"):
            obj = json.loads(raw)
            key = obj.get("ZHIPU_API_KEY") or obj.get("ZAI_API_KEY") or obj.get("api_key") or obj.get("key")
            if not key:
                raise RuntimeError(
                    f"JSON key file must contain ZHIPU_API_KEY/ZAI_API_KEY/api_key/key: {api_key_file}"
                )
            return str(key).strip()

        return raw.strip()

    env_key = os.getenv("ZAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
    if env_key:
        return env_key.strip()

    raise RuntimeError(
        "Missing API key. Provide --api_key_file, --api_key, or set ZAI_API_KEY/ZHIPU_API_KEY."
    )


def read_text_file(path: Path) -> str:
    for enc in ["utf-8", "utf-8-sig", "gb18030", "latin-1"]:
        try:
            return path.read_text(encoding=enc).strip()
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore").strip()


def encode_image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"

    with image_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def list_samples(data_dir: Path) -> List[Dict[str, str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    samples = []
    txt_files = sorted(data_dir.glob("*.txt"), key=lambda p: natural_key(p.stem))

    for txt_path in txt_files:
        sid = txt_path.stem

        image_path = None
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidate = data_dir / f"{sid}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"[WARN] Missing image for id={sid}, txt={txt_path}", file=sys.stderr)
            continue

        text = read_text_file(txt_path)

        samples.append(
            {
                "id": sid,
                "image_path": str(image_path),
                "text_path": str(txt_path),
                "text": text,
            }
        )

    return samples


def extract_json(raw: str) -> Dict[str, Any]:
    if raw is None:
        raise ValueError("Empty model response")

    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Cannot parse JSON: {raw[:500]}")


def normalize_sentiment(value: Any, default: str = "neutral") -> str:
    if not isinstance(value, str):
        return default
    value = value.strip().lower()
    if value in ALLOWED_SENTIMENT:
        return value
    return default


def fallback_distribution(sentiment: str, peak: float = 0.80) -> Dict[str, float]:
    sentiment = normalize_sentiment(sentiment, default="neutral")
    other = round((1.0 - peak) / 2.0, 6)
    dist = {label: other for label in SENTIMENT_ORDER}
    dist[sentiment] = round(peak, 6)
    return dist


def normalize_distribution(value: Any, sentiment: str) -> Dict[str, float]:
    if not isinstance(value, dict):
        return fallback_distribution(sentiment)

    values = []
    for label in SENTIMENT_ORDER:
        try:
            prob = float(value.get(label, 0.0))
        except (TypeError, ValueError):
            prob = 0.0
        values.append(max(prob, 0.0))

    total = sum(values)
    if total <= 0.0:
        return fallback_distribution(sentiment)

    normalized = [v / total for v in values]
    rounded = [round(v, 6) for v in normalized]
    diff = round(1.0 - sum(rounded), 6)
    max_idx = max(range(len(rounded)), key=lambda i: rounded[i])
    rounded[max_idx] = round(rounded[max_idx] + diff, 6)
    return dict(zip(SENTIMENT_ORDER, rounded))


def normalize_prediction(obj: Dict[str, Any], sentiment_field: str, distribution_field: str) -> Dict[str, Any]:
    sentiment = normalize_sentiment(obj.get(sentiment_field), default="neutral")
    distribution = normalize_distribution(obj.get(distribution_field), sentiment)
    sentiment = max(SENTIMENT_ORDER, key=lambda label: distribution[label])
    return {
        "sentiment": sentiment,
        "distribution": distribution,
    }


def is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    quota_markers = (
        "quota",
        "insufficient",
        "balance",
        "billing",
        "arrearage",
        "freeallocatedquotaexceeded",
        "allocated quota",
        "no enough",
        "not enough",
        "payment required",
    )
    return any(marker in text for marker in quota_markers)


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    rate_limit_markers = (
        "limit_requests",
        "rate-limit",
        "rate limit",
        "request limit",
        "too many requests",
        "throttl",
    )
    return any(marker in text for marker in rate_limit_markers)


def is_failure_fallback_row(obj: Dict[str, Any]) -> bool:
    fallback = fallback_distribution("neutral")
    sentiment_fields = (
        "image_sentiment",
        "text_sentiment",
        "multimodal_sentiment",
    )
    distribution_fields = (
        "image_distribution",
        "text_distribution",
        "multimodal_distribution",
    )

    if any(normalize_sentiment(obj.get(field), default="neutral") != "neutral" for field in sentiment_fields):
        return False

    for field in distribution_fields:
        if normalize_distribution(obj.get(field), "neutral") != fallback:
            return False

    return True


def build_image_prompt() -> str:
    """
    v3 image prompt:
    Keep the v2 correction against over-neutral prediction, but avoid
    inferring polarity merely from objects, colors, scenery, or activities.
    """
    return """
You are generating an external IMAGE-ONLY sentiment pseudo-label for a multimodal sentiment model.

Task:
Judge the sentiment conveyed by the image alone.

Output:
Choose exactly one label:
positive, neutral, negative.
Also provide image_distribution over exactly these three labels.
The distribution values must be numeric probabilities, sum to 1, and the highest-probability label must match image_sentiment.

The distribution is an approximate semantic uncertainty estimate, not a calibrated statistical probability.
Do not reuse a fixed probability template simply because the predicted label is the same.
Adjust the distribution according to the strength, ambiguity, and competing evidence in the current image.
Avoid over-confident probabilities. Use exact 0.00 or 1.00 only when the visual evidence is exceptionally clear and unambiguous.

Do not output uncertain.
Do not output explanation.
Do not output confidence, relation, or dominant_modality.
Return only valid JSON.

Important rules:
1. Use only visible image content.
2. Do NOT use any external caption or text outside the image.
3. If the image contains visible words, meme text, poster text, or screenshot text, they are part of the image and may be used.
4. This is a social-media sentiment task. The label should reflect the affective impression of the image, not just whether a facial expression is visible.
5. Do not default to neutral merely because there is no explicit facial expression. However, do not infer positive or negative sentiment from objects, colors, scenery, people, brands, or activities alone unless they create a clear affective impression.
6. Choose positive when the image visually conveys a favorable, pleasant, cute, joyful, celebratory, beautiful, humorous, affectionate, exciting, successful, or supportive impression.
7. Choose negative when the image visually conveys sadness, anger, fear, disgust, pain, distress, violence, accident, damage, illness, insult, complaint, threat, or negative meme/screenshot text.
8. Choose neutral when the image is mainly factual, documentary, ordinary, product-like, informational, weakly emotional, ambiguous, or has no clear positive or negative affective impression.
9. If the image contains both emotional and non-emotional cues, mixed cues, or weak affective evidence, keep the distribution moderate and retain meaningful probability mass on plausible alternatives.

Positive visual cues include:
- smiling, celebration, love, affection, cute animals/children, pleasant food, beautiful scenery when presented affectively, fashion/beauty praise, awards, success, party, enjoyable activity, positive visible text, uplifting meme.

Negative visual cues include:
- crying, anger, injury, disaster, conflict, unpleasant scene, stress, violence, broken objects, threatening content, negative visible text, complaint meme.

Neutral visual cues include:
- plain object, ordinary street/building/room, neutral screenshot, factual poster, product display without affect, informational content, unclear or weak emotion.

Output JSON:
{
  "image_sentiment": "positive/neutral/negative",
  "image_distribution": {
    "positive": 0.00,
    "neutral": 0.00,
    "negative": 0.00
  }
}
""".strip()

def build_text_prompt(text: str) -> str:
    """
    v3 text prompt:
    Preserve the v2 neutral safeguards and further reduce false polarity
    from sentiment-bearing words used as names, titles, slogans, or weak fragments.
    """
    return f"""
You are generating an external TEXT-ONLY sentiment pseudo-label for a multimodal sentiment model.

Task:
Judge the sentiment expressed by the text alone.

Text:
"{text}"

Output:
Choose exactly one label:
positive, neutral, negative.
Also provide text_distribution over exactly these three labels.
The distribution values must be numeric probabilities, sum to 1, and the highest-probability label must match text_sentiment.

The distribution is an approximate semantic uncertainty estimate, not a calibrated statistical probability.
Do not reuse a fixed probability template simply because the predicted label is the same.
Adjust the distribution according to the strength, ambiguity, and competing evidence in the current text.
Avoid over-confident probabilities. Use exact 0.00 or 1.00 only when the text explicitly and unambiguously expresses that sentiment.

Do not output uncertain.
Do not output explanation.
Do not output confidence, relation, or dominant_modality.
Return only valid JSON.

Important rules:
1. Use only the given text.
2. Do NOT use image information.
3. Positive or negative requires explicit subjective affect, emotional attitude, praise, complaint, evaluation, emotional emoji, affective hashtag, or clear sarcasm/irony.
4. Do NOT label text as positive only because it mentions a good object, celebrity, event, brand, product, place, activity, attractive topic, or sentiment-bearing word used as a proper name/title.
5. Do NOT label text as negative only because it mentions a serious topic unless the text expresses negative attitude, complaint, sadness, anger, fear, disgust, criticism, or a clearly negative hashtag.
6. If the text is mainly factual, descriptive, promotional, named entities, news-like, location/event/product description, title, slogan, short fragment, or emotionally weak/unclear, choose neutral.
7. Hashtags and emojis are important only when they clearly express sentiment in context.
8. If the text contains weak praise, weak complaint, slogans, names, topics, or ambiguous short fragments, prefer a moderate distribution with non-trivial neutral probability.
9. Sentiment should be directed toward an identifiable target, event, situation, or experience. A sentiment-bearing word without clear evaluative scope is not sufficient by itself.
10. If both a neutral interpretation and an emotional interpretation are plausible and the emotional evidence is weak, prefer neutral and reflect the ambiguity in the distribution.
11. For sarcasm or irony, infer polarity only when the text itself provides enough evidence; do not assume sarcasm merely because wording is unusual.

Positive text cues:
- love, like, happy, great, amazing, beautiful, thanks, proud, excited, enjoy, congratulations, support, positive emoji, positive hashtag when used evaluatively.

Negative text cues:
- hate, angry, sad, bad, terrible, disappointed, disgusting, pain, fear, complaint, criticism, insult, negative emoji, negative hashtag, sarcasm with clearly negative affect.

Neutral text cues:
- factual description, title, name, product/event/location information, plain statement, ambiguous short text, slogan without clear attitude, weak affect.

Output JSON:
{{
  "text_sentiment": "positive/neutral/negative",
  "text_distribution": {{
    "positive": 0.00,
    "neutral": 0.00,
    "negative": 0.00
  }}
}}
""".strip()

def build_multimodal_prompt(text: str, image_sentiment: str, text_sentiment: str) -> str:
    """
    v3 multimodal prompt:
    Judge raw image-text semantics first. The single-modality predictions are
    noisy auxiliary cues rather than deterministic fusion inputs.
    """
    return f"""
You are generating an external MULTIMODAL sentiment pseudo-label for a semi-supervised image-text sentiment model.

Task:
Judge the overall sentiment conveyed by the COMPLETE image-text social-media post.

You are given:
1. the original image,
2. the original text,
3. two noisy single-modality sentiment predictions.

Text:
"{text}"

Auxiliary single-modality predictions:
- image_sentiment = "{image_sentiment}"
- text_sentiment = "{text_sentiment}"

IMPORTANT:
The two single-modality predictions are only auxiliary cues and may be wrong.
Do NOT mechanically combine them.
Do NOT use fixed fusion rules such as:
- negative overrides positive,
- emotional overrides neutral,
- agreement automatically determines the multimodal label.

The multimodal sentiment must primarily be determined from the joint semantic meaning of the ORIGINAL image and ORIGINAL text.

Output:
Choose exactly one label:
positive, neutral, negative.
Also provide multimodal_distribution over exactly these three labels.
The distribution values must be numeric probabilities, sum to 1, and the highest-probability label must match multimodal_sentiment.

The distribution is an approximate semantic uncertainty estimate, not a calibrated statistical probability.
Do not reuse a fixed probability template simply because the predicted label is the same.
Adjust the distribution according to the strength, ambiguity, cross-modal relation, and competing evidence in the current sample.
Avoid over-confident probabilities. Use exact 0.00 or 1.00 only when the complete image-text meaning is exceptionally explicit and unambiguous.

Do not output conflict.
Do not output uncertain.
Do not output explanation.
Do not output confidence, relation, or dominant_modality.
Return only valid JSON.

Decision principles:
1. Interpret the original image and original text jointly before deciding the final sentiment.
2. Determine what role the text plays relative to the image. The text may describe, complement, explain, contradict, reinterpret, mock, criticize, praise, or be only weakly related to the image.
3. Judge the sentiment of the WHOLE POST as it would normally be understood by a viewer, rather than simply selecting one modality label.
4. Agreement between image_sentiment and text_sentiment is supporting evidence, but it does not guarantee the multimodal label if the raw image-text meaning suggests otherwise.
5. When one auxiliary prediction is neutral and the other is emotional, inspect the raw evidence carefully. Do not automatically follow the emotional prediction. A factual caption combined with an emotional-looking image, or vice versa, may still yield neutral or weak overall sentiment.
6. When image and text express opposite sentiments, determine whether one modality clearly supplies the intended attitude toward the other, whether the combination produces sarcasm/irony, or whether the two signals genuinely cancel or remain mixed. If neither polarity clearly dominates the semantic meaning of the whole post, choose neutral.
7. Positive requires that the overall post conveys a favorable, pleasant, approving, joyful, affectionate, humorous, celebratory, supportive, or otherwise clearly positive attitude.
8. Negative requires that the overall post conveys criticism, complaint, sadness, anger, fear, disgust, distress, hostility, disappointment, or another clearly negative attitude.
9. Choose neutral when the whole post is mainly factual, descriptive, informational, weakly emotional, ambiguous, mixed, weakly related across modalities, or lacks a clearly dominant positive or negative attitude.
10. Sarcasm and irony should be judged from the combined image-text meaning. Do not assume sarcasm merely because the modalities differ.
11. Do not infer hidden identity, private information, unstated events, or unsupported intentions.
12. Use common-sense knowledge only when it is directly supported by visible image content or explicit text.

Distribution rules:
13. The probability distribution should reflect the uncertainty of the current image-text meaning rather than a fixed label-to-probability template.
14. When image and text conflict, are weak, or are only partially related, keep meaningful probability mass on competing labels, especially neutral.
15. High confidence should be used only when the overall image-text meaning is explicit and unambiguous.

Output JSON:
{{
  "multimodal_sentiment": "positive/neutral/negative",
  "multimodal_distribution": {{
    "positive": 0.00,
    "neutral": 0.00,
    "negative": 0.00
  }}
}}
""".strip()

def chat_completion_text(
    client: ZhipuAiClient,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> str:
    # GLM-4.6V: disable thinking and sampling for deterministic offline evidence.
    # temperature is retained in the function/CLI for compatibility with the Qwen script,
    # but it is intentionally not sent; do_sample=False gives greedy deterministic decoding.
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        do_sample=False,
        thinking={"type": "disabled"},
        stream=False,
    )
    return completion.choices[0].message.content


def call_with_retries(
    fn,
    max_retries: int,
    retry_sleep: float,
    max_rate_limit_retries: int,
    rate_limit_sleep: float,
    default_value: Any,
    tag: str,
) -> Any:
    last_error = None
    rate_limit_count = 0
    attempt = 0

    while attempt < max_retries:
        try:
            return fn()
        except Exception as e:
            last_error = e
            if is_quota_error(e):
                raise QuotaExhaustedError(f"{tag}: {e}") from e
            if is_rate_limit_error(e):
                rate_limit_count += 1
                if rate_limit_count > max_rate_limit_retries:
                    raise RateLimitExceededError(f"{tag}: {e}") from e
                sleep_seconds = rate_limit_sleep * rate_limit_count
                print(
                    f"[RATE_LIMIT] {tag} hit request limit "
                    f"{rate_limit_count}/{max_rate_limit_retries}; sleep {sleep_seconds:.1f}s: {e}",
                    file=sys.stderr,
                )
                time.sleep(sleep_seconds)
                continue
            attempt += 1
            print(f"[WARN] {tag} attempt {attempt}/{max_retries} failed: {e}", file=sys.stderr)
            time.sleep(retry_sleep * attempt)

    print(
        f"[ERROR] {tag} failed after retries. Use default={default_value}. error={last_error}",
        file=sys.stderr,
    )
    return default_value


def predict_image_sentiment(
    client: ZhipuAiClient,
    model: str,
    image_path: str,
    max_retries: int,
    retry_sleep: float,
    max_rate_limit_retries: int,
    rate_limit_sleep: float,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    def _call():
        image_url = encode_image_to_data_url(Path(image_path))
        prompt = build_image_prompt()

        raw = chat_completion_text(
            client=client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict image sentiment pseudo-label annotator. Return valid JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        obj = extract_json(raw)
        return normalize_prediction(
            obj,
            sentiment_field="image_sentiment",
            distribution_field="image_distribution",
        )

    return call_with_retries(
        fn=_call,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        max_rate_limit_retries=max_rate_limit_retries,
        rate_limit_sleep=rate_limit_sleep,
        default_value={
            "sentiment": "neutral",
            "distribution": fallback_distribution("neutral"),
        },
        tag=f"image_sentiment image={image_path}",
    )


def predict_text_sentiment(
    client: ZhipuAiClient,
    model: str,
    text: str,
    max_retries: int,
    retry_sleep: float,
    max_rate_limit_retries: int,
    rate_limit_sleep: float,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    def _call():
        prompt = build_text_prompt(text)

        raw = chat_completion_text(
            client=client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict text sentiment pseudo-label annotator. Return valid JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        obj = extract_json(raw)
        return normalize_prediction(
            obj,
            sentiment_field="text_sentiment",
            distribution_field="text_distribution",
        )

    return call_with_retries(
        fn=_call,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        max_rate_limit_retries=max_rate_limit_retries,
        rate_limit_sleep=rate_limit_sleep,
        default_value={
            "sentiment": "neutral",
            "distribution": fallback_distribution("neutral"),
        },
        tag="text_sentiment",
    )


def derive_multimodal_fallback(image_sentiment: str, text_sentiment: str) -> str:
    image_sentiment = normalize_sentiment(image_sentiment, default="neutral")
    text_sentiment = normalize_sentiment(text_sentiment, default="neutral")

    if image_sentiment == text_sentiment:
        return image_sentiment

    if image_sentiment == "neutral" and text_sentiment != "neutral":
        return text_sentiment

    if text_sentiment == "neutral" and image_sentiment != "neutral":
        return image_sentiment

    # Opposite and no reliable model decision: conservative fallback.
    return "neutral"


def predict_multimodal_sentiment(
    client: ZhipuAiClient,
    model: str,
    image_path: str,
    text: str,
    image_sentiment: str,
    text_sentiment: str,
    max_retries: int,
    retry_sleep: float,
    max_rate_limit_retries: int,
    rate_limit_sleep: float,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    def _call():
        image_url = encode_image_to_data_url(Path(image_path))
        prompt = build_multimodal_prompt(
            text=text,
            image_sentiment=image_sentiment,
            text_sentiment=text_sentiment,
        )

        raw = chat_completion_text(
            client=client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict semantic-first multimodal sentiment pseudo-label annotator. Judge the raw image-text meaning jointly and return valid JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        obj = extract_json(raw)
        return normalize_prediction(
            obj,
            sentiment_field="multimodal_sentiment",
            distribution_field="multimodal_distribution",
        )

    fallback_sentiment = derive_multimodal_fallback(image_sentiment, text_sentiment)
    return call_with_retries(
        fn=_call,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        max_rate_limit_retries=max_rate_limit_retries,
        rate_limit_sleep=rate_limit_sleep,
        default_value={
            "sentiment": fallback_sentiment,
            "distribution": fallback_distribution(fallback_sentiment),
        },
        tag=f"multimodal_sentiment image={image_path}",
    )


def derive_relation(image_sentiment: str, text_sentiment: str) -> str:
    image_sentiment = normalize_sentiment(image_sentiment, default="neutral")
    text_sentiment = normalize_sentiment(text_sentiment, default="neutral")

    if image_sentiment == text_sentiment:
        return "same"
    if image_sentiment == "neutral" and text_sentiment != "neutral":
        return "image_neutral"
    if text_sentiment == "neutral" and image_sentiment != "neutral":
        return "text_neutral"
    return "opposite"


def load_done_ids(out_file: Path, ignore_failure_fallback_rows: bool = True) -> tuple:
    done = set()
    ignored_fallback = 0

    if not out_file.exists():
        return done, ignored_fallback

    with out_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    if ignore_failure_fallback_rows and is_failure_fallback_row(obj):
                        ignored_fallback += 1
                    else:
                        done.add(str(obj["id"]))
            except Exception:
                continue

    return done, ignored_fallback


def clean_failure_fallback_rows(out_file: Path) -> tuple:
    if not out_file.exists():
        return 0, 0, None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = out_file.with_suffix(out_file.suffix + f".bak_{timestamp}")
    tmp_path = out_file.with_suffix(out_file.suffix + ".tmp")

    kept = 0
    removed = 0
    backup_path.write_bytes(out_file.read_bytes())

    with out_file.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except Exception:
                fout.write(line)
                kept += 1
                continue

            if is_failure_fallback_row(obj):
                removed += 1
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    tmp_path.replace(out_file)
    return kept, removed, backup_path


def iter_progress(items, desc: str):
    if tqdm is not None:
        return tqdm(items, desc=desc)
    return items


def print_distribution(out_file: Path) -> None:
    from collections import Counter

    counters = {
        "image_sentiment": Counter(),
        "text_sentiment": Counter(),
        "multimodal_sentiment": Counter(),
        "derived_relation": Counter(),
    }

    total = 0

    if not out_file.exists():
        print(f"[WARN] out_file not found: {out_file}")
        return

    with out_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            total += 1
            img = obj.get("image_sentiment")
            txt = obj.get("text_sentiment")
            mm = obj.get("multimodal_sentiment")

            counters["image_sentiment"][str(img)] += 1
            counters["text_sentiment"][str(txt)] += 1
            counters["multimodal_sentiment"][str(mm)] += 1
            counters["derived_relation"][derive_relation(str(img), str(txt))] += 1

    print("\n========== Distribution ==========")
    print("total:", total)
    for k, v in counters.items():
        print(k, dict(v))
    print("==================================\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)

    parser.add_argument("--api_key_file", type=str, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)

    parser.add_argument("--limit", type=int, default=0, help="0 means all samples")
    parser.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between samples")
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--rate_limit_sleep", type=float, default=30.0)
    parser.add_argument("--max_rate_limit_retries", type=int, default=120)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep_fallback_done",
        action="store_true",
        help="When resuming, keep previous all-neutral fallback rows as completed instead of regenerating them.",
    )
    parser.add_argument(
        "--clean_fallback_rows",
        action="store_true",
        help="Before resuming, back up the output JSONL and remove previous all-neutral fallback rows.",
    )
    parser.add_argument("--print_distribution", action="store_true")

    args = parser.parse_args()

    if args.resume and args.overwrite:
        raise RuntimeError("Do not use --resume and --overwrite together.")

    if ZhipuAiClient is None:
        raise RuntimeError("Missing dependency: install the zai-sdk package before calling GLM-4.6V API: pip install zai-sdk")

    api_key = load_api_key(api_key=args.api_key, api_key_file=args.api_key_file)

    data_dir = Path(args.data_dir)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and out_file.exists():
        out_file.unlink()
        print(f"[INFO] Removed existing output file: {out_file}")

    samples = list_samples(data_dir)

    if args.limit > 0:
        samples = samples[: args.limit]

    cleaned_fallback = 0
    clean_backup_path = None
    if args.resume and args.clean_fallback_rows:
        _, cleaned_fallback, clean_backup_path = clean_failure_fallback_rows(out_file)

    ignored_fallback = 0
    if args.resume:
        done_ids, ignored_fallback = load_done_ids(
            out_file,
            ignore_failure_fallback_rows=not args.keep_fallback_done,
        )
    else:
        done_ids = set()

    client = ZhipuAiClient(
        api_key=api_key,
        base_url=args.base_url,
    )

    print("[INFO] GLM-4.6V tri-view sentiment pseudo-label generation v3")
    print(f"[INFO] data_dir = {data_dir}")
    print(f"[INFO] out_file = {out_file}")
    print(f"[INFO] model = {args.model}")
    print(f"[INFO] base_url = {args.base_url}")
    print(f"[INFO] samples = {len(samples)}")
    print(f"[INFO] resume = {args.resume}")
    print(f"[INFO] done_ids = {len(done_ids)}")
    print(f"[INFO] cleaned_fallback_rows = {cleaned_fallback}")
    print(f"[INFO] clean_backup_path = {clean_backup_path}")
    print(f"[INFO] ignored_fallback_rows = {ignored_fallback}")
    print(f"[INFO] sleep = {args.sleep}")
    print(f"[INFO] rate_limit_sleep = {args.rate_limit_sleep}")
    print(f"[INFO] max_rate_limit_retries = {args.max_rate_limit_retries}")
    print(
        "[INFO] output fields = id, image_sentiment, text_sentiment, multimodal_sentiment, "
        "image_distribution, text_distribution, multimodal_distribution"
    )

    written = 0
    skipped = 0
    stopped_by_quota = False
    stopped_by_rate_limit = False

    with out_file.open("a", encoding="utf-8") as fout:
        for sample in iter_progress(samples, desc="Generating tri-view sentiment labels v2"):
            sid = str(sample["id"])

            if sid in done_ids:
                skipped += 1
                continue

            try:
                image_result = predict_image_sentiment(
                    client=client,
                    model=args.model,
                    image_path=sample["image_path"],
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    max_rate_limit_retries=args.max_rate_limit_retries,
                    rate_limit_sleep=args.rate_limit_sleep,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                image_sentiment = image_result["sentiment"]
                image_distribution = image_result["distribution"]

                text_result = predict_text_sentiment(
                    client=client,
                    model=args.model,
                    text=sample["text"],
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    max_rate_limit_retries=args.max_rate_limit_retries,
                    rate_limit_sleep=args.rate_limit_sleep,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                text_sentiment = text_result["sentiment"]
                text_distribution = text_result["distribution"]

                multimodal_result = predict_multimodal_sentiment(
                    client=client,
                    model=args.model,
                    image_path=sample["image_path"],
                    text=sample["text"],
                    image_sentiment=image_sentiment,
                    text_sentiment=text_sentiment,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    max_rate_limit_retries=args.max_rate_limit_retries,
                    rate_limit_sleep=args.rate_limit_sleep,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                multimodal_sentiment = multimodal_result["sentiment"]
                multimodal_distribution = multimodal_result["distribution"]
            except QuotaExhaustedError as e:
                stopped_by_quota = True
                print(f"[STOP] Quota/balance exhausted at id={sid}. No fallback row was written.", file=sys.stderr)
                print(f"[STOP] {e}", file=sys.stderr)
                break
            except RateLimitExceededError as e:
                stopped_by_rate_limit = True
                print(f"[STOP] Request rate limit persisted at id={sid}. No fallback row was written.", file=sys.stderr)
                print(f"[STOP] {e}", file=sys.stderr)
                break

            row = {
                "id": sid,
                "image_sentiment": image_sentiment,
                "text_sentiment": text_sentiment,
                "multimodal_sentiment": multimodal_sentiment,
                "image_distribution": image_distribution,
                "text_distribution": text_distribution,
                "multimodal_distribution": multimodal_distribution,
            }

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            written += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"[DONE] written = {written}")
    print(f"[DONE] skipped = {skipped}")
    print(f"[DONE] output = {out_file}")
    print(f"[DONE] stopped_by_quota = {stopped_by_quota}")
    print(f"[DONE] stopped_by_rate_limit = {stopped_by_rate_limit}")

    if args.print_distribution:
        print_distribution(out_file)

    if stopped_by_quota or stopped_by_rate_limit:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

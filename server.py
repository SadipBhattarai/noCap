from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel
import json
import math
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

app = FastAPI()

MODEL_NAME = os.environ.get("NOCAP_WHISPER_MODEL", "large-v3")


def load_gemini_api_key():
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key

    local_config = os.path.join(os.path.dirname(__file__), ".env.local")
    if not os.path.isfile(local_config):
        return None

    with open(local_config, "r", encoding="utf-8") as config_file:
        for line in config_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "GEMINI_API_KEY":
                return value.strip().strip("\"'")
    return None


GEMINI_API_KEY = load_gemini_api_key()
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODELS = (
    GEMINI_MODEL,
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
)

if GEMINI_API_KEY:
    print(f"Gemini API configured; using {GEMINI_MODEL} for transcription.")
    model = None
else:
    print(f"Loading faster-whisper model ({MODEL_NAME})...")
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8", cpu_threads=8)
    print("Faster-Whisper model loaded successfully!")


class GeminiAPIError(RuntimeError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


TERM_MAP = [
    (re.compile(r"^(diplomas?|डिप्लोमा)$", re.I), "Diploma"),
    (re.compile(r"^(bachelors?|ब्याचलर|बैचलर)$", re.I), "Bachelor"),
    (re.compile(r"^(masters?|मास्टर)$", re.I), "Master"),
    (re.compile(r"^(malta|माल्टा)$", re.I), "Malta"),
    (re.compile(r"^(ctevt|ctvt|ctbt)$", re.I), "CTEVT"),
    (re.compile(r"^rightway$", re.I), "Right Way"),
    (re.compile(r"^(ielts|ilts|ayles|ailes|eylts|आईइएलटीएस|आईएलटीएस)$", re.I), "IELTS"),
    (re.compile(r"^(pte|ptee|पीटीई)$", re.I), "PTE"),
    (re.compile(r"^(apply|aply|एप्लाई|एप्लाइ)$", re.I), "apply"),
    (re.compile(r"^(study|स्टडी|स्टडी)$", re.I), "study"),
    (re.compile(r"^(journey|जर्नी|जरनी)$", re.I), "journey"),
    (re.compile(r"^(course|कोर्स)$", re.I), "course"),
]

TRANSCRIPTION_PROMPT = (
    "बर्दिबास, Caregiver Training, CTEVT, Right Way, three months, "
    "practical training, real skills, apply, डोको नाम्लो किसान घर, विद्यार्थी, भर्ना।"
)

GEMINI_TRANSCRIPTION_PROMPT = """
कार्य: अडियोमा बोलेका शब्द जस्ताको तस्तै उतार्ने, अनुवाद गर्ने होइन।

यो मुख्यतः नेपाली भाषामा बोलेको अडियो हो, जसमा बीचबीचमा केही English शब्द र
वाक्यांश छन्। नेपालीमा बोलिएका शब्द नेपाली देवनागरी लिपिमै लेख्नुहोस्।
वक्ताले वास्तवमै English मा बोलेका शब्द वा वाक्यांश मात्र English अक्षरमा
लेख्नुहोस्। नेपाली वाक्यलाई English मा अनुवाद नगर्नुहोस्। English शब्दलाई
नेपाली लिपिमा नलेख्नुहोस्। अर्थ मिलाउन आफ्नै शब्द नथप्नुहोस् र नसुनिएको
शब्द अनुमान गरेर नलेख्नुहोस्।

उदाहरणको ढाँचा मात्र: “हामीले Caregiver Training सुरु गरेका छौं।”
यदि “Caregiver Training” English मै बोलिएको छ भने Latin अक्षरमै राख्नुहोस्।
Bardibas, Caregiver Training, CTEVT, Right Way, practical training,
real skills, apply जस्ता शब्द अडियोमा सुनिएमा तिनको यही spelling प्रयोग गर्नुहोस्।
व्यवसायको नाम “डोको नाम्लो किसान घर” हो। त्यो नाम सुनिएको सन्दर्भमा
“डोको नाम्लो” नै लेख्नुहोस्, “ढोका नजिक” भनेर अर्थ नफेर्नुहोस्।

JSON मा "segments" array फर्काउनुहोस्। प्रत्येक segment मा एउटा मात्र बोलेको
शब्द, त्यस शब्द सुरु भएको वास्तविक सेकेन्डमा numeric "start", र सकिएको
वास्तविक सेकेन्डमा numeric "end" राख्नुहोस्। शब्दहरू नछुटाउनुहोस् वा दोहोर्‍याउनुहोस्।
सामान्य विरामचिह्न शब्दसँगै राख्नुहोस्। सबै शब्दको समय छुट्टाछुट्टै निर्धारण
गर्नुहोस्; एउटै वाक्य वा लामो खण्डलाई एउटै segment नबनाउनुहोस्। सबै segment मा
साझा समय-offset नथप्नुहोस्।
सबै caption English मा नलेख्नुहोस्।
Summary, translation वा commentary नदिनुहोस्।
""".strip()

GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "text": {"type": "string"},
                },
                "required": ["start", "end", "text"],
            },
        },
    },
    "required": ["segments"],
}

GEMINI_NEPALI_RENDER_PROMPT = """
The following is an English transcript of audio that was spoken primarily in
Nepali. Translate its meaning into natural, grammatically correct Nepali written
in Devanagari. This output is for Nepali subtitles, not English subtitles.

Keep only these spoken English names and technical terms in Latin script when
they occur: Bardibas, Caregiver Training, CTEVT, Right Way, practical training,
real skills, apply. Translate all other English sentence wording into Nepali.
Do not add facts or omit meaning. Keep each segment's id and return exactly one
translated segment per input segment. Output only the requested JSON.
""".strip()

GEMINI_NEPALI_RENDER_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
            },
        },
    },
    "required": ["segments"],
}

PHRASE_FIXES = [
    (re.compile(r"\bCaregivers Training\b", re.I), "Caregiver Training"),
    (re.compile(r"\bCTV[TB]\b", re.I), "CTEVT"),
    (re.compile(r"(दाजुभाइ|दाजुभाई)(बीचको|बिचको)"), r"\1 \2"),
    (re.compile(r"आजे"), "आजै"),
    (re.compile(r"साते"), "साथै"),
    (re.compile(r"योटक"), "एउटा"),
    (re.compile(r"बिसेश"), "विशेष"),
    (re.compile(r"हजुळहरू"), "हजुरहरू"),
    (re.compile(r"एक्पलाय"), "apply"),
    (re.compile(r"गढना"), "गर्न"),
    (re.compile(r"सक्नो\s+उन्छा"), "सक्नुहुन्छ"),
    (re.compile(r"मालटा\s+में"), "Malta मा"),
    (re.compile(r"सम्बर्क"), "सम्पर्क"),
    (re.compile(r"गार्णु\s+होस"), "गर्नुहोस्"),
    (re.compile(r"\bAyles\b", re.I), "IELTS"),
    (re.compile(r"\bIelts\b", re.I), "IELTS"),
    (re.compile(r"\bP\s*\.?\s*T\s*\.?\s*E\.?\b", re.I), "PTE"),
    (re.compile(r"\bright\s+way\b", re.I), "Right Way"),
    (re.compile(r"आफने|अपने"), "आफ्नो"),
    (re.compile(r"\bआप\b"), "आफ्नो"),
    (re.compile(r"चाहनुहुन्चं"), "चाहनुहुन्छ"),
    (re.compile(r"गर्न सक्नुहुन्चं"), "गर्न सक्नुहुन्छ"),
    (re.compile(r"गरिहाल्नुहोस(?=$|[\s।,.!?])"), "गरिहाल्नुहोस्"),
    (
        re.compile(
            r"गर्नुहोस(?=$|[\s।,.!?])|गरनुस(?=$|[\s।,.!?])|गर्नुस(?=$|[\s।,.!?])"
        ),
        "गर्नुहोस्",
    ),
    (re.compile(r"समपर(?:क)?\s*को"), "सम्पर्क"),
    (re.compile(r"इसले|यसले"), "त्यसैले"),
    (re.compile(r"आब"), "अब"),
    (re.compile(r"आज(?:रोरु|रोरु|रोरू)|अजरोरु"), "यदि हजुरहरू"),
    (re.compile(r"(?:आल्स|अल्स|ऐल्स)(?=$|[\s।,.!?])"), "IELTS"),
    (re.compile(r"माल्टा|मार्ल्टा|माल्ता"), "Malta"),
    (re.compile(r"(?:पनी|पनि)(?=$|[\s।,.!?])"), "पनि"),
    (re.compile(r"(?:गरना|गरन)\s*सक्नुहुन्छ|गरना\s*सक्नूर्छा"), "गर्न सक्नुहुन्छ"),
    (re.compile(r"(?:तेसेले|त्यसले|तेसैले)"), "त्यसैले"),
    (re.compile(r"(?:एदि|यदि)\s+(?:आ|आउ?)\s*(?:जररो|जर्नी)?"), "यदि हजुरहरू"),
    (re.compile(r"(?:आम्नो|आफ्नो)\s+study\b"), "आफ्नो study"),
]

PUNCH_WORDS = {"malta", "apply", "pte", "ielts"}
TRAILING_PUNCT = "।,!?."

PAUSE_SPLIT_SEC = 0.16
MAX_CUE_SEC = 2.4
MAX_CUE_WORDS = 5
MAX_CUE_CHARS = 36
MAX_WORD_DURATION_SEC = 4.0


class CaptionRequest(BaseModel):
    file_path: str


def open_gemini_request(request):
    retryable_statuses = (429, 500, 502, 503, 504)
    for attempt in range(3):
        try:
            return urllib.request.urlopen(request, timeout=180)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            message = f"Gemini API returned HTTP {error.code}: {detail[:1000]}"
            if error.code not in retryable_statuses or attempt == 2:
                raise GeminiAPIError(message, error.code) from error
            print(f"{message}; retrying in {attempt + 1}s.")
        except urllib.error.URLError as error:
            message = f"Could not reach the Gemini API: {error.reason}"
            if attempt == 2:
                raise GeminiAPIError(message) from error
            print(f"{message}; retrying in {attempt + 1}s.")
        time.sleep(attempt + 1)


def gemini_request(url, method, payload=None, headers=None):
    request_headers = {"x-goog-api-key": load_gemini_api_key() or ""}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
        method=method,
    )
    with open_gemini_request(request) as response:
        return response.read(), response.headers


def interaction_output_text(interaction):
    output_text = interaction.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    for step in reversed(interaction.get("steps", [])):
        if step.get("type") != "model_output":
            continue
        text = "".join(
            part.get("text", "")
            for part in step.get("content", [])
            if part.get("type") == "text" and isinstance(part.get("text"), str)
        )
        if text.strip():
            return text
    return None


def gemini_interaction(payload):
    last_error = None
    for model_name in GEMINI_FALLBACK_MODELS:
        request_payload = {**payload, "model": model_name}
        try:
            response_bytes, _ = gemini_request(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                "POST",
                request_payload,
                {"Api-Revision": "2026-05-20"},
            )
            interaction = json.loads(response_bytes.decode("utf-8"))
            if not interaction_output_text(interaction):
                raise GeminiAPIError(
                    f"{model_name} completed without returning caption text."
                )
            if model_name != GEMINI_MODEL:
                print(f"Gemini fallback succeeded with {model_name}.")
            return interaction
        except (GeminiAPIError, json.JSONDecodeError) as error:
            last_error = error
            print(f"Gemini model {model_name} failed: {error}")

    if last_error is not None:
        raise GeminiAPIError(
            "All configured Gemini models failed. Last error: "
            f"{last_error}",
            getattr(last_error, "status_code", None),
        ) from last_error
    raise GeminiAPIError("No Gemini models are configured.")


def devanagari_and_latin_counts(captions):
    text = " ".join(caption["text"] for caption in captions)
    devanagari = len(re.findall(r"[\u0900-\u097f]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return devanagari, latin


def render_captions_in_nepali(captions):
    devanagari, latin = devanagari_and_latin_counts(captions)
    if devanagari and devanagari > latin:
        return captions
    if not load_gemini_api_key():
        raise GeminiAPIError(
            "The recognizer returned English text, but no Gemini API key is "
            "configured to translate it into Nepali. Add GEMINI_API_KEY to "
            "the project's .env.local file; .env.local.example is only a "
            "template. A valid key is required because local Whisper cannot "
            "translate English captions."
        )

    input_segments = [
        {"id": index, "text": caption["text"]}
        for index, caption in enumerate(captions)
    ]
    interaction = gemini_interaction(
        {
            "input": [
                {
                    "type": "text",
                    "text": (
                        GEMINI_NEPALI_RENDER_PROMPT
                        + "\n\nTranscript segments:\n"
                        + json.dumps(input_segments, ensure_ascii=False)
                    ),
                }
            ],
            "response_format": GEMINI_NEPALI_RENDER_SCHEMA,
        }
    )
    output_text = interaction_output_text(interaction)
    if not isinstance(output_text, str) or not output_text.strip():
        raise GeminiAPIError("Gemini returned no Nepali caption rendering.")

    translated = json.loads(output_text).get("segments")
    if not isinstance(translated, list) or len(translated) != len(captions):
        raise GeminiAPIError("Gemini returned an incomplete Nepali caption rendering.")

    translated_by_id = {}
    for segment in translated:
        segment_id = segment.get("id")
        text = segment.get("text")
        if (
            not isinstance(segment_id, int)
            or segment_id < 0
            or segment_id >= len(captions)
            or segment_id in translated_by_id
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise GeminiAPIError("Gemini returned invalid Nepali caption segments.")
        translated_by_id[segment_id] = text.strip()

    if len(translated_by_id) != len(captions):
        raise GeminiAPIError("Gemini omitted Nepali caption segments.")

    nepali_captions = [
        {
            **caption,
            "text": translated_by_id[index],
        }
        for index, caption in enumerate(captions)
    ]
    devanagari, latin = devanagari_and_latin_counts(nepali_captions)
    if not devanagari or devanagari <= latin:
        raise GeminiAPIError(
            "Gemini did not return predominantly Nepali captions; refusing to "
            "import English subtitles."
        )
    return nepali_captions


def transcribe_with_gemini(audio_path: str) -> list:
    if not load_gemini_api_key():
        raise RuntimeError("Set GEMINI_API_KEY to use online transcription.")

    audio_size = os.path.getsize(audio_path)
    _, upload_headers = gemini_request(
        "https://generativelanguage.googleapis.com/upload/v1beta/files",
        "POST",
        {"file": {"display_name": os.path.basename(audio_path)}},
        {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(audio_size),
            "X-Goog-Upload-Header-Content-Type": "audio/wav",
        },
    )
    upload_url = upload_headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise RuntimeError("Gemini did not return an audio upload URL.")

    with open(audio_path, "rb") as audio_file:
        upload_request = urllib.request.Request(
            upload_url,
            data=audio_file.read(),
            headers={
                "Content-Length": str(audio_size),
                "Content-Type": "audio/wav",
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            method="POST",
        )
        with open_gemini_request(upload_request) as response:
            uploaded_bytes = response.read()

    uploaded_file = json.loads(uploaded_bytes.decode("utf-8"))
    file_info = uploaded_file.get("file", {})
    file_uri = file_info.get("uri")
    file_name = file_info.get("name")
    if not file_uri or not file_name:
        raise RuntimeError("Gemini returned incomplete audio upload metadata.")

    try:
        interaction = gemini_interaction(
            {
                "input": [
                    {"type": "audio", "uri": file_uri, "mime_type": "audio/wav"},
                    {"type": "text", "text": GEMINI_TRANSCRIPTION_PROMPT},
                ],
                "response_format": GEMINI_RESPONSE_SCHEMA,
            }
        )
        output_text = interaction_output_text(interaction)
        if not isinstance(output_text, str) or not output_text.strip():
            raise RuntimeError("Gemini returned no transcript.")

        transcription = json.loads(output_text)
        segments = transcription.get("segments")
        if not isinstance(segments, list) or not segments:
            raise RuntimeError("Gemini returned no timestamped word segments.")

        words = []
        for segment in segments:
            start = segment.get("start")
            end = segment.get("end")
            text = segment.get("text")
            if (
                not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end <= start
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise RuntimeError("Gemini returned an invalid word segment.")

            tokens = text.split()
            duration_per_word = (float(end) - float(start)) / len(tokens)
            for index, token in enumerate(tokens):
                words.append({
                    "word": token,
                    "start": float(start) + index * duration_per_word,
                    "end": (
                        float(end)
                        if index == len(tokens) - 1
                        else float(start) + (index + 1) * duration_per_word
                    ),
                })

        words.sort(key=lambda word: word["start"])
        captions = pack_cues(words)
        if not captions:
            raise RuntimeError("Gemini transcription produced no caption cues.")
        return render_captions_in_nepali(captions)
    finally:
        try:
            gemini_request(
                f"https://generativelanguage.googleapis.com/v1beta/{file_name}",
                "DELETE",
            )
        except RuntimeError as error:
            print(f"Gemini temporary-file cleanup failed: {error}")


def format_timestamp(seconds: float) -> str:
    if seconds is None or seconds < 0:
        seconds = 0
    ms = int(round(float(seconds) * 1000))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def captions_to_srt(captions: list) -> str:
    blocks = []
    for cap in captions:
        start = float(cap["start"])
        end = max(float(cap["end"]), start + 0.08)
        text = " ".join(str(cap["text"]).replace("\r", " ").replace("\n", " ").split())
        if not text:
            continue
        blocks.append(
            f"{len(blocks) + 1}\r\n{format_timestamp(start)} --> {format_timestamp(end)}\r\n{text}"
        )
    return "\r\n\r\n".join(blocks) + "\r\n"


def write_srt(captions: list) -> str:
    srt_body = captions_to_srt(captions)
    if not srt_body.strip():
        raise ValueError("No caption text to write into SRT.")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        prefix="nocap-captions-",
        suffix=".srt",
        delete=False,
    ) as f:
        f.write(srt_body)
        srt_path = f.name
    if os.path.getsize(srt_path) < 10:
        os.remove(srt_path)
        raise ValueError("Wrote an empty SRT file.")
    return srt_path


def split_keep_punct(token: str) -> list:
    token = token.strip()
    return [token] if token else []


def normalize_token(token: str) -> str:
    raw = token.strip()
    if not raw:
        return ""
    punct = ""
    core = raw
    while core and core[-1] in TRAILING_PUNCT:
        punct = core[-1] + punct
        core = core[:-1]
    if not core:
        return ""
    for pattern, replacement in TERM_MAP:
        if pattern.match(core):
            return replacement + punct
    return core + punct


def is_latin(token: str) -> bool:
    return bool(re.search(r"[A-Za-z]", token))


def is_punch(token: str) -> bool:
    core = re.sub(r"[^\w]", "", token).lower()
    return core in PUNCH_WORDS


def merge_phrases(words: list) -> list:
    merged = []
    i = 0
    while i < len(words):
        cur = words[i]
        nxt = words[i + 1] if i + 1 < len(words) else None
        current_word = cur["word"].rstrip(TRAILING_PUNCT)
        next_word = nxt["word"].rstrip(TRAILING_PUNCT) if nxt else ""
        is_right_way = (
            current_word.lower() == "right"
            and next_word.lower().startswith("way")
        )
        is_doko_namlo = current_word == "डोको" and next_word == "नाम्लो"
        if nxt and (is_right_way or is_doko_namlo):
            punct = "".join(ch for ch in nxt["word"] if ch in TRAILING_PUNCT)
            merged.append({
                "word": ("Right Way" if is_right_way else "डोको नाम्लो") + punct,
                "start": cur["start"],
                "end": nxt["end"],
            })
            i += 2
            continue
        merged.append(cur)
        i += 1
    return merged


def collect_words(segments) -> list:
    words = []
    fallback_segments = []
    for segment in segments:
        fallback_segments.append(segment)
        if not segment.words:
            continue
        for w in segment.words:
            for part in split_keep_punct(w.word):
                token = normalize_token(part)
                if not token:
                    continue
                start = float(w.start)
                end = float(w.end)
                if end - start > MAX_WORD_DURATION_SEC:
                    raise RuntimeError(
                        f"Transcription returned an invalid {end - start:.1f}-second "
                        f"word ({token!r}); no captions were imported."
                    )
                words.append({
                    "word": token,
                    "start": start,
                    "end": max(end, start + 0.06),
                })

    if not words:
        for segment in fallback_segments:
            parts = split_keep_punct(segment.text)
            if not parts:
                continue
            dur = max(segment.end - segment.start, 0.1)
            step = dur / len(parts)
            for idx, part in enumerate(parts):
                token = normalize_token(part)
                if not token:
                    continue
                start = segment.start + idx * step
                words.append({
                    "word": token,
                    "start": start,
                    "end": start + step,
                })
    return merge_phrases(words)


def apply_phrase_fixes(text: str) -> str:
    fixed = text
    for pattern, replacement in PHRASE_FIXES:
        fixed = pattern.sub(replacement, fixed)
    fixed = re.sub(r"्{2,}", "्", fixed)
    return " ".join(fixed.split())


def should_start_new_cue(chunk: list, nxt: dict, gap: float) -> bool:
    if not chunk:
        return False
    if gap >= PAUSE_SPLIT_SEC:
        return True
    duration = nxt["end"] - chunk[0]["start"]
    if duration >= MAX_CUE_SEC:
        return True
    preview = " ".join([w["word"] for w in chunk] + [nxt["word"]])
    if len(preview.split()) > MAX_CUE_WORDS:
        return True
    if len(preview) > MAX_CUE_CHARS:
        return True
    last = chunk[-1]["word"]
    if last.endswith(("।", ",", "?", "!")):
        return True
    if is_punch(nxt["word"]) and (len(chunk) >= 2 or gap >= 0.08):
        return True
    if is_punch(last) and not is_punch(nxt["word"]) and gap >= 0.05:
        return True
    if is_latin(last) and not is_latin(nxt["word"]) and gap >= 0.1:
        return True
    return False


def pack_cues(words: list) -> list:
    words = merge_phrases(correct_known_brand(words))
    chunks = []
    current = []
    for w in words:
        if current:
            gap = w["start"] - current[-1]["end"]
            if should_start_new_cue(current, w, gap):
                chunks.append(current)
                current = []
        current.append(w)
    if current:
        chunks.append(current)

    index = 0
    while index < len(chunks) - 1:
        previous, following = chunks[index:index + 2]
        gap = following[0]["start"] - previous[-1]["end"]
        if (
            len(" ".join(word["word"] for word in previous).split())
            == MAX_CUE_WORDS
            and len(" ".join(word["word"] for word in following).split()) == 1
            and gap < PAUSE_SPLIT_SEC
            and not previous[-1]["word"].endswith(("।", ",", "?", "!"))
        ):
            combined = previous + following
            split_at = len(combined) // 2
            chunks[index:index + 2] = [
                combined[:split_at],
                combined[split_at:],
            ]
            index += 2
        else:
            index += 1

    captions = []
    for chunk in chunks:
        text = apply_phrase_fixes(" ".join(w["word"] for w in chunk))
        if not text:
            continue
        captions.append({
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "text": text,
        })
    return captions


def correct_known_brand(words: list) -> list:
    corrected = [dict(word) for word in words]
    for index in range(len(corrected) - 3):
        first = corrected[index]["word"]
        second = corrected[index + 1]["word"]
        third = corrected[index + 2]["word"]
        fourth = corrected[index + 3]["word"]
        first_core = first.rstrip(TRAILING_PUNCT)
        second_core = second.rstrip(TRAILING_PUNCT)
        third_core = third.rstrip(TRAILING_PUNCT)
        fourth_core = fourth.rstrip(TRAILING_PUNCT)
        if (
            first_core in {"ढोका", "डोका"}
            and second_core == "नजिक"
            and third_core == "किसान"
            and fourth_core.startswith("घर")
        ):
            first_punctuation = first[len(first_core):]
            second_punctuation = second[len(second_core):]
            corrected[index]["word"] = "डोको" + first_punctuation
            corrected[index + 1]["word"] = "नाम्लो" + second_punctuation
    return corrected


def extract_audio(src_path: str, audio_path: str) -> None:
    extract_cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
        "-c:a", "pcm_s16le",
        audio_path,
    ]
    extract = subprocess.run(extract_cmd, capture_output=True, text=True)
    if (
        extract.returncode != 0
        or not os.path.isfile(audio_path)
        or os.path.getsize(audio_path) == 0
    ):
        if os.path.exists(audio_path):
            os.remove(audio_path)
        fallback = [
            "ffmpeg", "-y", "-i", src_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            audio_path,
        ]
        extract = subprocess.run(fallback, capture_output=True, text=True)
        if (
            extract.returncode != 0
            or not os.path.isfile(audio_path)
            or os.path.getsize(audio_path) == 0
        ):
            err = (extract.stderr or extract.stdout or "ffmpeg failed").strip()
            raise RuntimeError(f"Audio extract failed: {err[-800:]}")


def transcribe_with_local_whisper(audio_path):
    global model
    if model is None:
        print(f"Loading local faster-whisper model ({MODEL_NAME}) as fallback...")
        model = WhisperModel(
            MODEL_NAME,
            device="cpu",
            compute_type="int8",
            cpu_threads=8,
        )

    segments, _ = model.transcribe(
        audio_path,
        language="ne",
        multilingual=True,
        task="transcribe",
        beam_size=5,
        best_of=5,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=220,
            speech_pad_ms=120,
        ),
        condition_on_previous_text=True,
        word_timestamps=True,
        hallucination_silence_threshold=0.35,
        no_repeat_ngram_size=3,
        repetition_penalty=1.05,
        initial_prompt=TRANSCRIPTION_PROMPT,
    )
    all_words = collect_words(segments)
    if not all_words:
        raise RuntimeError("Transcription produced no words.")
    return render_captions_in_nepali(pack_cues(all_words))


@app.post("/generate-captions")
def generate_captions(req: CaptionRequest):
    if not os.path.isfile(req.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_path = audio_file.name

    try:
        print(f"Extracting clean audio from: {req.file_path}")
        extract_audio(req.file_path, audio_path)

        provider = "local-whisper"
        if load_gemini_api_key():
            print(f"Transcribing with {GEMINI_MODEL}...")
            try:
                caption_list = transcribe_with_gemini(audio_path)
                provider = "gemini"
            except GeminiAPIError as error:
                if error.status_code not in (429, 500, 502, 503, 504):
                    raise
                print(
                    f"Gemini unavailable ({error}); falling back to local Whisper."
                )
                caption_list = transcribe_with_local_whisper(audio_path)
                provider = "local-whisper-fallback"
        else:
            print("Transcribing with local faster-whisper...")
            caption_list = transcribe_with_local_whisper(audio_path)
        if not caption_list:
            raise RuntimeError("Could not pack caption cues.")

        srt_path = write_srt(caption_list)
        print(
            f"{len(caption_list)} caption cues -> {srt_path}"
        )

        return {
            "status": "success",
            "srt_path": srt_path,
            "captions": caption_list,
            "transcription_provider": provider,
        }

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(audio_path)
        except FileNotFoundError:
            pass

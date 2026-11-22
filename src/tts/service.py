"""ElevenLabs text-to-speech integration."""

from __future__ import annotations

import os
import base64
import binascii

import httpx

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"


class TTSConfigurationError(Exception):
    """Raised when the server's TTS configuration is incomplete."""


class TTSError(Exception):
    """Raised for safe, user-presentable TTS provider failures."""


async def generate_speech(text: str) -> tuple[bytes, dict[str, object]]:
    """Generate MPEG audio and character timing alignment for subtitle playback."""

    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
    if not api_key:
        raise TTSConfigurationError("Speech is not configured: the API key is missing.")
    if not voice_id:
        raise TTSConfigurationError("Speech is not configured: the voice ID is missing.")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                ELEVENLABS_TTS_URL.format(voice_id=voice_id) + "/with-timestamps",
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": api_key, "Accept": "audio/mpeg"},
                json={"text": text, "model_id": ELEVENLABS_MODEL_ID},
            )
    except httpx.TimeoutException as exc:
        raise TTSError("Speech generation timed out. Please try again.") from exc
    except httpx.HTTPError as exc:
        raise TTSError("Could not connect to the speech service. Please try again.") from exc

    if response.is_error:
        # Provider response bodies are deliberately not forwarded or logged; they may
        # contain sensitive or implementation-specific information.
        if response.status_code in (401, 403, 404):
            raise TTSError("Speech service configuration was rejected. Check the server settings.")
        if response.status_code == 429:
            raise TTSError("Speech service is busy. Please try again shortly.")
        raise TTSError("Speech could not be generated. Please try again.")

    try:
        result = response.json()
        audio = base64.b64decode(result["audio_base64"], validate=True)
        alignment = result.get("normalized_alignment") or result.get("alignment")
    except (ValueError, KeyError, TypeError, binascii.Error) as exc:
        raise TTSError("Speech service returned an invalid audio response.") from exc

    is_mp3_frame = len(audio) > 1 and audio[0] == 0xFF and audio[1] & 0xE0 == 0xE0
    if not audio or not (audio.startswith(b"ID3") or is_mp3_frame) or not isinstance(alignment, dict):
        raise TTSError("Speech service returned an invalid audio response.")
    if not all(
        isinstance(alignment.get(key), list)
        for key in ("characters", "character_start_times_seconds", "character_end_times_seconds")
    ):
        raise TTSError("Speech service returned invalid subtitle timing data.")
    return audio, alignment

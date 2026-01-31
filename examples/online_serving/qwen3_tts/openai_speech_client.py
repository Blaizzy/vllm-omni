"""OpenAI-compatible client for Qwen3-TTS via /v1/audio/speech endpoint.

This script demonstrates how to use the OpenAI-compatible speech API
to generate audio from text using Qwen3-TTS models.

Examples:
    # CustomVoice task (predefined speaker)
    python openai_speech_client.py --text "Hello, how are you?" --voice Vivian

    # CustomVoice with emotion instruction
    python openai_speech_client.py --text "I'm so happy!" --voice Vivian \
        --instructions "Speak with excitement"

    # VoiceDesign task (voice from description)
    python openai_speech_client.py --text "Hello world" \
        --task-type VoiceDesign \
        --instructions "A warm, friendly female voice"

    # Base task (voice cloning)
    python openai_speech_client.py --text "Hello world" \
        --task-type Base \
        --ref-audio "https://example.com/reference.wav" \
        --ref-text "This is the reference transcript"

    # Batch mode (multiple texts at once)
    python openai_speech_client.py --batch \
        --texts "Hello world" "How are you?" "Goodbye!" \
        --voice Vivian --output-dir ./outputs
"""

import argparse
import base64
import os

import httpx

# Default server configuration
DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_API_KEY = "EMPTY"


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Detect MIME type from extension
    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    else:
        mime_type = "audio/wav"  # Default

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def run_tts_generation(args) -> None:
    """Run TTS generation via OpenAI-compatible /v1/audio/speech API."""

    # Build request payload
    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": args.response_format,
    }

    # Add optional parameters
    if args.instructions:
        payload["instructions"] = args.instructions
    if args.task_type:
        payload["task_type"] = args.task_type
    if args.language:
        payload["language"] = args.language
    if args.max_new_tokens:
        payload["max_new_tokens"] = args.max_new_tokens

    # Voice clone parameters (Base task)
    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://")):
            payload["ref_audio"] = args.ref_audio
        else:
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text
    if args.x_vector_only:
        payload["x_vector_only_mode"] = True

    print(f"Model: {args.model}")
    print(f"Task type: {args.task_type or 'CustomVoice'}")
    print(f"Text: {args.text}")
    print(f"Voice: {args.voice}")
    print("Generating audio...")

    # Make the API call
    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    with httpx.Client(timeout=300.0) as client:
        response = client.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    # Save audio response
    output_path = args.output or "tts_output.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def run_batch_tts_generation(args) -> None:
    """Run batch TTS generation via /v1/audio/speech/batch API."""

    # Build batch request payload
    requests = []
    for i, text in enumerate(args.texts):
        item = {
            "custom_id": f"request_{i}",
            "input": text,
            "voice": args.voice,
            "response_format": args.response_format,
        }

        # Add optional parameters
        if args.instructions:
            item["instructions"] = args.instructions
        if args.task_type:
            item["task_type"] = args.task_type
        if args.language:
            item["language"] = args.language
        if args.max_new_tokens:
            item["max_new_tokens"] = args.max_new_tokens

        # Voice clone parameters (Base task)
        if args.ref_audio:
            if args.ref_audio.startswith(("http://", "https://")):
                item["ref_audio"] = args.ref_audio
            else:
                item["ref_audio"] = encode_audio_to_base64(args.ref_audio)
        if args.ref_text:
            item["ref_text"] = args.ref_text
        if args.x_vector_only:
            item["x_vector_only_mode"] = True

        requests.append(item)

    payload = {
        "model": args.model,
        "requests": requests,
    }

    print(f"Model: {args.model}")
    print(f"Task type: {args.task_type or 'CustomVoice'}")
    print(f"Batch size: {len(args.texts)} texts")
    print(f"Voice: {args.voice}")
    print("Generating audio batch...")

    # Make the API call
    api_url = f"{args.api_base}/v1/audio/speech/batch"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    with httpx.Client(timeout=600.0) as client:
        response = client.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    # Parse JSON response
    batch_response = response.json()

    # Create output directory
    output_dir = args.output_dir or "."
    os.makedirs(output_dir, exist_ok=True)

    # Save each audio result
    success_count = 0
    error_count = 0
    for result in batch_response.get("results", []):
        custom_id = result.get("custom_id", "unknown")
        if result.get("error"):
            print(f"  {custom_id}: Error - {result['error']}")
            error_count += 1
        elif result.get("audio_base64"):
            # Decode and save audio
            audio_data = base64.b64decode(result["audio_base64"])
            output_path = os.path.join(output_dir, f"{custom_id}.{args.response_format}")
            with open(output_path, "wb") as f:
                f.write(audio_data)
            print(f"  {custom_id}: Saved to {output_path}")
            success_count += 1
        else:
            print(f"  {custom_id}: No audio data")
            error_count += 1

    print(f"\nBatch complete: {success_count} succeeded, {error_count} failed")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible client for Qwen3-TTS via /v1/audio/speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Server configuration
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"API base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="API key (default: EMPTY)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Model name/path",
    )

    # Task configuration
    parser.add_argument(
        "--task-type",
        "-t",
        type=str,
        default=None,
        choices=["CustomVoice", "VoiceDesign", "Base"],
        help="TTS task type (default: CustomVoice)",
    )

    # Batch mode
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch mode for processing multiple texts at once",
    )

    # Input text (single mode)
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to synthesize (single mode)",
    )

    # Input texts (batch mode)
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        default=None,
        help="Multiple texts to synthesize (batch mode)",
    )

    # Voice/speaker
    parser.add_argument(
        "--voice",
        type=str,
        default="Vivian",
        help="Speaker/voice name (default: Vivian). Options: Vivian, Ryan, etc.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language: Auto, Chinese, English, etc.",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Voice style/emotion instructions",
    )

    # Base (voice clone) parameters
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio file path or URL for voice cloning (Base task)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Reference audio transcript for voice cloning (Base task)",
    )
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use x-vector only mode for voice cloning (no ICL)",
    )

    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate",
    )

    # Output
    parser.add_argument(
        "--response-format",
        type=str,
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio output format (default: wav)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output audio file path (single mode, default: tts_output.wav)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for batch mode (default: current directory)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.batch:
        # Batch mode
        if not args.texts:
            print("Error: --texts is required for batch mode")
            exit(1)
        run_batch_tts_generation(args)
    else:
        # Single mode
        if not args.text:
            print("Error: --text is required for single mode")
            exit(1)
        run_tts_generation(args)

import asyncio
from typing import Any

from fastapi import Request
from fastapi.responses import Response
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.protocol.audio import (
    AudioResponse,
    BatchSpeechRequest,
    BatchSpeechResponse,
    CreateAudio,
    OpenAICreateSpeechRequest,
    SpeechRequestItem,
    SpeechResultItem,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

# TTS Configuration (currently supports Qwen3-TTS)
_TTS_MODEL_STAGES: set[str] = {"qwen3_tts"}
_TTS_LANGUAGES: set[str] = {
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
}
_TTS_MAX_INSTRUCTIONS_LENGTH = 500
_TTS_MAX_NEW_TOKENS_MIN = 1
_TTS_MAX_NEW_TOKENS_MAX = 4096


class OmniOpenAIServingSpeech(OpenAIServing, AudioMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load supported speakers
        self.supported_speakers = self._load_supported_speakers()
        logger.info(f"Loaded {len(self.supported_speakers)} supported speakers: {sorted(self.supported_speakers)}")

    def _load_supported_speakers(self) -> set[str]:
        """Load supported speakers (case-insensitive) from the model configuration."""
        try:
            talker_config = self.engine_client.model_config.hf_config.talker_config

            # Check for speakers in either spk_id or speaker_id
            for attr_name in ["spk_id", "speaker_id"]:
                speakers_dict = getattr(talker_config, attr_name, None)
                if speakers_dict and isinstance(speakers_dict, dict):
                    # Normalize to lowercase for case-insensitive matching
                    return {speaker.lower() for speaker in speakers_dict.keys()}

            logger.warning("No speakers found in talker_config (checked spk_id and speaker_id)")
        except Exception as e:
            logger.warning(f"Could not load speakers from model config: {e}")

        return set()

    def _is_tts_model(self) -> bool:
        """Check if the current model is a supported TTS model."""
        stage_list = getattr(self.engine_client, "stage_list", None)
        if stage_list:
            for stage in stage_list:
                model_stage = getattr(stage, "model_stage", None)
                if model_stage in _TTS_MODEL_STAGES:
                    return True
        return False

    def _validate_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate TTS request parameters. Returns error message or None."""
        task_type = request.task_type or "CustomVoice"

        # Normalize voice to lowercase for case-insensitive matching
        if request.voice is not None:
            request.voice = request.voice.lower()

        # Validate input is not empty
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # Validate language
        if request.language is not None and request.language not in _TTS_LANGUAGES:
            return f"Invalid language '{request.language}'. Supported: {', '.join(sorted(_TTS_LANGUAGES))}"

        # Validate speaker for CustomVoice task
        if task_type == "CustomVoice" and request.voice is not None:
            if self.supported_speakers and request.voice not in self.supported_speakers:
                return f"Invalid speaker '{request.voice}'. Supported: {', '.join(sorted(self.supported_speakers))}"

        # Validate Base task requirements
        if task_type == "Base":
            if request.ref_audio is None:
                return "Base task requires 'ref_audio' for voice cloning"
            # Validate ref_audio format
            if not (request.ref_audio.startswith(("http://", "https://")) or request.ref_audio.startswith("data:")):
                return "ref_audio must be a URL (http/https) or base64 data URL (data:...)"

        # Validate cross-parameter dependencies
        if task_type != "Base":
            if request.ref_text is not None:
                return "'ref_text' is only valid for Base task"
            if request.x_vector_only_mode is not None:
                return "'x_vector_only_mode' is only valid for Base task"

        # Validate VoiceDesign task requirements
        if task_type == "VoiceDesign" and not request.instructions:
            return "VoiceDesign task requires 'instructions' to describe the voice"

        # Validate instructions length
        if request.instructions and len(request.instructions) > _TTS_MAX_INSTRUCTIONS_LENGTH:
            return f"Instructions too long (max {_TTS_MAX_INSTRUCTIONS_LENGTH} characters)"

        # Validate max_new_tokens range
        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"

        return None

    def _build_tts_prompt(self, text: str) -> str:
        """Build TTS prompt from input text."""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_tts_params(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
        """Build TTS parameters from request.

        Processes each parameter if present, skips if not.
        Values are wrapped in lists as required by the model.
        """
        params: dict[str, Any] = {}

        # Text content (always required)
        params["text"] = [request.input]

        # Task type
        if request.task_type is not None:
            params["task_type"] = [request.task_type]
        else:
            params["task_type"] = ["CustomVoice"]

        # Language
        if request.language is not None:
            params["language"] = [request.language]
        else:
            params["language"] = ["Auto"]

        # Speaker (voice)
        if request.voice is not None:
            params["speaker"] = [request.voice]
        elif params["task_type"][0] == "CustomVoice":
            params["speaker"] = ["Vivian"]  # Default for CustomVoice

        # Instructions for style/emotion control
        if request.instructions is not None:
            params["instruct"] = [request.instructions]
        else:
            params["instruct"] = [""]

        # Voice clone parameters (used with Base task)
        if request.ref_audio is not None:
            params["ref_audio"] = [request.ref_audio]
        if request.ref_text is not None:
            params["ref_text"] = [request.ref_text]
        if request.x_vector_only_mode is not None:
            params["x_vector_only_mode"] = [request.x_vector_only_mode]

        # Generation parameters
        if request.max_new_tokens is not None:
            params["max_new_tokens"] = [request.max_new_tokens]
        else:
            params["max_new_tokens"] = [2048]

        return params

    def _build_batch_tts_params(self, items: list[SpeechRequestItem]) -> dict[str, Any]:
        """Build batched TTS parameters from multiple request items.

        Combines parameters from all items into lists for true batch processing.
        All values must be lists (not scalars) for the input processor.
        """
        params: dict[str, Any] = {}

        # Collect all texts
        params["text"] = [item.input for item in items]

        # Collect task types
        task_types = []
        for item in items:
            task_types.append(item.task_type if item.task_type is not None else "CustomVoice")
        params["task_type"] = task_types

        # Collect languages
        params["language"] = [
            item.language if item.language is not None else "Auto"
            for item in items
        ]

        # Collect speakers (voices)
        speakers = []
        for i, item in enumerate(items):
            if item.voice is not None:
                speakers.append(item.voice.lower())
            elif task_types[i] == "CustomVoice":
                speakers.append("vivian")
            else:
                speakers.append("")
        params["speaker"] = speakers

        # Collect instructions
        params["instruct"] = [
            item.instructions if item.instructions is not None else ""
            for item in items
        ]

        # Voice clone parameters (used with Base task)
        ref_audios = [item.ref_audio if item.ref_audio else "" for item in items]
        if any(r for r in ref_audios):
            params["ref_audio"] = ref_audios
        ref_texts = [item.ref_text if item.ref_text else "" for item in items]
        if any(r for r in ref_texts):
            params["ref_text"] = ref_texts
        x_vector_modes = [item.x_vector_only_mode if item.x_vector_only_mode else False for item in items]
        if any(x for x in x_vector_modes):
            params["x_vector_only_mode"] = x_vector_modes

        # Generation parameters - use max from all items
        max_tokens_list = [item.max_new_tokens for item in items]
        if any(m is not None for m in max_tokens_list):
            max_val = max(m if m is not None else 2048 for m in max_tokens_list)
            params["max_new_tokens"] = [max_val]
        else:
            params["max_new_tokens"] = [2048]

        return params

    async def create_speech(
        self,
        request: OpenAICreateSpeechRequest,
        raw_request: Request | None = None,
    ):
        """
        Create Speech API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createSpeech
        for the API specification. This API mimics the OpenAI
        Create Speech API.

        For Qwen3-TTS models, additional parameters are supported:
        - task_type: "CustomVoice", "VoiceDesign", or "Base"
        - language: Language code (e.g., "Chinese", "English", "Auto")
        - voice: Speaker name (e.g., "Vivian", "Ryan") for CustomVoice
        - instructions: Voice style/emotion instructions
        - ref_audio: Reference audio for voice cloning (Base task)
        - ref_text: Transcript of reference audio (Base task)
        - x_vector_only_mode: Use speaker embedding only (Base task)

        NOTE: Streaming audio generation is not currently supported.
        """

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_id = f"speech-{random_uuid()}"

        try:
            if self._is_tts_model():
                # Validate TTS parameters
                validation_error = self._validate_tts_request(request)
                if validation_error:
                    return self.create_error_response(validation_error)

                # Build TTS parameters and prompt
                tts_params = self._build_tts_params(request)
                prompt_text = self._build_tts_prompt(request.input)
                prompt = {
                    "prompt": prompt_text,
                    "additional_information": tts_params,
                }
            else:
                # Fallback for unsupported models
                tts_params = {}
                prompt = {"prompt": request.input}

            logger.info(
                "TTS speech request %s: text=%r, task_type=%s",
                request_id,
                request.input[:50] + "..." if len(request.input) > 50 else request.input,
                tts_params.get("task_type", ["unknown"])[0],
            )

            sampling_params_list = self.engine_client.default_sampling_params_list

            generator = self.engine_client.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                output_modalities=["audio"],
            )

            final_output: OmniRequestOutput | None = None
            async for res in generator:
                final_output = res

            if final_output is None:
                return self.create_error_response("No output generated from the model.")

            # Extract audio from output
            # Audio can be in final_output.multimodal_output or final_output.request_output.multimodal_output
            audio_output = None
            if hasattr(final_output, "multimodal_output") and final_output.multimodal_output:
                audio_output = final_output.multimodal_output
            if (not audio_output or "audio" not in audio_output) and hasattr(final_output, "request_output"):
                if final_output.request_output and hasattr(final_output.request_output, "multimodal_output"):
                    audio_output = final_output.request_output.multimodal_output

            if not audio_output or "audio" not in audio_output:
                return self.create_error_response("TTS model did not produce audio output.")

            audio_tensor = audio_output["audio"]
            sample_rate = audio_output.get("sr", 24000)
            if hasattr(sample_rate, "item"):
                sample_rate = sample_rate.item()

            # Convert tensor to numpy
            if hasattr(audio_tensor, "float"):
                audio_tensor = audio_tensor.float().detach().cpu().numpy()

            # Squeeze batch dimension if present, but preserve channel dimension for stereo
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()

            audio_obj = CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=int(sample_rate),
                response_format=request.response_format or "wav",
                speed=request.speed or 1.0,
                stream_format=request.stream_format,
                base64_encode=False,
            )

            audio_response: AudioResponse = self.create_audio(audio_obj)
            return Response(content=audio_response.audio_data, media_type=audio_response.media_type)

        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(str(e))
        except Exception as e:
            logger.exception("Speech generation failed: %s", e)
            return self.create_error_response(f"Speech generation failed: {e}")

    def _convert_batch_item_to_request(
        self,
        item: SpeechRequestItem,
        model: str | None = None,
    ) -> OpenAICreateSpeechRequest:
        """Convert a batch item to a full speech request."""
        return OpenAICreateSpeechRequest(
            input=item.input,
            model=model,
            voice=item.voice,
            instructions=item.instructions,
            response_format=item.response_format,
            speed=item.speed,
            task_type=item.task_type,
            language=item.language,
            ref_audio=item.ref_audio,
            ref_text=item.ref_text,
            x_vector_only_mode=item.x_vector_only_mode,
            max_new_tokens=item.max_new_tokens,
        )

    async def _generate_single_speech(
        self,
        item: SpeechRequestItem,
        model: str | None = None,
    ) -> SpeechResultItem:
        """Generate speech for a single batch item and return the result."""
        import base64

        try:
            # Convert to full request
            request = self._convert_batch_item_to_request(item, model)

            # Validate
            if self._is_tts_model():
                validation_error = self._validate_tts_request(request)
                if validation_error:
                    return SpeechResultItem(custom_id=item.custom_id, error=validation_error)

                tts_params = self._build_tts_params(request)
                prompt_text = self._build_tts_prompt(request.input)
                prompt = {
                    "prompt": prompt_text,
                    "additional_information": tts_params,
                }
            else:
                return SpeechResultItem(
                    custom_id=item.custom_id,
                    error="Model does not support TTS",
                )

            request_id = f"speech-batch-{random_uuid()}"
            sampling_params_list = self.engine_client.default_sampling_params_list

            generator = self.engine_client.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                output_modalities=["audio"],
            )

            final_output: OmniRequestOutput | None = None
            async for res in generator:
                final_output = res

            if final_output is None:
                return SpeechResultItem(
                    custom_id=item.custom_id,
                    error="No output generated from the model.",
                )

            # Extract audio from output
            audio_output = None
            if hasattr(final_output, "multimodal_output") and final_output.multimodal_output:
                audio_output = final_output.multimodal_output
            if (not audio_output or "audio" not in audio_output) and hasattr(final_output, "request_output"):
                if final_output.request_output and hasattr(final_output.request_output, "multimodal_output"):
                    audio_output = final_output.request_output.multimodal_output

            if not audio_output or "audio" not in audio_output:
                return SpeechResultItem(
                    custom_id=item.custom_id,
                    error="TTS model did not produce audio output.",
                )

            audio_tensor = audio_output["audio"]
            sample_rate = audio_output.get("sr", 24000)
            if hasattr(sample_rate, "item"):
                sample_rate = sample_rate.item()

            # Convert tensor to numpy
            if hasattr(audio_tensor, "float"):
                audio_tensor = audio_tensor.float().detach().cpu().numpy()

            # Squeeze batch dimension if present
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()

            audio_obj = CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=int(sample_rate),
                response_format=request.response_format or "wav",
                speed=request.speed or 1.0,
                stream_format=request.stream_format,
                base64_encode=True,  # For batch, return base64
            )

            audio_response: AudioResponse = self.create_audio(audio_obj)

            # Ensure base64 encoding for batch response
            audio_data = audio_response.audio_data
            if isinstance(audio_data, bytes):
                audio_data = base64.b64encode(audio_data).decode("utf-8")

            return SpeechResultItem(
                custom_id=item.custom_id,
                audio_base64=audio_data,
                media_type=audio_response.media_type,
            )

        except asyncio.CancelledError:
            return SpeechResultItem(custom_id=item.custom_id, error="Request cancelled")
        except Exception as e:
            logger.exception("Batch speech generation failed for %s: %s", item.custom_id, e)
            return SpeechResultItem(custom_id=item.custom_id, error=str(e))

    async def create_speech_batch(
        self,
        request: BatchSpeechRequest,
        raw_request: Request | None = None,
    ) -> BatchSpeechResponse:
        """
        Create Speech Batch API for processing multiple TTS requests in a single GPU batch.

        This endpoint allows you to submit multiple TTS requests at once and receive
        all results in a single response. Each request item must have a unique
        `custom_id` which is returned with the corresponding result.

        TRUE GPU BATCHING: All requests are processed together in a single model
        forward pass for maximum GPU efficiency.

        Args:
            request: BatchSpeechRequest containing a list of speech generation requests
            raw_request: The raw FastAPI request object

        Returns:
            BatchSpeechResponse with results for each request item
        """
        import base64

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        batch_size = len(request.requests)
        logger.info("TTS batch request (GPU batching): %d items", batch_size)

        # Validate all items first
        custom_ids = []
        valid_items = []
        results: list[SpeechResultItem | None] = [None] * batch_size
        item_indices: dict[int, int] = {}  # Map valid item index to original index

        if not self._is_tts_model():
            return BatchSpeechResponse(
                results=[
                    SpeechResultItem(custom_id=item.custom_id, error="Model does not support TTS")
                    for item in request.requests
                ]
            )

        for i, item in enumerate(request.requests):
            custom_ids.append(item.custom_id)
            full_request = self._convert_batch_item_to_request(item, request.model)
            validation_error = self._validate_tts_request(full_request)

            if validation_error:
                results[i] = SpeechResultItem(custom_id=item.custom_id, error=validation_error)
            else:
                item_indices[len(valid_items)] = i
                valid_items.append(item)

        if not valid_items:
            return BatchSpeechResponse(results=results)

        try:
            # Build batched TTS parameters
            batch_params = self._build_batch_tts_params(valid_items)

            # Build prompt (text content is in batch_params)
            prompt_text = self._build_tts_prompt(valid_items[0].input)
            prompt = {
                "prompt": prompt_text,
                "additional_information": batch_params,
            }

            request_id = f"speech-batch-{random_uuid()}"
            sampling_params_list = self.engine_client.default_sampling_params_list

            generator = self.engine_client.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                output_modalities=["audio"],
            )

            final_output: OmniRequestOutput | None = None
            async for res in generator:
                final_output = res

            if final_output is None:
                for valid_idx, orig_idx in item_indices.items():
                    results[orig_idx] = SpeechResultItem(
                        custom_id=custom_ids[orig_idx],
                        error="No output generated from the model.",
                    )
                return BatchSpeechResponse(results=results)

            # Extract audio output
            audio_output = None
            if hasattr(final_output, "multimodal_output") and final_output.multimodal_output:
                audio_output = final_output.multimodal_output
            if (not audio_output or "audio" not in audio_output) and hasattr(final_output, "request_output"):
                if final_output.request_output and hasattr(final_output.request_output, "multimodal_output"):
                    audio_output = final_output.request_output.multimodal_output

            if not audio_output:
                for valid_idx, orig_idx in item_indices.items():
                    results[orig_idx] = SpeechResultItem(
                        custom_id=custom_ids[orig_idx],
                        error="TTS model did not produce audio output.",
                    )
                return BatchSpeechResponse(results=results)

            # Get concatenated audio and lengths
            audio_concat = audio_output.get("audio")
            audio_lengths = audio_output.get("audio_lengths")
            sample_rate = audio_output.get("sr", 24000)
            if hasattr(sample_rate, "item"):
                sample_rate = sample_rate.item()

            # Check if this is a batch output (has lengths) or single output
            if audio_lengths is not None:
                # True batch output - split concatenated audio using lengths
                if hasattr(audio_lengths, "tolist"):
                    lengths_list = audio_lengths.tolist()
                else:
                    lengths_list = list(audio_lengths)

                # Convert audio_concat to numpy if needed
                if hasattr(audio_concat, "float"):
                    audio_concat = audio_concat.float().detach().cpu().numpy()

                # Split into individual audio tensors
                audio_tensors = []
                offset = 0
                for length in lengths_list:
                    audio_tensors.append(audio_concat[offset:offset + length])
                    offset += length

                logger.info("Batch output: %d audio segments from concatenated tensor", len(audio_tensors))
            else:
                # Single output (fallback)
                if hasattr(audio_concat, "float"):
                    audio_concat = audio_concat.float().detach().cpu().numpy()
                if audio_concat.ndim > 1:
                    audio_concat = audio_concat.squeeze()
                audio_tensors = [audio_concat]

            # Process each audio and create results
            for valid_idx, audio_tensor in enumerate(audio_tensors):
                orig_idx = item_indices.get(valid_idx)
                if orig_idx is None:
                    continue

                item = valid_items[valid_idx]

                try:
                    # Squeeze if needed
                    if audio_tensor.ndim > 1:
                        audio_tensor = audio_tensor.squeeze()

                    audio_obj = CreateAudio(
                        audio_tensor=audio_tensor,
                        sample_rate=int(sample_rate),
                        response_format=item.response_format or "wav",
                        speed=item.speed or 1.0,
                        base64_encode=True,
                    )

                    audio_response: AudioResponse = self.create_audio(audio_obj)

                    audio_bytes = audio_response.audio_data
                    if isinstance(audio_bytes, bytes):
                        audio_bytes = base64.b64encode(audio_bytes).decode("utf-8")

                    results[orig_idx] = SpeechResultItem(
                        custom_id=item.custom_id,
                        audio_base64=audio_bytes,
                        media_type=audio_response.media_type,
                    )
                except Exception as e:
                    logger.exception("Failed to process batch audio item %d: %s", valid_idx, e)
                    results[orig_idx] = SpeechResultItem(
                        custom_id=item.custom_id,
                        error=f"Failed to process audio: {e}",
                    )

            # Fill in any missing results
            for i, result in enumerate(results):
                if result is None:
                    results[i] = SpeechResultItem(
                        custom_id=custom_ids[i],
                        error="Unexpected error: result not generated",
                    )

            return BatchSpeechResponse(results=results)

        except asyncio.CancelledError:
            return BatchSpeechResponse(
                results=[
                    SpeechResultItem(custom_id=cid, error="Request cancelled")
                    for cid in custom_ids
                ]
            )
        except Exception as e:
            logger.exception("Batch speech generation failed: %s", e)
            return BatchSpeechResponse(
                results=[
                    SpeechResultItem(custom_id=cid, error=str(e))
                    for cid in custom_ids
                ]
            )

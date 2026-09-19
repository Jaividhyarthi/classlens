"""Thin Sarvam AI HTTP client.

Every endpoint path, auth header, and request field name here is taken
verbatim from Sarvam's own published Python SDK (`sarvamai` on PyPI,
v0.1.34, Fern-generated from their real API definition) -- pulled and
read directly since docs.sarvam.ai is blocked by this environment's
network egress policy. See pipeline/SARVAM_API_NOTES.md for the full
trail of what was verified and how.
"""
import os
import time
import typing
from pathlib import Path

import httpx

SARVAM_BASE_URL = "https://api.sarvam.ai"


class SarvamConfigError(RuntimeError):
    pass


class SarvamAPIError(RuntimeError):
    def __init__(self, status_code: int, body: typing.Any):
        super().__init__(f"Sarvam API error {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def _api_key() -> str:
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        raise SarvamConfigError(
            "SARVAM_API_KEY is not set. Get a key from https://dashboard.sarvam.ai "
            "and export it before running the pipeline."
        )
    return key


def _headers(extra: typing.Optional[dict] = None) -> dict:
    # Confirmed from sarvamai/core/client_wrapper.py: header name is
    # "api-subscription-key", not "Authorization: Bearer ...".
    headers = {"api-subscription-key": _api_key()}
    if extra:
        headers.update(extra)
    return headers


def _raise_for_status(response: httpx.Response) -> None:
    if 200 <= response.status_code < 300:
        return
    try:
        body = response.json()
    except Exception:
        body = response.text
    raise SarvamAPIError(response.status_code, body)


def chat_completion(
    messages: list[dict],
    model: str = "sarvam-m",
    temperature: float = 0.2,
    max_tokens: typing.Optional[int] = None,
    response_format: typing.Optional[dict] = None,
    timeout: float = 60.0,
) -> str:
    """POST /v1/chat/completions -- confirmed OpenAI-compatible shape from
    sarvamai/chat/raw_client.py and types/create_chat_completion_response.py.
    Returns choices[0].message.content.
    """
    payload: dict = {"messages": messages, "model": model, "temperature": temperature}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if response_format is not None:
        payload["response_format"] = response_format

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{SARVAM_BASE_URL}/v1/chat/completions",
            headers=_headers({"content-type": "application/json"}),
            json=payload,
        )
    _raise_for_status(resp)
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _guess_content_type(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return {
        "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
        "ogg": "audio/ogg", "m4a": "audio/mp4", "aac": "audio/aac",
    }.get(ext, "audio/wav")


class SpeechToTextBatchJob:
    """Batch STT job with diarization.

    Diarization is confirmed NOT available on the synchronous /speech-to-text
    REST endpoint (per that endpoint's own docstring in the SDK) -- only the
    batch job flow supports with_diarization. Workflow, verified against
    sarvamai/speech_to_text_job/raw_client.py and job.py:
      1. POST /speech-to-text/job/v1                 (create job)
      2. POST /speech-to-text/job/v1/upload-files     (get signed PUT URL)
      3. PUT <signed url>                             (upload audio bytes)
      4. POST /speech-to-text/job/v1/{job_id}/start   (kick off processing)
      5. GET  /speech-to-text/job/v1/{job_id}/status  (poll until Completed)
      6. POST /speech-to-text/job/v1/download-files   (get signed GET URL)
      7. GET  <signed url>                            (download result JSON)
    """

    def __init__(
        self,
        audio_path: Path,
        language_code: str = "unknown",
        model: str = "saaras:v3",
        num_speakers: typing.Optional[int] = None,
        input_audio_codec: typing.Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.audio_path = audio_path
        self.language_code = language_code
        self.model = model
        self.num_speakers = num_speakers
        self.input_audio_codec = input_audio_codec
        self.timeout = timeout
        self.job_id: typing.Optional[str] = None

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout)

    def initialise(self) -> str:
        job_parameters = {
            "language_code": self.language_code,
            "model": self.model,
            "with_timestamps": True,
            "with_diarization": True,
        }
        if self.num_speakers:
            job_parameters["num_speakers"] = self.num_speakers
        if self.input_audio_codec:
            job_parameters["input_audio_codec"] = self.input_audio_codec

        with self._client() as client:
            resp = client.post(
                f"{SARVAM_BASE_URL}/speech-to-text/job/v1",
                headers=_headers({"content-type": "application/json"}),
                json={"job_parameters": job_parameters},
            )
        _raise_for_status(resp)
        data = resp.json()
        self.job_id = data["job_id"]
        return self.job_id

    def upload(self) -> None:
        assert self.job_id, "call initialise() first"
        file_name = self.audio_path.name
        with self._client() as client:
            resp = client.post(
                f"{SARVAM_BASE_URL}/speech-to-text/job/v1/upload-files",
                headers=_headers({"content-type": "application/json"}),
                json={"job_id": self.job_id, "files": [file_name]},
            )
        _raise_for_status(resp)
        upload_url = resp.json()["upload_urls"][file_name]["file_url"]

        with self._client() as client:
            put_resp = client.put(
                upload_url,
                headers={
                    "x-ms-blob-type": "BlockBlob",
                    "Content-Type": _guess_content_type(self.audio_path),
                },
                content=self.audio_path.read_bytes(),
            )
        if not (200 <= put_resp.status_code < 300):
            raise SarvamAPIError(put_resp.status_code, put_resp.text)

    def start(self) -> dict:
        assert self.job_id, "call initialise() first"
        with self._client() as client:
            resp = client.post(
                f"{SARVAM_BASE_URL}/speech-to-text/job/v1/{self.job_id}/start",
                headers=_headers(),
            )
        _raise_for_status(resp)
        return resp.json()

    def get_status(self) -> dict:
        assert self.job_id, "call initialise() first"
        with self._client() as client:
            resp = client.get(
                f"{SARVAM_BASE_URL}/speech-to-text/job/v1/{self.job_id}/status",
                headers=_headers(),
            )
        _raise_for_status(resp)
        return resp.json()

    def wait_until_complete(self, poll_interval: float = 5.0, overall_timeout: float = 900.0) -> dict:
        start_time = time.monotonic()
        while True:
            status = self.get_status()
            state = status.get("job_state", "").lower()
            if state in ("completed", "failed"):
                return status
            if time.monotonic() - start_time > overall_timeout:
                raise TimeoutError(f"Sarvam job {self.job_id} did not finish within {overall_timeout}s (last state: {state})")
            time.sleep(poll_interval)

    def download_result(self, output_file_name: str) -> dict:
        assert self.job_id, "call initialise() first"
        with self._client() as client:
            resp = client.post(
                f"{SARVAM_BASE_URL}/speech-to-text/job/v1/download-files",
                headers=_headers({"content-type": "application/json"}),
                json={"job_id": self.job_id, "files": [output_file_name]},
            )
        _raise_for_status(resp)
        download_url = resp.json()["download_urls"][output_file_name]["file_url"]

        with self._client() as client:
            get_resp = client.get(download_url)
        if not (200 <= get_resp.status_code < 300):
            raise SarvamAPIError(get_resp.status_code, get_resp.text)
        return get_resp.json()

    def run(self, poll_interval: float = 5.0, overall_timeout: float = 900.0) -> dict:
        """Runs the full workflow and returns the raw per-file output JSON."""
        self.initialise()
        self.upload()
        self.start()
        status = self.wait_until_complete(poll_interval=poll_interval, overall_timeout=overall_timeout)
        if status.get("job_state", "").lower() != "completed":
            raise SarvamAPIError(0, status.get("error_message") or status)

        job_details = status.get("job_details") or []
        matching = [d for d in job_details if d.get("state") == "Success" and d.get("outputs")]
        if not matching:
            raise SarvamAPIError(0, f"No successful outputs in job {self.job_id}: {job_details}")
        output_file_name = matching[0]["outputs"][0]["file_name"]
        return self.download_result(output_file_name)

# Sarvam API verification notes

`docs.sarvam.ai` is blocked by this environment's network egress policy
(confirmed via the proxy status endpoint, not a transient failure). Per
constraint #3 in the task brief, nothing here was guessed from training
data. Instead I pulled the real, currently-published Sarvam Python SDK
(`pip download sarvamai==0.1.34`, latest on PyPI as of this session) and
read its source directly -- it's Fern-generated straight from Sarvam's own
API definition, which is about as authoritative as the live docs site.

## Confirmed directly from SDK source (high confidence)

- Base URL: `https://api.sarvam.ai` (`sarvamai/environment.py`)
- Auth header: `api-subscription-key: <key>` -- **not** `Authorization: Bearer`
  (`sarvamai/core/client_wrapper.py`)
- `POST /speech-to-text` (sync REST): multipart `file` + `model`, `mode`,
  `language_code`, `with_timestamps`, `input_audio_codec`, `keyterms`.
  **Diarization is not supported here** -- confirmed in the endpoint's own
  docstring in `sarvamai/speech_to_text/raw_client.py`.
- Batch job flow (required for diarization), all in
  `sarvamai/speech_to_text_job/raw_client.py` + `job.py`:
  - `POST /speech-to-text/job/v1` -- body `{"job_parameters": {...}}`, returns
    `{job_id, job_state, storage_container_type, job_parameters}`
  - `job_parameters` fields (`sarvamai/requests/speech_to_text_job_parameters.py`):
    `language_code`, `model`, `mode`, `with_timestamps`, `with_diarization`,
    `num_speakers`, `input_audio_codec`, `keyterms`
  - `POST /speech-to-text/job/v1/upload-files` -- body `{job_id, files: [name]}`,
    returns `upload_urls: {name: {file_url, file_metadata}}`
  - `PUT <file_url>` with headers `x-ms-blob-type: BlockBlob`, `Content-Type`
    (Azure Blob SAS URL pattern) and raw audio bytes as the body
  - `POST /speech-to-text/job/v1/{job_id}/start` (optional `ptu_id` query param)
  - `GET /speech-to-text/job/v1/{job_id}/status` -- returns
    `{job_state: Accepted|Pending|Running|Completed|Failed, job_details: [{inputs, outputs, state: Success|"API Error"|"Internal Server Error", error_message}]}`
  - `POST /speech-to-text/job/v1/download-files` -- body `{job_id, files: [output_name]}`,
    returns `download_urls: {name: {file_url}}`
  - `GET <file_url>` -- raw output JSON for that input file
- `POST /v1/chat/completions` -- OpenAI-compatible: `{messages, model,
  temperature, response_format, ...}` -> `{choices: [{message: {role,
  content}}], usage, ...}` (`sarvamai/chat/raw_client.py` +
  `types/create_chat_completion_response.py`). `response_format` supports
  `{"type": "json_schema", "json_schema": {...}}` for structured output
  (`types/response_format.py`) -- used in `relevance.py` and `report.py`.
  Model IDs seen in the SDK's own docstrings/comments: `sarvam-m` (used as
  the default here), `sarvam-105b`, `glm5.3`, `gemma4`, `deepseekv4-flash`.
  `GET /v2/models` lists what's actually available to your key.

## Inferred, not directly confirmed (documented so it's not silently trusted)

The exact JSON shape of a **batch job's per-file output** (the file you
download from `download-files`) isn't formally typed anywhere in the SDK --
the client just downloads raw bytes to disk. My best evidence: the SDK
ships dedicated `DiarizedTranscript`/`DiarizedEntry` types
(`transcript`, `start_time_seconds`, `end_time_seconds`, `speaker_id`) built
for exactly this purpose, and the plain `SpeechToTextResponse` type has
`transcript`, `timestamps`, `language_code`, `language_probability`. I've
assumed the batch output merges these: a top-level object with those REST
fields plus a `diarized_transcript: {entries: [...]}` key.

`pipeline/asr_diarize.py`'s `_parse_batch_output` is written defensively
against this uncertainty: if `diarized_transcript.entries` isn't where
expected, it raises with the actual top-level keys in the payload instead
of silently returning an empty transcript. **The first real run against a
live API key is the actual verification of this one field** -- if it
raises, fix the key name in `_parse_batch_output` from the error message.

## Not verified at all

No `SARVAM_API_KEY` was available in this session, so no live call was made
end-to-end. Everything above is verified against the published client
contract, not a live response.

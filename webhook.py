"""Async webhook client — sends caller text, receives response for TTS."""

import logging

import httpx

log = logging.getLogger(__name__)

FALLBACK_MESSAGE = "I'm having trouble processing that. Could you say that again?"


async def get_response(
    url: str,
    text: str,
    call_sid: str,
    caller_number: str,
    timeout: float = 5.0,
) -> str:
    """POST transcribed text to webhook, return response text for TTS.

    Args:
        url: Webhook endpoint URL.
        text: Transcribed caller speech.
        call_sid: SignalWire call SID.
        caller_number: Caller's phone number.
        timeout: HTTP request timeout in seconds.

    Returns:
        Response text to speak. Includes [HANGUP] if the webhook signals it.
        Returns a fallback message on any error.
    """
    payload = {
        "text": text,
        "call_sid": call_sid,
        "caller_number": caller_number,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )

        if response.status_code != 200:
            log.error("Webhook returned %d: %s", response.status_code, response.text[:200])
            return FALLBACK_MESSAGE

        body = response.text.strip()
        if not body:
            log.warning("Webhook returned empty body")
            return FALLBACK_MESSAGE

        log.info("Webhook response: %s", body[:100])
        return body

    except httpx.TimeoutException:
        log.error("Webhook timed out after %.1fs: %s", timeout, url)
        return FALLBACK_MESSAGE
    except httpx.ConnectError as e:
        log.error("Webhook unreachable: %s — %s", url, e)
        return FALLBACK_MESSAGE
    except Exception as e:
        log.error("Webhook request failed: %s: %s", type(e).__name__, e)
        return FALLBACK_MESSAGE

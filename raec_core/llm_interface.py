import requests
import time

class LLMInterface:
    def __init__(
        self,
        model="raec:latest",
        endpoint="http://localhost:11434/api/generate",
        timeout=300
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def generate(
        self,
        prompt,
        temperature=0.7,
        max_tokens=1024,
        stop=None
    ):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        }

        if stop:
            payload["stop"] = stop

        for attempt in range(3):
            try:
                r = requests.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout
                )
                r.raise_for_status()
                return r.json()["response"].strip()
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"LLM call failed: {e}")
                time.sleep(1.5)

    # --- Streaming version ---
    def stream(self, prompt, temperature=0.7, max_tokens=1024, stop=None):
        """
        Generator: yields tokens from the LLM as they arrive.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": True  # streaming mode
        }

        if stop:
            payload["stop"] = stop

        with requests.post(self.endpoint, json=payload, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    # Each line is a partial response token from the API
                    yield line

"""Regression checks for the OpenCode backend."""
import json
import tempfile
import unittest
from types import SimpleNamespace

import llmbench as L


class OpenCodeTests(unittest.TestCase):
    def setUp(self):
        self.backend = L.OpenCodeBackend(SimpleNamespace(
            model="test-model", endpoint="http://localhost:1234/v1",
            api_key="secret-value", context_tokens=8192))

    def test_provider_is_selected_without_writing_config_or_secret(self):
        with tempfile.TemporaryDirectory() as scratch:
            cmd, env = self.backend.build({"prompt": "hello"}, scratch)
            config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
            self.assertEqual(cmd[cmd.index("--model") + 1], "llmbench/test-model")
            self.assertIn("--standalone", cmd)
            self.assertEqual(env["XDG_CONFIG_HOME"], scratch)
            self.assertEqual(env["OPENCODE_CONFIG_PROJECT_DISABLE"], "true")
            provider = config["providers"]["llmbench"]
            self.assertEqual(provider["package"], "@opencode/ai/providers/openai-compatible")
            self.assertEqual(provider["settings"]["apiKey"], "{env:LLMBENCH_OPENCODE_API_KEY}")
            self.assertEqual(env["LLMBENCH_OPENCODE_API_KEY"], "secret-value")
            self.assertEqual(L.os.listdir(scratch), [])

    def test_failure_not_masked_by_partial_text(self):
        outcome = L.TaskOutcome()
        outcome.text = "partial"
        outcome.exit_code = 1
        outcome.stderr_tail = "failure"
        self.backend.finalize(outcome)
        self.assertIn("failure", outcome.error)


if __name__ == "__main__":
    unittest.main()

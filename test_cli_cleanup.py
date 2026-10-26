"""Regression tests for CLI child lifecycle cleanup."""
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import llmbench as L


class ProbeBackend(L.CliBackend):
    def build(self, task, scratch):
        return [sys.executable, "-u", "-c", task["script"]], None

    def parse_line(self, outcome, line):
        raise KeyboardInterrupt


class CliCleanupTests(unittest.TestCase):
    def run_probe(self, task, set_root, timeout=5):
        backend = ProbeBackend(SimpleNamespace(timeout=timeout))
        real_popen = subprocess.Popen
        children = []

        def spawn(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            children.append(proc)
            return proc

        try:
            with tempfile.TemporaryDirectory() as scratch:
                with patch.object(L.subprocess, "Popen", side_effect=spawn):
                    with self.assertRaises(KeyboardInterrupt):
                        backend.run(task, scratch, set_root)
            self.assertIsNotNone(children[0].returncode,
                                 "interrupted child must be killed and reaped")
            for pipe in (children[0].stdin, children[0].stdout, children[0].stderr):
                if pipe is not None:
                    self.assertTrue(pipe.closed)
        finally:
            for proc in children:
                if proc.poll() is None:
                    backend._kill(proc)
                proc.wait()
                for pipe in (proc.stdin, proc.stdout, proc.stderr):
                    if pipe is not None:
                        pipe.close()

    def test_interrupt_during_output_processing(self):
        self.run_probe({"prompt": "hi", "script":
                        "import time; print('ready'); time.sleep(60)"},
                       lambda pid: None)

    def test_normal_exit_and_timeout(self):
        backend = ProbeBackend(SimpleNamespace(timeout=0.1))
        for script, expected_error in (("pass", None),
                                       ("import time; time.sleep(60)",
                                        "timeout after 0s")):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as scratch:
                pids = []
                outcome = backend.run({"prompt": "hi", "script": script},
                                      scratch, pids.append)
                self.assertEqual(outcome.error, expected_error)
                self.assertIsNotNone(outcome.exit_code)
                with self.assertRaises(ProcessLookupError):
                    L.os.kill(pids[0], 0)

    def test_interrupt_immediately_after_spawn(self):
        def interrupt(pid):
            raise KeyboardInterrupt

        self.run_probe({"prompt": "hi", "script": "import time; time.sleep(60)"},
                       interrupt)


if __name__ == "__main__":
    unittest.main()

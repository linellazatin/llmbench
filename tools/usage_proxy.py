#!/usr/bin/env python3
"""Loopback usage-logging proxy for OpenAI-compatible endpoints.

Sits in front of a local OpenAI-compatible server (default upstream
127.0.0.1:8000, e.g. vLLM) and forwards every request unchanged, except
that it injects `stream_options.include_usage = true` into streaming
chat/completions & completions bodies — the one thing a client cannot ask
for in a non-streaming call is already included by default. It appends one
JSON record per request to the log file with:

  - server-reported usage (prompt/completion/total tokens)
  - a fingerprint of the request's last user message (normalized head/tail/
    char count) so records can be correlated to benchmark tasks afterwards
  - path, model, stream flag, status, epoch time

Stdlib only, no dependencies.

  python3 tools/usage_proxy.py --port 8001 --log results/dsh-proxy-usage.jsonl

Point clients at http://127.0.0.1:8001/v1 (model names pass through).
"""
import argparse
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {"host", "connection", "content-length", "transfer-encoding",
       "keep-alive", "proxy-authorization", "te", "upgrade"}


def norm(s: str) -> str:
    return " ".join((s or "").split())


def _msg_content(m: dict):
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(b.get("text", "") for b in c if isinstance(b, dict))
    return ""


def _last_user(j: dict):
    for m in reversed(j.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return norm(_msg_content(m))
    return ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llmbench-usage-proxy/1.0"
    upstream = ("127.0.0.1", 8000)
    log_path = "usage-proxy.jsonl"
    _log_lock = threading.Lock()
    _seq = 0

    def log(self, **rec):
        type(self)._seq += 1
        rec["seq"] = type(self)._seq
        with type(self)._log_lock:
            with open(type(self).log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    # -- plumbing ----------------------------------------------------------
    def _fwd_headers(self):
        return [(k, v) for k, v in self.headers.items() if k.lower() not in HOP]

    def _read_body(self):
        cl = self.headers.get("content-length")
        return self.rfile.read(int(cl)) if cl else b""

    def _close_headers(self, upstream_headers):
        self.send_header("connection", "close")
        for k, v in upstream_headers:
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.end_headers()

    def _proxy(self, method):
        path = self.path
        body = self._read_body()
        meta, is_stream = {}, False
        if method == "POST" and path.rstrip("/").endswith(
                ("chat/completions", "completions")):
            try:
                j = json.loads(body)
                if j.get("stream"):
                    j.setdefault("stream_options", {})["include_usage"] = True
                    body, is_stream = json.dumps(j).encode(), True
                u = _last_user(j)
                meta = {"model": j.get("model"), "stream": is_stream,
                        "n_messages": len(j.get("messages") or []),
                        "max_tokens": j.get("max_tokens"),
                        "temperature": j.get("temperature"),
                        "user_chars": len(u),
                        "user_full": u,
                        "user_head": u[:120], "user_tail": u[-80:]}
            except (json.JSONDecodeError, AttributeError):
                pass
        try:
            conn = http.client.HTTPConnection(*self.upstream, timeout=3600)
            hdrs = dict(self._fwd_headers())
            hdrs["content-length"] = str(len(body))
            conn.request(method, path, body=body or None, headers=hdrs)
            r = conn.getresponse()
        except OSError as e:
            self.send_error(502, f"upstream error: {e}")
            self.log(t=time.time(), method=method, path=path, status=502,
                       error=str(e), **meta)
            return
        if is_stream:
            self.send_response(r.status)
            self._close_headers(r.getheaders())
            usage, sse = {}, ""
            try:
                while True:
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    sse += chunk.decode("utf-8", "replace")
                    while "\n" in sse:
                        line, sse = sse.split("\n", 1)
                        line = line.strip()
                        if line.startswith("data:") and not line[5:].strip() == "[DONE]":
                            try:
                                ev = json.loads(line[5:].strip())
                            except json.JSONDecodeError:
                                continue
                            if isinstance(ev.get("usage"), dict):
                                usage = ev["usage"]
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            data = r.read()
            try:
                usage = json.loads(data).get("usage") or {}
            except json.JSONDecodeError:
                usage = {}
            self.send_response(r.status)
            self._close_headers(r.getheaders())
            self.wfile.write(data)
        self.log(t=time.time(), method=method, path=path, status=r.status,
                 prompt_tokens=usage.get("prompt_tokens"),
                 completion_tokens=usage.get("completion_tokens"),
                 total_tokens=usage.get("total_tokens"),
                 completion_tokens_details=usage.get("completion_tokens_details"),
                 **meta)

    def do_GET(self):
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

    def log_message(self, *a):  # silence default stderr noise
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--upstream-host", default="127.0.0.1")
    ap.add_argument("--upstream-port", type=int, default=8000)
    ap.add_argument("--log", default="usage-proxy.jsonl")
    a = ap.parse_args()
    Handler.upstream = (a.upstream_host, a.upstream_port)
    Handler.log_path = a.log
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

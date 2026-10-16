"""Quick verifier self-tests for llmbench (not part of the tool itself)."""
import llmbench as L

for name, tasks in L.SUITES.items():
    print(name, len(tasks), [t["id"].split("/")[1] for t in tasks])
print("grid len/path:", L._GRID_LEN, L._GRID_PATH)
print("mapwalk:", L.MAPWALK_ANSWER)
d, q, a = L._gen_scene_count()
print("scene-count:", q, "->", a)
d, q, a = L._gen_scene_compare()
print("scene-compare:", q, "->", a)
d, q, a = L._gen_scene_right()
print("scene-right:", q, "->", a)

assert L.check_exact({"response": "pong "}, {"value": "pong"})["passed"]
assert L.check_answer({"response": "x\nANSWER: 408"},
                      {"value": 408, "numeric": True})["passed"]
assert L.check_answer({"response": "ANSWER: 2, 2, west"},
                      {"value": "2,2,west", "compact": True})["passed"]
assert L.check_gridpath({"response": "ANSWER: DDDDRRRR"},
                        {"grid": L.GRID_B, "expect_len": L._GRID_LEN})["passed"]
assert not L.check_gridpath({"response": "ANSWER: RRRRD"},
                            {"grid": L.GRID_B,
                             "expect_len": L._GRID_LEN})["passed"]
assert L.safe_eval("128*46") == 5888

v = L.check_toolcall(
    {"response": "", "tool_calls": [
        {"id": None, "name": "get_weather", "args": {"city": "Tokyo"}}]},
    {"calls": [{"name": "get_weather",
                "args_subset": {"city": "tokyo"}}]})
assert v["passed"], v

v = L.check_toolcall(
    {"response": "56", "tool_calls": []},
    {"no_calls": True, "answer": 56, "numeric": True})
assert v["passed"], v

v = L.check_constraints(
    {"response": "The lighthouse has a battery. It glows at night."},
    {"checks": [{"type": "sentences", "count": 2},
                {"type": "include_all", "words": ["battery"]}]})
assert v["passed"], v

shape = {"name": "str", "age": "int",
         "hobbies": {"type": "list", "len": 3}}
v = L.check_constraints(
    {"response": '{"name": "rex", "age": 3, "hobbies": ["a", "b", "c"]}'},
    {"checks": [{"type": "json", "shape": shape}]})
assert v["passed"], v

import os, subprocess, tempfile

# --- check_file (agentic): verify a file the agent created in scratch ---
with tempfile.TemporaryDirectory() as wd:
    with open(os.path.join(wd, "pong.txt"), "w") as f:
        f.write("pong")
    assert L.check_file({"scratch": wd}, {"path": "pong.txt",
                                          "content": "pong"})["passed"]
    assert not L.check_file({"scratch": wd}, {"path": "pong.txt",
                                              "content": "ping"})["passed"]
    assert not L.check_file({"scratch": wd}, {"path": "missing.txt",
                                              "content": "x"})["passed"]

# --- check_run (agentic): re-run a command in scratch, all tests must pass ---
with tempfile.TemporaryDirectory() as wd:
    with open(os.path.join(wd, "ok.py"), "w") as f:
        f.write("import json,sys\n"
                "print(json.dumps({'total':2,'passed':2,'failed':[]}))\n")
    v = L.check_run({"scratch": wd},
                    {"cmd": ["python3", "ok.py"], "timeout": 30})
    assert v["passed"], v
with tempfile.TemporaryDirectory() as wd:
    with open(os.path.join(wd, "bad.py"), "w") as f:
        f.write("import json\n"
                "print(json.dumps({'total':2,'passed':1,'failed':['x']}))\n")
    assert not L.check_run({"scratch": wd},
                           {"cmd": ["python3", "bad.py"],
                            "timeout": 30})["passed"]

# --- R7: sentence counter must see a final sentence ending in a quote ---
v = L.check_constraints(
    {"response": 'Der Leuchtturm leuchtete. "Alles ist still."'},
    {"checks": [{"type": "sentences", "count": 2}]})
assert v["passed"], v
# a decimal point (23.30) must NOT count as a sentence end
v = L.check_constraints(
    {"response": "Es war 23.30 Uhr."},
    {"checks": [{"type": "sentences", "count": 1}]})
assert v["passed"], v

# --- R8: trailing punctuation tolerance in answer checks ---
assert L.check_answer({"response": "ANSWER: B."},
                       {"value": "B", "compact": True})["passed"]
assert L.check_answer({"response": "ANSWER: A1B2, C3D4, E5F6."},
                       {"value": "A1B2, C3D4, E5F6", "compact": True})["passed"]
assert not L.check_answer({"response": "ANSWER: A"},
                           {"value": "B", "compact": True})["passed"]

# --- R10: script constraint (response must be in the target script) ---
hi5 = "नमस्ते दुनिया में आपका स्वागत"
en5 = "hello world from devanagari land"
v = L.check_constraints({"response": hi5},
                        {"checks": [{"type": "script",
                                     "char_class": "[\\u0900-\\u097F]",
                                     "min_pct": 0.5}]})
assert v["passed"], v
v = L.check_constraints({"response": en5},
                        {"checks": [{"type": "script",
                                     "char_class": "[\\u0900-\\u097F]",
                                     "min_pct": 0.5}]})
assert not v["passed"], v
assert any(c["type"] == "script" for c in
           L.SUITES["mullang"][7]["verify"]["checks"]) or \
       any(c.get("type") == "script" for c in
           {t["id"]: t for t in L.SUITES["mullang"]}["mullang/hi-words"]
           ["verify"]["checks"])

# --- R9: task filter implies its suite (longctx reachable without --suites) ---
import llmbench as _L
cfg = _L.RunConfig(backend="api", model="qwen3.8",
                   endpoint="http://127.0.0.1:8000/v1", api_key="k",
                   suites=[], tasks=["longctx/needle-8k"], repeat=1,
                   max_tokens=4096, temperature=0.0, timeout=600,
                   sample_interval=2.0, retries=1, concurrency=1,
                   warmup=False, npu_container=None,
                   dsh_api_key_env="X", codex_provider="vllm")
ids = [t["id"] for t in _L.selected_tasks(cfg)[0]]
assert "longctx/needle-8k" in ids, ids
cfg2 = _L.RunConfig(backend="api", model="qwen3.8",
                    endpoint="http://127.0.0.1:8000/v1", api_key="k",
                    suites=[], tasks=[], repeat=1, max_tokens=4096,
                    temperature=0.0, timeout=600, sample_interval=2.0,
                    retries=1, concurrency=1, warmup=False, npu_container=None,
                    dsh_api_key_env="X", codex_provider="vllm")
all_ids = {t["id"] for t in _L.selected_tasks(cfg2)[0]}
assert not any(i.startswith("longctx/") for i in all_ids)

# --- mullang: ground truths ---
ml = {t["id"]: t for t in L.SUITES["mullang"]}
assert len(ml) == 8
assert ml["mullang/zh-arith"]["verify"]["value"] == 3 + 5 * 4
assert ml["mullang/ja-arith"]["verify"]["value"] == 7 + 8
assert ml["mullang/ko-arith"]["verify"]["value"] == 6 + 7
assert ml["mullang/ar-math"]["verify"]["value"] == 7 + 8
assert ml["mullang/es-answer"]["verify"]["value"] == "B"
assert ml["mullang/ru-grid"]["verify"]["expect_len"] == L._GRID_LEN
assert ml["mullang/ru-grid"]["verify"]["grid"] == L.GRID_B
# devanagari word counting (hi-words verifier path)
import re as _re
assert len(_re.findall(r"\S+", "नमस्ते दुनिया में आपका स्वागत")) == 5

# --- longctx: ground-truth sanity (codes appear exactly once, in order) ---
for t in L.SUITES["longctx"]:
    needles = [ln for ln in t["prompt"].split("\n") if ln.startswith("MAGIC CODE:")]
    codes = [ln.split("MAGIC CODE: ")[1].split(" ")[0] for ln in needles]
    assert len(needles) == len(set(needles)), t["id"]
    exp = t["verify"]["value"]
    if "," in exp:
        assert codes == exp.split(", "), (t["id"], codes, exp)
    else:
        assert codes == [exp], (t["id"], codes, exp)
assert all(t.get("backends") == ["api", "pi"] for t in L.SUITES["longctx"])
assert "longctx" in L.DEFAULT_EXCLUDE

# --- CI helper ---
assert L.stats_ci95([1.0, 2.0, 3.0]) is not None
assert L.stats_ci95([1.0]) is None
assert L.stats_ci95([]) is None
assert L._fmt_ci(5.0, 0.25) == "5.0 ±0.25"
assert L._fmt_ci(5.0, None) == "5.0"

# --- agentic suite wiring: CLI-only filter + setup files ---
ag = {t["id"]: t for t in L.SUITES["agentic"]}
assert len(ag) == 5
assert all(t.get("backends") == L.CLI_BACKENDS for t in ag.values())
assert "data.txt" in ag["agentic/data-sum"].get("setup", {})
assert "tools" in {t["id"].split("/")[0] for t in L.SUITES["tools"]}
assert all(t.get("backends") == ["api"] for t in L.SUITES["tools"])

# --- backend characterization probes (offline: no model, no network) ---------
# build() argv/env/overlay snapshots for every CLI backend + pi parse/finalize
# fixtures. These double as the regression net for the --pi-tools/
# --pi-thinking dialect args (hardening phases 3-4).
import json as _json
import os as _os
import tempfile as _tmp


def _bcfg(backend, **kw):
    base = dict(backend=backend, model="qwen3.8",
                endpoint="http://127.0.0.1:8000/v1", api_key="llmbench",
                suites=[], tasks=[], repeat=1, max_tokens=4096,
                temperature=0.0, timeout=600.0, sample_interval=2.0,
                retries=1, concurrency=1, warmup=True, npu_container=None,
                dsh_api_key_env="MY_GATEWAY_API_KEY", codex_provider="llmbench")
    base.update(kw)
    return L.RunConfig(**base)


BTASK = L.task("probe/x", "Say pong.", {"type": "exact", "value": "pong"})
BTASK_SYS = L.task("probe/y", "Say pong.", {"type": "exact", "value": "pong"},
                   system="You are terse.")

# --- pi: build + parse_line + finalize --------------------------------------
with _tmp.TemporaryDirectory() as scratch:
    b = L.PiBackend(_bcfg("pi"))
    cmd, env = b.build(BTASK_SYS, scratch)
    assert cmd == ["pi", "--mode", "json", "--no-session", "--provider",
                   "llmbench", "--model", "qwen3.8", "--api-key", "llmbench",
                   "--append-system-prompt", "You are terse.", "--",
                   "Say pong."], cmd
    assert env["PI_CODING_AGENT_DIR"] == _os.path.join(scratch, ".pi-agent")
    p = _json.load(open(_os.path.join(scratch, ".pi-agent", "models.json")))
    p = p["providers"]["llmbench"]
    assert p["baseUrl"] == "http://127.0.0.1:8000/v1"
    assert p["api"] == "openai-completions" and p["apiKey"] == "llmbench"
    assert p["models"][0]["id"] == "qwen3.8"
    # maxTokens mirrors --max-tokens (parity with the api backend)
    assert p["models"][0]["maxTokens"] == 4096
    _cmdm, _ = L.PiBackend(_bcfg("pi", max_tokens=8192)).build(BTASK, scratch)
    assert (_json.load(open(_os.path.join(scratch, ".pi-agent", "models.json")))
            ["providers"]["llmbench"]["models"][0]["maxTokens"] == 8192)
    # thinking declaration: absent by default (wire unchanged); present when
    # --pi-thinking is set, so pi maps the level to enable_thinking
    assert "reasoning" not in p["models"][0]
    assert "compat" not in p["models"][0]
    _cmdh, _envh = L.PiBackend(_bcfg("pi", pi_thinking="off")).build(BTASK, scratch)
    mh = _json.load(open(_os.path.join(scratch, ".pi-agent", "models.json")))
    assert mh["providers"]["llmbench"]["models"][0]["reasoning"] is True
    assert (mh["providers"]["llmbench"]["models"][0]
            ["compat"]["thinkingFormat"] == "qwen-chat-template")
    assert "--tools" not in cmd  # default: no pinning
    cmd2, _ = b.build(BTASK, scratch)  # no system -> no append-system-prompt
    assert "--append-system-prompt" not in cmd2 and cmd2[-2:] == ["--", "Say pong."]
    # --pi-tools pinning: verbatim, placed before the -- prompt separator
    cmdt, _ = L.PiBackend(_bcfg("pi", pi_tools="read,bash")).build(BTASK_SYS, scratch)
    assert cmdt[-6:] == ["--append-system-prompt", "You are terse.",
                         "--tools", "read,bash", "--", "Say pong."], cmdt[-6:]
    # --pi-thinking pinning: absent by default, verbatim when set
    assert "--thinking" not in cmd
    cmdh, _ = L.PiBackend(_bcfg("pi", pi_thinking="off")).build(BTASK_SYS, scratch)
    assert cmdh[-4:] == ["--thinking", "off", "--", "Say pong."], cmdh[-4:]
    # thinking capture: message_end thinking block is authoritative, with the
    # thinking_delta buffer as fallback
    o6 = L.TaskOutcome()
    for e in [
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        {"type": "message_update",
         "assistantMessageEvent": {"type": "thinking_delta",
                                    "contentIndex": 0, "delta": "let me "}},
        {"type": "message_update",
         "assistantMessageEvent": {"type": "thinking_delta",
                                    "contentIndex": 0, "delta": "think"}},
        {"type": "message_end", "message": {
            "role": "assistant", "stopReason": "stop",
            "usage": {"input": 10, "output": 5, "totalTokens": 15},
            "content": [{"type": "thinking", "thinking": "let me think"},
                        {"type": "text", "text": "42"}]}},
    ]:
        b.parse_line(o6, _json.dumps(e))
    assert o6.reasoning == "let me think", o6.reasoning
    assert o6.text == "42" and (o6.prompt_tokens, o6.completion_tokens) == (10, 5)
    o7 = L.TaskOutcome()  # no thinking block in message_end -> delta fallback
    for e in [
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        {"type": "message_update",
         "assistantMessageEvent": {"type": "thinking_delta",
                                    "contentIndex": 0, "delta": "abc"}},
        {"type": "message_end", "message": {
            "role": "assistant", "stopReason": "stop",
            "content": [{"type": "text", "text": "x"}]}},
    ]:
        b.parse_line(o7, _json.dumps(e))
    assert o7.reasoning == "abc", o7.reasoning
    o8 = L.TaskOutcome()  # no thinking at all -> reasoning stays empty
    for e in [
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        {"type": "message_end", "message": {
            "role": "assistant", "stopReason": "stop",
            "content": [{"type": "text", "text": "y"}]}},
    ]:
        b.parse_line(o8, _json.dumps(e))
    assert o8.reasoning == ""
    # stdin hook: base backends never feed stdin; pi pipes only >96k prompts
    # (the per-argument limit), and then argv must not repeat prompt/system
    assert L.DshBackend(_bcfg("dsh")).stdin_for(BTASK) is None
    big = {"prompt": "x" * 100_000, "system": "sys"}
    cmdl, _ = L.PiBackend(_bcfg("pi")).build(big, scratch)
    assert "--" not in cmdl and "--append-system-prompt" not in cmdl, cmdl
    assert L.PiBackend(_bcfg("pi")).stdin_for(big) is not None
    assert L.PiBackend(_bcfg("pi")).stdin_for(BTASK_SYS) is None

    o = L.TaskOutcome()
    evs = [
        {"type": "session", "version": 3, "id": "u", "cwd": scratch},
        {"type": "agent_start"},
        {"type": "message_start", "message": {"role": "user"}},
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        {"type": "message_update",
         "usage": {"input": 120, "output": 4, "totalTokens": 124},
         "assistantMessageEvent": {"type": "text_delta", "delta": "pong"}},
        {"type": "message_end", "message": {
            "role": "assistant", "stopReason": "stop",
            "usage": {"input": 120, "output": 4, "totalTokens": 124},
            "content": [{"type": "text", "text": "pong "}]}},
        {"type": "turn_end", "message": {"role": "assistant"},
         "toolResults": [{"toolCallId": "c1", "toolName": "bash",
                           "isError": False}]},
        {"type": "agent_end", "messages": [], "willRetry": False},
        {"type": "agent_settled", "aborted": False},
        "not json, ignore",
    ]
    for e in evs:
        b.parse_line(o, _json.dumps(e) if not isinstance(e, str) else e)
    assert o.text == "pong " and o.finish_reason == "stop"
    assert (o.prompt_tokens, o.completion_tokens) == (120, 4)
    assert o.token_source == "server" and o.ttft_model_s is not None
    names = [e["event"] for e in o.events]
    assert names == ["assistant_message_end", "tool_result", "agent_end",
                     "agent_settled"], names
    assert o.events[1]["toolName"] == "bash"

    o2 = L.TaskOutcome()  # multi-turn usage summing
    for e in [evs[3], evs[4], evs[5],
              {"type": "message_start", "message": {"role": "assistant"}},
              {"type": "message_update",
               "usage": {"input": 340, "output": 20, "totalTokens": 360}},
              {"type": "message_end", "message": {
                  "role": "assistant", "stopReason": "stop",
                  "content": [{"type": "text", "text": "final"}]}}]:
        b.parse_line(o2, _json.dumps(e))
    assert (o2.prompt_tokens, o2.completion_tokens) == (460, 24)

    o3 = L.TaskOutcome()  # estimate fallback (no usage reported)
    o3.text, o3.exit_code, b._prompt = "pong", 0, "Say pong."
    b.finalize(o3)
    assert o3.token_source == "estimate" and o3.prompt_tokens is not None
    o3b = L.TaskOutcome()  # per-outcome estimate basis (concurrency-safe path)
    o3b.text, o3b.exit_code = "pong", 0
    o3b._est_prompt = "x" * 400
    b.finalize(o3b)
    assert o3b.prompt_tokens == 100 and o3b.token_source == "estimate"

    o4 = L.TaskOutcome()  # nonzero exit, no text
    o4.exit_code, o4.stderr_tail = 3, "boom" * 100
    b.finalize(o4)
    assert o4.error and o4.error.startswith("pi exit 3:"), o4.error
    o5 = L.TaskOutcome()  # zero exit, no assistant message
    o5.exit_code = 0
    b.finalize(o5)
    assert o5.error == "no assistant message received", o5.error

# --- dsh: build (patch overlay + argv/env) ---------------------------------
with _tmp.TemporaryDirectory() as scratch:
    b = L.DshBackend(_bcfg("dsh"))
    cmd, env = b.build(BTASK_SYS, scratch)
    assert cmd[:4] == ["dsh", "--profile", "headless", "--patch"], cmd[:4]
    patch = cmd[4]
    assert _os.path.dirname(patch) == scratch
    assert cmd[5] == "You are terse.\n\nSay pong."  # system concatenated
    ytxt = open(patch).read()
    for frag in ("baseURL: http://127.0.0.1:8000/v1", "api: openai-completions",
                 "apiKeyEnv: MY_GATEWAY_API_KEY", "id: qwen3.8",
                 "provider: llmbench"):
        assert frag in ytxt, frag
    assert env["MY_GATEWAY_API_KEY"] == "llmbench"

# --- codex: build (exec flags + provider overrides) --------------------------
with _tmp.TemporaryDirectory() as scratch:
    b = L.CodexBackend(_bcfg("codex"))
    cmd, env = b.build(BTASK_SYS, scratch)
    assert cmd[:6] == ["codex", "exec", "--json", "--skip-git-repo-check",
                       "--ephemeral", "-C"], cmd[:6]
    assert cmd[6] == scratch and cmd[7:9] == ["-m", "qwen3.8"]
    assert cmd[9:11] == ["-c", 'model_provider="llmbench"']
    assert cmd[11:13] == ["-c", 'model_providers.llmbench.base_url="http://127.0.0.1:8000/v1"']
    assert cmd[13:15] == ["-c", 'model_providers.llmbench.name="llmbench"']
    assert cmd[15] == "--dangerously-bypass-approvals-and-sandbox"
    assert cmd[16] == "You are terse.\n\nSay pong."  # system concatenated
    assert env["OPENAI_API_KEY"] == "llmbench"

# --- claude: build + after_run (hermetic HOME; real ~/.claude.json untouched)
_old_home = _os.environ.get("HOME")
try:
    with _tmp.TemporaryDirectory() as fakehome, \
            _tmp.TemporaryDirectory() as scratch:
        _os.environ["HOME"] = fakehome
        b = L.ClaudeBackend(_bcfg("claude"))
        cmd, env = b.build(BTASK_SYS, scratch)
        assert cmd[:7] == ["claude", "-p", "--output-format", "stream-json",
                           "--verbose", "--model", "qwen3.8"], cmd[:7]
        assert cmd[7] == "--no-session-persistence"
        assert cmd[8] == "Say pong."  # prompt; no-system variant below
        assert cmd[9:11] == ["--append-system-prompt", "You are terse."]
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "llmbench"
        assert _json.load(open(_os.path.join(scratch, ".claude", "settings.json"))) \
            ["permissions"]["allow"] == ["Bash", "Read", "Write", "Edit"]
        trust = _os.path.join(fakehome, ".claude.json")
        assert _json.load(open(trust))["projects"][scratch] \
            ["hasTrustDialogAccepted"] is True
        b.after_run(scratch)  # trust entry removed again
        assert scratch not in _json.load(open(trust)).get("projects", {})
        cmd2, _ = L.ClaudeBackend(_bcfg("claude")).build(BTASK, scratch)
        assert "--append-system-prompt" not in cmd2
finally:
    if _old_home is None:
        _os.environ.pop("HOME", None)
    else:
        _os.environ["HOME"] = _old_home

# --- api: _apply_delta (reasoning alias + tool accumulation), _est_if_missing
# (token-estimate fallback), preflight_report (/models parse) ----------------
cr = L.ChatResult()
acc = {}
L.ApiBackend("http://x/v1", "m")._apply_delta(
    cr, {"delta": {"reasoning_content": "th "}}, 0.0, acc)
assert cr.reasoning == "th " and cr.reasoning_chunks == 1
assert cr.ttft_first_reasoning_s is not None
crb = L.ChatResult()
L.ApiBackend("http://x/v1", "m")._apply_delta(
    crb, {"delta": {"reasoning": "x"}, "finish_reason": "stop"}, 0.0, {})
assert crb.reasoning == "x" and crb.finish_reason == "stop"
crt = L.ChatResult()  # tool-call accumulation across deltas
accd = {}
L.ApiBackend("http://x/v1", "m")._apply_delta(
    crt, {"delta": {"tool_calls": [
        {"index": 0, "id": "c1", "function": {"name": "get_", "arguments": "{\"a\":"}}]}},
    0.0, accd)
L.ApiBackend("http://x/v1", "m")._apply_delta(
    crt, {"delta": {"tool_calls": [
        {"index": 0, "function": {"name": "weather", "arguments": " 1}"}}]}},
    0.0, accd)
assert accd[0] == {"id": "c1", "name": "get_weather", "args": '{"a": 1}'}

cr_est = L.ChatResult()  # no usage -> same ~4 chars/token estimate as CLIs
assert L.ApiBackend._est_if_missing(cr_est, "x" * 400) == "estimate"
assert (cr_est.prompt_tokens, cr_est.completion_tokens,
        cr_est.total_tokens) == (100, 1, 101)
cr_srv = L.ChatResult()  # usage present -> server; total patched if absent
cr_srv.prompt_tokens, cr_srv.completion_tokens = 10, 20
assert L.ApiBackend._est_if_missing(cr_srv, "x") == "server"
assert cr_srv.total_tokens == 30

assert L.preflight_report('{"data": [{"id": "m1"}]}', "m1") is None
assert "not in /models list" in L.preflight_report('{"data": [{"id": "m1"}]}', "m2")
assert "non-OpenAI" in L.preflight_report("not json", "m")
assert "non-OpenAI" in L.preflight_report('{"data": "nope"}', "m1")

# --- api request body: temperature is OMITTED unless explicitly set ----------
# Default None + omit-the-key (never null) is what makes the harness measure
# the endpoint's configured sampling, and what lets reasoning endpoints that
# reject the parameter answer at all.
_ab = L.ApiBackend("http://127.0.0.1:1/v1", "m")
assert _ab.temperature is None
_body = _ab.request_body([{"role": "user", "content": "x"}])
assert "temperature" not in _body, _body
assert _body["max_tokens"] == 4096 and _body["stream_options"]
assert _ab.request_body([{"role": "user", "content": "x"}],
                        tools=[{"type": "function"}])["tools"]
for _tv in (0.0, 0.7):  # 0.0 must still be sent (greedy baselines)
    _b = L.ApiBackend("http://127.0.0.1:1/v1", "m", temperature=_tv)\
        .request_body([{"role": "user", "content": "x"}])
    assert _b["temperature"] == _tv, _b

# --- retry policy: only transient/transport shapes are retried ---------------
assert not L.retryable_status(400)      # unsupported param / context too long
assert not L.retryable_status(404) and not L.retryable_status(422)
assert L.retryable_status(408) and L.retryable_status(429)
assert L.retryable_status(500) and L.retryable_status(503)

# --- longctx: token-based sizing, 128k-window fit, and the context gate ------
_lc = {t["id"]: t for t in L.SUITES["longctx"]}
assert all(t.get("min_context_tokens") for t in _lc.values())
# the conservative divisor (3.8) must never understate what chars//4 claims
for _id, _t in _lc.items():
    assert _t["min_context_tokens"] >= len(_t["prompt"]) // 4, _id
# top tier + the default output budget fits both 128000 and 131072 windows
_heavy = _lc["longctx/needle-128k"]["min_context_tokens"] + 4096
assert _heavy <= 128_000, (_heavy, "128k-class models must be able to run it")
assert len(_lc["longctx/needle-128k"]["prompt"]) < 512_000


def _scfg(backend="api", context_tokens=0, max_tokens=4096, **kw):
    base = dict(backend=backend, model="m",
                endpoint="http://127.0.0.1:1/v1", api_key="k",
                suites=["longctx"], tasks=[], repeat=1, max_tokens=max_tokens,
                temperature=None, timeout=600.0, sample_interval=2.0,
                retries=1, concurrency=1, warmup=False, npu_container=None,
                dsh_api_key_env="X", codex_provider="vllm")
    base.update(kw)
    return L.RunConfig(context_tokens=context_tokens, **base)


# window unknown (default): everything is sent, exactly as before
_ids = [t["id"] for t in L.selected_tasks(_scfg())[0]]
assert _ids == list(_lc), _ids
# 128k-class windows: all three tiers still fit (incl. pi's ~3.1k scaffolding)
for _ctx in (128_000, 131_072, 262_144):
    for _bk in ("api", "pi"):
        _ids = [t["id"] for t in L.selected_tasks(
            _scfg(_bk, context_tokens=_ctx))[0]]
        assert _ids == list(_lc), (_bk, _ctx, _ids)
# smaller windows: only the over-sized tiers are skipped
_sel, _sk, _big = L.selected_tasks(_scfg("api", context_tokens=65_536))
assert [t["id"] for t in _sel] == ["longctx/needle-8k", "longctx/multi-32k"]
assert [d["id"] for d in _big] == ["longctx/needle-128k"], _big
# a skip carries its own arithmetic, so the reason is legible without the code
_d = _big[0]
assert _d["prompt_est"] == _lc["longctx/needle-128k"]["min_context_tokens"]
assert _d["need"] == _d["prompt_est"] + _d["max_tokens"] + _d["scaffold_est"]
assert _d["scaffold_est"] == 0 and _d["declared_context"] == 65_536
assert _d["shortfall"] == _d["need"] - 65_536 > 0
assert L.context_shortfall(_scfg("api", context_tokens=65_536),
                           _lc["longctx/needle-128k"]) == _d["shortfall"]
_sel, _sk, _big = L.selected_tasks(_scfg("api", context_tokens=32_768))
assert [t["id"] for t in _sel] == ["longctx/needle-8k"], [t["id"] for t in _sel]
assert [d["id"] for d in _big] == ["longctx/multi-32k", "longctx/needle-128k"], _big
_sel, _sk, _big = L.selected_tasks(_scfg("api", context_tokens=12_288))
assert _sel == [] and [d["id"] for d in _big] == list(_lc), (len(_sel), _big)
# pi's scaffolding raises the bar a little
_sel, _sk, _big = L.selected_tasks(_scfg("pi", context_tokens=12_288))
assert _sel == [] and [d["id"] for d in _big] == list(_lc), (len(_sel), _big)
assert all(d["scaffold_est"] == L._PI_SCAFFOLD_TOKENS for d in _big)
# a larger --max-tokens reservation raises the bar accordingly
assert L.context_shortfall(_scfg("api", context_tokens=128_000, max_tokens=8_192),
                           _lc["longctx/needle-128k"]) > 0
assert L.context_shortfall(_scfg("api", context_tokens=128_000),
                           _lc["longctx/needle-128k"]) == 0
# pi pays for its own scaffolding
assert L.context_shortfall(_scfg("pi", context_tokens=128_000),
                           _lc["longctx/needle-128k"]) == 0
assert L.context_shortfall(_scfg("pi", context_tokens=125_000),
                           _lc["longctx/needle-128k"]) > 0

# pi declares the real window in its generated models.json (was 262144 always)
with _tmp.TemporaryDirectory() as scratch:
    L.PiBackend(_bcfg("pi")).build(BTASK, scratch)
    _m = _json.load(open(_os.path.join(scratch, ".pi-agent", "models.json")))
    assert _m["providers"]["llmbench"]["models"][0]["contextWindow"] == 262144
    L.PiBackend(_bcfg("pi", context_tokens=131_072)).build(BTASK, scratch)
    _m = _json.load(open(_os.path.join(scratch, ".pi-agent", "models.json")))
    assert _m["providers"]["llmbench"]["models"][0]["contextWindow"] == 131_072

# --- strict vs content: a decorated correct answer is still a strict fail ----
def _v(ty, spec, resp):
    return L.VERIFIERS[ty]({"response": resp, "tool_calls": [], "task_id": "x",
                            "out_dir": "/tmp", "scratch": "/tmp"}, spec)

_M = {"value": 408}
for _r, _strict, _content in [
        ("ANSWER: 408", True, True),
        ("ANSWER: **408**", False, True),        # markdown emphasis
        ("ANSWER: 408 items", False, True),      # trailing unit
        ("ANSWER: The answer is 408.", False, True),
        ("ANSWER: 409", False, False),           # genuinely wrong
        ("ANSWER: 1,408", False, False),         # different number, not a separator
        ("ANSWER: 1408", False, False)]:
    _v_ = _v("math", _M, _r)
    assert bool(_v_["passed"]) is _strict, (_r, _v_)
    assert bool(_v_["content_ok"]) is _content, (_r, _v_)

_A = {"value": "A", "compact": True}
for _r, _strict, _content in [
        ("ANSWER: A", True, True),
        ("ANSWER: A is the knight", False, True),        # the real-world flip
        ("ANSWER: **A**", False, True),
        ("ANSWER: the answer is A", False, True),
        ("ANSWER: B is the knight", False, False),
        ("ANSWER: 4A", False, False)]:
    _v_ = _v("answer", _A, _r)
    assert bool(_v_["passed"]) is _strict, (_r, _v_)
    assert bool(_v_["content_ok"]) is _content, (_r, _v_)

# ordered multi-value answers keep their order requirement in both readings
_O = {"value": "A1B2, C3D4, E5F6", "compact": True}
assert _v("answer", _O, "ANSWER: **A1B2, C3D4, E5F6**")["content_ok"] is True
assert _v("answer", _O, "ANSWER: E5F6, C3D4, A1B2")["content_ok"] is False

# whole-reply exact tasks: trailing punctuation passes strictly (the agreed
# smoke/pong change); prose around the value is a strict fail but does count as
# delivered -- content_ok means "the value appears", not "it was asserted"
assert _v("exact", {"value": "pong"}, "Pong.")["passed"] is True
assert _v("exact", {"value": "pong"}, "The word is pong")["passed"] is False
assert _v("exact", {"value": "pong"}, "The word is pong")["content_ok"] is True
assert _v("exact", {"value": "pong"}, "The word is ping")["content_ok"] is False
assert _v("exact", {"value": "pong"}, "Pong.")["content_ok"] is True

# move strings: strict needs a bare run, content accepts decoration around a
# valid shortest path (DDDDRRRR is the real answer for the task grid)
_G = {"grid": ["S....", ".###.", ".#.#.", ".###.", "....G"], "expect_len": 8}
assert _v("gridpath", _G, "ANSWER: DDDDRRRR")["passed"] is True
assert _v("gridpath", _G, "ANSWER: **DDDDRRRR**")["passed"] is False
assert _v("gridpath", _G, "ANSWER: **DDDDRRRR**")["content_ok"] is True
assert _v("gridpath", _G, "ANSWER: DDRR")["content_ok"] is False

# number scanning keeps whole numbers: 4+ digit runs must not split at 3 digits,
# and a thousands separator must not be read as two values
assert L._numbers_in("**5888**") == [5888.0]
assert L._numbers_in("1,408.50") == [1408.5]
assert L._numbers_in("total 5888 of 12") == [5888.0, 12.0]
assert L._value_present("ANSWER: 5888", "5888", numeric=True) is True
assert L._value_present("ANSWER: 1,408", "408", numeric=True) is False

# totals split: strict passes and delivered values are counted separately
def _tres(status, verdict):
    return {"id": status, "task_id": status, "suite": "s", "repeat": 0,
            "backend": "api", "status": status, "verdict": verdict, "error": None,
            "retryable": True, "truncated": False, "ttft_s": 1.0, "total_s": 2.0,
            "ttft_model_s": 1.0, "ttft_first_reasoning_s": None,
            "decode_tps": 5.0, "gen_tps": 5.0, "overall_tps": 5.0,
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
            "token_source": "server", "resources": {}}

_rs = [_tres("pass", {"passed": True}),
       _tres("fail", {"passed": False, "content_ok": True}),
       _tres("fail", {"passed": False, "content_ok": False})]
_tt = L.compute_totals(_rs)
assert _tt["n_pass"] == 1 and _tt["n_content"] == 2, _tt
assert _tt["suites"]["s"]["passed"] == 1 and _tt["suites"]["s"]["content"] == 2
assert _tt["suites"]["s"]["content_rate"] > _tt["suites"]["s"]["pass_rate"]

# --- verdict integrity: an empty answer is never a silent pass --------------
class _EmptyBackend:
    name = "stub"

    def __init__(self, outcome, writes=None):
        self._o = outcome
        self._writes = writes or {}

    def run(self, t, scratch, set_root):
        # stand in for an agent that produced files but no reply text
        for name, content in self._writes.items():
            with open(_os.path.join(scratch, name), "w") as f:
                f.write(content)
        return self._o


_ET = L.task("probe/empty", "hi", {"type": "exact", "value": "pong"})
with _tmp.TemporaryDirectory() as out_dir:
    # reasoning stream ate the output budget: a fail, with the reason recorded
    _o = L.TaskOutcome(finish_reason="length", reasoning_chunks=1024,
                       completion_tokens=4096)
    _r = L.run_one_task(_scfg(), _ET, _EmptyBackend(_o), L.NPUProber(None),
                        out_dir, 0, lambda s: None)
    assert _r["status"] == "fail", _r["status"]
    assert _r["verdict"] and not _r["verdict"]["passed"]
    assert "output budget" in _r["verdict"]["detail"], _r["verdict"]
    assert _r["truncated"] is True
    # truly empty response: an error (worth a retry), not a pass
    _o2 = L.TaskOutcome(finish_reason=None)
    _r2 = L.run_one_task(_scfg(), _ET, _EmptyBackend(_o2), L.NPUProber(None),
                         out_dir, 0, lambda s: None)
    assert _r2["status"] == "error" and not _r2.get("verdict"), _r2
    # a rejected request is flagged non-retryable so the retry loop skips it
    _o3 = L.TaskOutcome(error="HTTP 400: nope", retryable=False)
    _r3 = L.run_one_task(_scfg(), _ET, _EmptyBackend(_o3), L.NPUProber(None),
                         out_dir, 0, lambda s: None)
    assert _r3["status"] == "error" and _r3["retryable"] is False
    # a workspace verifier is scored even when the reply is empty: the agent
    # did the work, so the verdict is real (and a missing file is a fail, not
    # an "empty response" error)
    _ft = L.task("probe/file", "write pong.txt",
                 {"type": "file", "path": "pong.txt", "content": "pong"})
    _r4 = L.run_one_task(_scfg(), _ft, _EmptyBackend(L.TaskOutcome(),
                                                    {"pong.txt": "pong"}),
                         L.NPUProber(None), out_dir, 0, lambda s: None)
    assert _r4["status"] == "pass", (_r4["status"], _r4.get("error"))
    _r5 = L.run_one_task(_scfg(), _ft, _EmptyBackend(L.TaskOutcome()),
                         L.NPUProber(None), out_dir, 0, lambda s: None)
    assert _r5["status"] == "fail" and _r5["verdict"]["detail"] == "file missing", _r5

print("api parser probes OK")
print("backend characterization probes OK")
print("verifier self-tests OK")

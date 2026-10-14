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

print("backend characterization probes OK")
print("verifier self-tests OK")

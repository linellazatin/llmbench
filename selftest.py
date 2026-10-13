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

# --- agentic suite wiring: CLI-only filter + setup files ---
ag = {t["id"]: t for t in L.SUITES["agentic"]}
assert len(ag) == 5
assert all(t.get("backends") == ["dsh", "codex", "claude"] for t in ag.values())
assert "data.txt" in ag["agentic/data-sum"].get("setup", {})
assert "tools" in {t["id"].split("/")[0] for t in L.SUITES["tools"]}
assert all(t.get("backends") == ["api"] for t in L.SUITES["tools"])

print("verifier self-tests OK")

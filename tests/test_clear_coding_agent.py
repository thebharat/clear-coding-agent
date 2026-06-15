import io
import json
import urllib.error
import urllib.request
import pytest
from unittest.mock import patch

from clear_coding_agent import (
    ClearCodingAgent,
    FakeModelClient,
    OllamaModelClient,
    SessionStore,
    Tool,
    WorkspaceContext,
    _new_session,
    build_welcome,
    clip,
    main,
    middle,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".clear-coding-agent" / "sessions")
    approval_policy = kwargs.pop("approval_policy", "auto")
    return ClearCodingAgent(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy=approval_policy,
        **kwargs,
    )


def _ollama_available():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            data = json.loads(response.read())
            return any(m["name"] == "qwen3.5:4b" for m in data.get("models", []))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Utility helpers: clip and middle
# ---------------------------------------------------------------------------

def test_clip_leaves_short_text_unchanged():
    assert clip("hello", 10) == "hello"


def test_clip_truncates_long_text():
    result = clip("abcde", 3)
    assert result.startswith("abc")
    assert "truncated" in result


def test_clip_converts_non_string_input():
    assert clip(42, 100) == "42"


def test_middle_leaves_short_text_unchanged():
    assert middle("hello", 10) == "hello"


def test_middle_truncates_from_centre():
    # middle("abcdefghij", 7) → "ab...ij" (ellipsis in the middle, not the end)
    result = middle("abcdefghij", 7)
    assert len(result) == 7
    assert "..." in result
    assert result.startswith("ab")
    assert result.endswith("ij")


def test_middle_replaces_newlines_with_spaces():
    assert "\n" not in middle("a\nb", 100)


# ---------------------------------------------------------------------------
# _new_session helper
# ---------------------------------------------------------------------------

def test_new_session_has_required_fields():
    session = _new_session("/some/root")
    assert "id" in session
    assert "created_at" in session
    assert session["workspace_root"] == "/some/root"
    assert session["history"] == []
    assert session["memory"] == {"task": "", "files": [], "notes": []}


def test_new_session_generates_unique_ids():
    ids = {_new_session("/root")["id"] for _ in range(10)}
    assert len(ids) == 10


# ---------------------------------------------------------------------------
# Tool dataclass
# ---------------------------------------------------------------------------

def test_tool_dataclass_is_immutable():
    tool = Tool(schema={"path": "str"}, description="desc", risky=False, run=lambda _: "ok")
    with pytest.raises(Exception):
        tool.risky = True  # frozen dataclass must reject mutation


# ---------------------------------------------------------------------------
# remember() static method
# ---------------------------------------------------------------------------

def test_remember_appends_new_entry():
    items = []
    ClearCodingAgent.remember(items, "a", 5)
    assert items == ["a"]


def test_remember_promotes_existing_entry_to_end():
    items = ["a", "b", "c"]
    ClearCodingAgent.remember(items, "a", 5)
    assert items == ["b", "c", "a"]


def test_remember_enforces_max_size():
    items = ["a", "b", "c"]
    ClearCodingAgent.remember(items, "d", 3)
    assert len(items) == 3
    assert "d" in items
    assert "a" not in items


def test_remember_ignores_empty_entry():
    items = ["a"]
    ClearCodingAgent.remember(items, "", 5)
    assert items == ["a"]


# ---------------------------------------------------------------------------
# memory_text()
# ---------------------------------------------------------------------------

def test_memory_text_shows_task_files_and_notes(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.session["memory"]["task"] = "do stuff"
    agent.session["memory"]["files"] = ["foo.py", "bar.py"]
    agent.session["memory"]["notes"] = ["saw a bug"]

    text = agent.memory_text()

    assert "do stuff" in text
    assert "foo.py" in text
    assert "bar.py" in text
    assert "saw a bug" in text


def test_memory_text_shows_dash_for_empty_task(tmp_path):
    agent = build_agent(tmp_path, [])
    assert "task: -" in agent.memory_text()


# ---------------------------------------------------------------------------
# history_text() edge cases
# ---------------------------------------------------------------------------

def test_history_text_returns_empty_sentinel_when_no_history(tmp_path):
    agent = build_agent(tmp_path, [])
    assert agent.history_text() == "- empty"


def test_history_text_compresses_older_tool_entries(tmp_path):
    agent = build_agent(tmp_path, [])
    long_content = "x" * 500
    for i in range(10):
        agent.record({"role": "tool", "name": "list_files", "args": {}, "content": long_content, "created_at": str(i)})

    history = agent.history_text()

    # Older entries use 180-char limit so the full 500-char string must not appear verbatim
    # in those positions — just check overall size is under MAX_HISTORY_CHARS
    assert len(history) <= 12000


# ---------------------------------------------------------------------------
# parse() static method
# ---------------------------------------------------------------------------

def test_parse_extracts_json_tool_call():
    response_type, parsed = ClearCodingAgent.parse('<tool>{"name":"list_files","args":{"path":"."}}</tool>')
    assert response_type == "tool"
    assert parsed["name"] == "list_files"
    assert parsed["args"] == {"path": "."}


def test_parse_returns_final_for_final_tag():
    response_type, parsed = ClearCodingAgent.parse("<final>Done.</final>")
    assert response_type == "final"
    assert parsed == "Done."


def test_parse_returns_final_for_bare_text():
    response_type, parsed = ClearCodingAgent.parse("This is a plain answer.")
    assert response_type == "final"
    assert parsed == "This is a plain answer."


def test_parse_returns_retry_for_empty_string():
    response_type, _ = ClearCodingAgent.parse("")
    assert response_type == "retry"


def test_parse_returns_retry_for_malformed_tool_json():
    response_type, msg = ClearCodingAgent.parse("<tool>not json</tool>")
    assert response_type == "retry"
    assert "malformed" in msg


def test_parse_returns_retry_for_empty_final_tag():
    response_type, _ = ClearCodingAgent.parse("<final></final>")
    assert response_type == "retry"


def test_parse_prefers_tool_when_tool_comes_before_final():
    response = '<tool>{"name":"list_files","args":{}}</tool> <final>answer</final>'
    response_type, _ = ClearCodingAgent.parse(response)
    assert response_type == "tool"


def test_parse_prefers_final_when_final_comes_before_tool():
    response = '<final>answer</final> <tool>{"name":"list_files","args":{}}</tool>'
    response_type, _ = ClearCodingAgent.parse(response)
    assert response_type == "final"


def test_parse_sets_args_to_empty_dict_when_null():
    response_type, parsed = ClearCodingAgent.parse('<tool>{"name":"list_files","args":null}</tool>')
    assert response_type == "tool"
    assert parsed["args"] == {}


# ---------------------------------------------------------------------------
# parse_xml_tool() and parse_attrs()
# ---------------------------------------------------------------------------

def test_parse_xml_tool_extracts_name_path_and_content():
    response = '<tool name="write_file" path="foo.py"><content>print("hi")\n</content></tool>'
    parsed = ClearCodingAgent.parse_xml_tool(response)
    assert parsed is not None
    assert parsed["name"] == "write_file"
    assert parsed["args"]["path"] == "foo.py"
    assert parsed["args"]["content"] == 'print("hi")\n'


def test_parse_xml_tool_returns_none_when_name_missing():
    result = ClearCodingAgent.parse_xml_tool('<tool path="foo.py"><content>x</content></tool>')
    assert result is None


def test_parse_xml_tool_falls_back_to_body_as_content():
    response = '<tool name="write_file" path="foo.py">print("hi")\n</tool>'
    parsed = ClearCodingAgent.parse_xml_tool(response)
    assert parsed is not None
    assert 'print("hi")' in parsed["args"]["content"]


def test_parse_attrs_handles_double_and_single_quotes():
    attrs = ClearCodingAgent.parse_attrs(' name="write_file" path=\'foo.py\'')
    assert attrs["name"] == "write_file"
    assert attrs["path"] == "foo.py"


# ---------------------------------------------------------------------------
# approve()
# ---------------------------------------------------------------------------

def test_approve_returns_false_in_read_only_mode(tmp_path):
    agent = build_agent(tmp_path, [], read_only=True)
    assert agent.approve("write_file", {}) is False


def test_approve_returns_true_with_auto_policy(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="auto")
    assert agent.approve("write_file", {}) is True


def test_approve_returns_false_with_never_policy(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="never")
    assert agent.approve("write_file", {}) is False


def test_approve_returns_false_on_eof(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="ask")
    with patch("builtins.input", side_effect=EOFError):
        assert agent.approve("write_file", {}) is False


def test_approve_returns_true_on_yes_input(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="ask")
    with patch("builtins.input", return_value="y"):
        assert agent.approve("write_file", {}) is True


# ---------------------------------------------------------------------------
# note_tool()
# ---------------------------------------------------------------------------

def test_note_tool_tracks_read_file_path(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.note_tool("read_file", {"path": "foo.py"}, "content")
    assert "foo.py" in agent.session["memory"]["files"]


def test_note_tool_does_not_track_path_for_run_shell(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.note_tool("run_shell", {"command": "ls"}, "output")
    assert agent.session["memory"]["files"] == []


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_clears_history_and_memory(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.record({"role": "user", "content": "hello", "created_at": "1"})
    agent.session["memory"]["task"] = "something"

    agent.reset()

    assert agent.session["history"] == []
    assert agent.session["memory"]["task"] == ""
    assert agent.session["memory"]["files"] == []
    assert agent.session["memory"]["notes"] == []


# ---------------------------------------------------------------------------
# tool_list_files()
# ---------------------------------------------------------------------------

def test_list_files_shows_dirs_before_files(tmp_path):
    (tmp_path / "z_dir").mkdir()
    (tmp_path / "a_file.txt").write_text("x", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("list_files", {"path": "."})

    dir_pos = result.index("[D] z_dir")
    file_pos = result.index("[F] a_file.txt")
    assert dir_pos < file_pos


def test_list_files_returns_empty_sentinel_for_empty_dir(tmp_path):
    subdir = tmp_path / "empty"
    subdir.mkdir()
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("list_files", {"path": "empty"})
    assert result == "(empty)"


# ---------------------------------------------------------------------------
# tool_search() fallback (no rg)
# ---------------------------------------------------------------------------

def test_search_fallback_finds_pattern(tmp_path):
    (tmp_path / "notes.txt").write_text("the quick brown fox\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    with patch("shutil.which", return_value=None):
        result = agent.run_tool("search", {"pattern": "quick"})

    assert "notes.txt" in result
    assert "quick" in result


def test_search_returns_no_matches_sentinel(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing here\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    with patch("shutil.which", return_value=None):
        result = agent.run_tool("search", {"pattern": "xyzzy_not_found"})

    assert "(no matches)" in result


# ---------------------------------------------------------------------------
# tool_run_shell()
# ---------------------------------------------------------------------------

def test_run_shell_captures_stdout_and_exit_code(tmp_path):
    agent = build_agent(tmp_path, [])
    result = agent.run_tool("run_shell", {"command": "echo hello_world"})
    assert "exit_code: 0" in result
    assert "hello_world" in result


def test_run_shell_captures_non_zero_exit_code(tmp_path):
    agent = build_agent(tmp_path, [])
    result = agent.run_tool("run_shell", {"command": "exit 42", "timeout": 5})
    assert "exit_code: 42" in result


def test_run_shell_rejects_timeout_out_of_range(tmp_path):
    agent = build_agent(tmp_path, [])
    result = agent.run_tool("run_shell", {"command": "echo hi", "timeout": 200})
    assert result.startswith("error:")


# ---------------------------------------------------------------------------
# tool_write_file()
# ---------------------------------------------------------------------------

def test_write_file_creates_parent_directories(tmp_path):
    agent = build_agent(tmp_path, [])
    result = agent.run_tool("write_file", {"path": "nested/deep/file.txt", "content": "hello\n"})
    assert "wrote" in result
    assert (tmp_path / "nested" / "deep" / "file.txt").read_text(encoding="utf-8") == "hello\n"


def test_write_file_overwrites_existing_file(tmp_path):
    (tmp_path / "existing.txt").write_text("old content\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])
    agent.run_tool("write_file", {"path": "existing.txt", "content": "new content\n"})
    assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "new content\n"


# ---------------------------------------------------------------------------
# tool_patch_file()
# ---------------------------------------------------------------------------

def test_patch_file_replaces_exact_match(tmp_path):
    (tmp_path / "sample.txt").write_text("hello world\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("patch_file", {"path": "sample.txt", "old_text": "world", "new_text": "agent"})

    assert result == "patched sample.txt"
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "hello agent\n"


def test_patch_file_rejects_duplicate_old_text(tmp_path):
    (tmp_path / "dup.txt").write_text("foo foo\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("patch_file", {"path": "dup.txt", "old_text": "foo", "new_text": "bar"})

    assert result.startswith("error:")
    assert "exactly once" in result


def test_patch_file_rejects_missing_file(tmp_path):
    agent = build_agent(tmp_path, [])
    result = agent.run_tool("patch_file", {"path": "ghost.txt", "old_text": "x", "new_text": "y"})
    assert result.startswith("error:")


# ---------------------------------------------------------------------------
# tool_delegate()
# ---------------------------------------------------------------------------

def test_delegate_not_registered_at_max_depth(tmp_path):
    # When depth == max_depth, delegate is not added to the tool registry at all.
    agent = build_agent(tmp_path, [], max_depth=0)
    result = agent.run_tool("delegate", {"task": "inspect README", "max_steps": 2})
    assert result == "error: unknown tool 'delegate'"


# ---------------------------------------------------------------------------
# resolve_path() security
# ---------------------------------------------------------------------------

def test_path_rejects_parent_escape(tmp_path):
    agent = build_agent(tmp_path, [])
    with pytest.raises(ValueError, match="path escapes workspace"):
        agent.resolve_path("../outside.txt")


def test_path_rejects_symlink_escape(tmp_path):
    agent = build_agent(tmp_path, [])
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "outside-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")

    with pytest.raises(ValueError, match="path escapes workspace"):
        agent.resolve_path("outside-link/secret.txt")


def test_path_accepts_case_variant_on_case_insensitive_filesystems(tmp_path):
    project_root = tmp_path / "Proj"
    project_root.mkdir()
    agent = build_agent(project_root, [])
    variant = project_root.parent / project_root.name.lower() / "README.md"

    if not variant.exists():
        pytest.skip("case-sensitive filesystem")

    resolved = agent.resolve_path(str(variant))
    assert resolved.samefile(project_root / "README.md")


# ---------------------------------------------------------------------------
# OllamaModelClient error handling
# ---------------------------------------------------------------------------

def test_ollama_client_raises_on_http_error():
    def fake_urlopen(request, timeout):
        exc = urllib.error.HTTPError(
            url="http://localhost/api/generate",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b"model not found"),
        )
        raise exc

    client = OllamaModelClient(model="bad", host="http://127.0.0.1:11434", temperature=0.2, top_p=0.9, timeout=5)
    with patch("urllib.request.urlopen", fake_urlopen):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            client.complete("test", 10)


def test_ollama_client_raises_on_url_error():
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    client = OllamaModelClient(model="qwen3.5:4b", host="http://127.0.0.1:11434", temperature=0.2, top_p=0.9, timeout=5)
    with patch("urllib.request.urlopen", fake_urlopen):
        with pytest.raises(RuntimeError, match="Could not reach Ollama"):
            client.complete("test", 10)


def test_ollama_client_raises_on_model_error_in_response():
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"error": "model not found"}).encode()

    client = OllamaModelClient(model="bad", host="http://127.0.0.1:11434", temperature=0.2, top_p=0.9, timeout=5)
    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        with pytest.raises(RuntimeError, match="Ollama error"):
            client.complete("test", 10)


def test_ollama_client_posts_expected_payload():
    captured = {}

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": "<final>ok</final>"}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode())
        return FakeResponse()

    client = OllamaModelClient(model="qwen3.5:4b", host="http://127.0.0.1:11434", temperature=0.2, top_p=0.9, timeout=30)
    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete("hello", 42)

    assert result == "<final>ok</final>"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["timeout"] == 30
    assert captured["body"]["model"] == "qwen3.5:4b"
    assert captured["body"]["options"]["num_predict"] == 42
    assert captured["body"]["stream"] is False


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

def test_session_store_save_and_load(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = _new_session(str(tmp_path))
    session["history"].append({"role": "user", "content": "hi", "created_at": "1"})

    store.save(session)
    loaded = store.load(session["id"])

    assert loaded["history"][0]["content"] == "hi"


def test_session_store_latest_returns_most_recent(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    s1 = _new_session(str(tmp_path))
    s2 = _new_session(str(tmp_path))
    store.save(s1)
    store.save(s2)

    latest = store.latest()
    assert latest == s2["id"]


def test_session_store_latest_returns_none_when_empty(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    assert store.latest() is None


# ---------------------------------------------------------------------------
# build_arg_parser()
# ---------------------------------------------------------------------------

def test_arg_parser_defaults():
    from clear_coding_agent import build_arg_parser
    args = build_arg_parser().parse_args([])
    assert args.model == "qwen3.5:4b"
    assert args.host == "http://127.0.0.1:11434"
    assert args.approval == "ask"
    assert args.max_steps == 6
    assert args.max_new_tokens == 512
    assert args.temperature == 0.2
    assert args.top_p == 0.9
    assert args.resume is None


# ---------------------------------------------------------------------------
# main() one-shot and REPL
# ---------------------------------------------------------------------------

def test_main_one_shot_prints_answer_and_returns_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".clear-coding-agent" / "sessions")
    fake_agent = ClearCodingAgent(
        model_client=FakeModelClient(["<final>The answer is 42.</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )

    with patch("clear_coding_agent.create_agent", return_value=fake_agent):
        result = main(["What is the answer?"])

    captured = capsys.readouterr()
    assert result == 0
    assert "The answer is 42." in captured.out


def test_main_repl_help_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".clear-coding-agent" / "sessions")
    fake_agent = ClearCodingAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )

    with patch("clear_coding_agent.create_agent", return_value=fake_agent):
        with patch("builtins.input", side_effect=["/help", EOFError]):
            result = main([])

    captured = capsys.readouterr()
    assert result == 0
    assert "Commands:" in captured.out


def test_main_repl_memory_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".clear-coding-agent" / "sessions")
    fake_agent = ClearCodingAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )
    fake_agent.session["memory"]["task"] = "my remembered task"

    with patch("clear_coding_agent.create_agent", return_value=fake_agent):
        with patch("builtins.input", side_effect=["/memory", EOFError]):
            result = main([])

    captured = capsys.readouterr()
    assert result == 0
    assert "my remembered task" in captured.out


def test_main_repl_reset_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".clear-coding-agent" / "sessions")
    fake_agent = ClearCodingAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )
    fake_agent.record({"role": "user", "content": "old message", "created_at": "1"})

    with patch("clear_coding_agent.create_agent", return_value=fake_agent):
        with patch("builtins.input", side_effect=["/reset", EOFError]):
            main([])

    assert fake_agent.session["history"] == []


def test_main_repl_exit_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".clear-coding-agent" / "sessions")
    fake_agent = ClearCodingAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )

    with patch("clear_coding_agent.create_agent", return_value=fake_agent):
        with patch("builtins.input", return_value="/exit"):
            result = main([])

    assert result == 0


# ---------------------------------------------------------------------------
# Existing regression tests (kept verbatim)
# ---------------------------------------------------------------------------

def test_agent_runs_tool_then_final(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":2}}</tool>',
            "<final>Read the file successfully.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Read the file successfully."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    assert "hello.txt" in agent.session["memory"]["files"]


def test_agent_retries_after_empty_model_output(tmp_path):
    agent = build_agent(tmp_path, ["", "<final>Recovered after retry.</final>"])

    answer = agent.ask("Do the task")

    assert answer == "Recovered after retry."
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("empty response" in item for item in notices)


def test_agent_retries_after_malformed_tool_payload(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":"bad"}</tool>',
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":1}}</tool>',
            "<final>Recovered after malformed tool output.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Recovered after malformed tool output."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("valid <tool> call" in item for item in notices)


def test_agent_accepts_xml_write_file_tool(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="hello.py"><content>print("hi")\n</content></tool>',
            "<final>Done.</final>",
        ],
    )

    answer = agent.ask("Create hello.py")

    assert answer == "Done."
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == 'print("hi")\n'


def test_retries_do_not_consume_the_whole_budget(tmp_path):
    agent = build_agent(
        tmp_path,
        ["", "", "<final>Recovered after several retries.</final>"],
        max_steps=1,
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after several retries."


def test_agent_saves_and_resumes_session(tmp_path):
    agent = build_agent(tmp_path, ["<final>First pass.</final>"])
    assert agent.ask("Start a session") == "First pass."

    resumed = ClearCodingAgent.from_session(
        model_client=FakeModelClient(["<final>Resumed.</final>"]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.session["history"][0]["content"] == "Start a session"
    assert resumed.ask("Continue") == "Resumed."


def test_delegate_uses_child_agent(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"delegate","args":{"task":"inspect README","max_steps":2}}</tool>',
            "<final>Child result.</final>",
            "<final>Parent incorporated the child result.</final>",
        ],
    )

    answer = agent.ask("Use delegation")

    assert answer == "Parent incorporated the child result."
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert tool_events[0]["name"] == "delegate"
    assert "delegate_result" in tool_events[0]["content"]


def test_invalid_risky_tool_does_not_prompt_for_approval(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="ask")

    with patch("builtins.input") as mock_input:
        result = agent.run_tool("write_file", {})

    assert result.startswith("error: invalid arguments for write_file: 'path'")
    assert 'example: <tool name="write_file"' in result
    mock_input.assert_not_called()


def test_list_files_hides_internal_agent_state(tmp_path):
    agent = build_agent(tmp_path, [])
    (tmp_path / ".clear-coding-agent").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")

    result = agent.run_tool("list_files", {})

    assert ".clear-coding-agent" not in result
    assert ".git" not in result
    assert "[F] hello.txt" in result


def test_repeated_identical_tool_call_is_rejected(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "1"})
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "2"})

    result = agent.run_tool("list_files", {})

    assert result == "error: repeated identical tool call for list_files; choose a different tool or return a final answer"


def test_welcome_screen_keeps_box_shape_for_long_paths(tmp_path):
    deep = tmp_path / "very" / "long" / "path" / "for" / "the" / "clear" / "agent" / "welcome" / "screen"
    deep.mkdir(parents=True)
    agent = build_agent(deep, [])

    welcome = build_welcome(agent, model="qwen3.5:4b", host="http://127.0.0.1:11434")
    lines = welcome.splitlines()

    assert len(lines) >= 5
    assert len({len(line) for line in lines}) == 1
    assert "..." in welcome
    assert "O   O" in welcome
    assert "CLEAR-CODING-AGENT" not in welcome
    assert "CLEAR CODING AGENT" in welcome


def test_prompt_top_level_sections_stay_flush_left_with_multiline_content(tmp_path):
    workspace = WorkspaceContext(
        cwd=str(tmp_path),
        repo_root=str(tmp_path),
        branch="fix/prompt-indentation",
        default_branch="main",
        status=" M clear_coding_agent.py\n?? tests/test_prompt.py",
        recent_commits=["abc123 first commit", "def456 second commit"],
        project_docs={"README.md": "line1\nline2"},
    )
    store = SessionStore(tmp_path / ".clear-coding-agent" / "sessions")
    agent = ClearCodingAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )
    agent.session["memory"] = {
        "task": "verify prompt formatting",
        "files": ["clear_coding_agent.py"],
        "notes": ["saw inconsistent indentation", "need regression coverage"],
    }
    agent.record({"role": "user", "content": "inspect prompt()", "created_at": "1"})
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "clear_coding_agent.py"},
            "content": "    def prompt(self, user_message):\n        ...",
            "created_at": "2",
        }
    )

    prompt = agent.prompt("is this issue legit?")
    lines = prompt.splitlines()

    for label in ["Rules:", "Tools:", "Valid response examples:", "Workspace:", "Memory:", "Transcript:", "Current user request:"]:
        assert label in lines
        assert f"            {label}" not in prompt


def _make_filler(i):
    return {"role": "tool", "name": "list_files", "args": {}, "content": "", "created_at": str(i)}


def test_history_text_deduplicates_reads_but_not_after_write(tmp_path):
    """read_file deduplication must not skip a read that follows a write."""
    agent = build_agent(tmp_path, [])

    agent.record({"role": "user", "content": "update config", "created_at": "0"})
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"config.txt"}}</tool>', "created_at": "1"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "config.txt"}, "content": "# config.txt\n   1: setting=true\n", "created_at": "2"})
    agent.record({"role": "assistant", "content": '<tool>{"name":"write_file","args":{"path":"config.txt","content":"setting=false\n"}}</tool>', "created_at": "3"})
    agent.record({"role": "tool", "name": "write_file", "args": {"path": "config.txt", "content": "setting=false\n"}, "content": "wrote config.txt", "created_at": "4"})
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"config.txt"}}</tool>', "created_at": "5"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "config.txt"}, "content": "# config.txt\n   1: setting=false\n", "created_at": "6"})
    for i in range(7, 13):
        agent.record(_make_filler(i))

    history = agent.history_text()

    assert "# config.txt\n   1: setting=true\n" in history
    assert "# config.txt\n   1: setting=false\n" in history
    assert history.count("setting=true") == 1


def test_history_text_deduplicates_unchanged_repeated_reads(tmp_path):
    """read_file deduplication should still skip repeated reads with no write in between."""
    agent = build_agent(tmp_path, [])

    agent.record({"role": "user", "content": "check logs", "created_at": "0"})
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"log.txt"}}</tool>', "created_at": "1"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "log.txt"}, "content": "# log.txt\n   1: stable\n", "created_at": "2"})
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"log.txt"}}</tool>', "created_at": "3"})
    for i in range(4, 10):
        agent.record(_make_filler(i))

    history = agent.history_text()

    assert history.count("stable") == 1


# ---------------------------------------------------------------------------
# End-to-end test with real Ollama (skipped if not available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running or qwen3.5:4b not available")
def test_end_to_end_real_ollama_lists_files(tmp_path):
    """Agent must call at least one tool and return a non-empty final answer."""
    (tmp_path / "main.py").write_text('def greet():\n    return "hello"\n', encoding="utf-8")
    (tmp_path / "utils.py").write_text("# utilities\n", encoding="utf-8")

    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".clear-coding-agent" / "sessions")
    agent = ClearCodingAgent(
        model_client=OllamaModelClient(
            model="qwen3.5:4b",
            host="http://127.0.0.1:11434",
            temperature=0.1,
            top_p=0.9,
            timeout=120,
        ),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        max_steps=4,
        max_new_tokens=300,
    )

    answer = agent.ask("List the Python files in this workspace.")

    assert answer, "agent returned an empty answer"
    assert not answer.startswith("Stopped after"), f"agent hit limit: {answer}"
    tool_calls = [e for e in agent.session["history"] if e["role"] == "tool"]
    assert len(tool_calls) >= 1, "agent never called any tool"


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running or qwen3.5:4b not available")
def test_end_to_end_real_ollama_creates_file(tmp_path):
    """Agent must write a file when asked and that file must exist on disk."""
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".clear-coding-agent" / "sessions")
    agent = ClearCodingAgent(
        model_client=OllamaModelClient(
            model="qwen3.5:4b",
            host="http://127.0.0.1:11434",
            temperature=0.1,
            top_p=0.9,
            timeout=120,
        ),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        max_steps=4,
        max_new_tokens=300,
    )

    agent.ask("Create a file called hello.txt containing exactly the text: Hello from clear-coding-agent")

    write_calls = [
        e for e in agent.session["history"]
        if e["role"] == "tool" and e["name"] == "write_file"
    ]
    assert len(write_calls) >= 1, "agent never called write_file"
    assert (tmp_path / "hello.txt").exists(), "hello.txt was not created on disk"

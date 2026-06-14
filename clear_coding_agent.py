import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
HELP_COMMANDS = "/help, /memory, /session, /reset, /exit"
WELCOME_ART = (
    "/\\     /\\\\",
    "{  `---'  }",
    "{  O   O  }",
    "~~>  V  <~~",
    "\\\\  \\|/  /",
    "`-----'__",
)
HELP_DETAILS = "\n".join(
    [
        "Commands:",
        "/help    Show this help message.",
        "/memory  Show the agent's distilled working memory.",
        "/session Show the path to the saved session file.",
        "/reset   Clear the current session history and memory.",
        "/exit    Exit the agent.",
    ]
)
MAX_TOOL_OUTPUT_CHARS = 4000
MAX_HISTORY_CHARS = 12000
IGNORED_DIR_NAMES = {
    ".git",
    ".clear-coding-agent",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
}

##############################
#### Six Agent Components ####
##############################
# 1) Live Repo Context -> WorkspaceContext
# 2) Prompt Shape And Cache Reuse -> _build_prefix, memory_text, prompt
# 3) Structured Tools, Validation, And Permissions -> _build_tools, run_tool, validate_tool, approve, parse, resolve_path, tool_*
# 4) Context Reduction And Output Management -> clip, history_text
# 5) Transcripts, Memory, And Resumption -> SessionStore, record, note_tool, ask, reset
# 6) Delegation And Bounded Subagents -> tool_delegate


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Supporting helper for component 4 (context reduction and output management).
def clip(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def middle(text: str, max_chars: int) -> str:
    text = str(text).replace("\n", " ")
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    left_len = (max_chars - 3) // 2
    right_len = max_chars - 3 - left_len
    return text[:left_len] + "..." + text[-right_len:]


##############################
#### 1) Live Repo Context ####
##############################
class WorkspaceContext:
    def __init__(self, cwd, repo_root, branch, default_branch, status, recent_commits, project_docs):
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs

    @classmethod
    def build(cls, cwd):
        cwd = Path(cwd).resolve()

        def run_git(git_args, fallback=""):
            try:
                result = subprocess.run(
                    ["git", *git_args],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        repo_root = Path(run_git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        doc_snippets = {}
        for search_dir in (repo_root, cwd):
            for doc_name in PROJECT_DOC_NAMES:
                doc_path = search_dir / doc_name
                if not doc_path.exists():
                    continue
                relative_doc_path = str(doc_path.relative_to(repo_root))
                if relative_doc_path in doc_snippets:
                    continue
                doc_snippets[relative_doc_path] = clip(doc_path.read_text(encoding="utf-8", errors="replace"), 1200)

        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=run_git(["branch", "--show-current"], "-") or "-",
            default_branch=(
                run_git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main"
            ).removeprefix("origin/"),
            status=clip(run_git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[line for line in run_git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=doc_snippets,
        )

    def text(self) -> str:
        commit_lines = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        doc_lines = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"
        return "\n".join(
            [
                "Workspace:",
                f"- cwd: {self.cwd}",
                f"- repo_root: {self.repo_root}",
                f"- branch: {self.branch}",
                f"- default_branch: {self.default_branch}",
                "- status:",
                self.status,
                "- recent_commits:",
                commit_lines,
                "- project_docs:",
                doc_lines,
            ]
        )


##############################
#### 5) Session Memory #######
##############################
class SessionStore:
    def __init__(self, sessions_dir):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def session_file_path(self, session_id):
        return self.sessions_dir / f"{session_id}.json"

    def save(self, session):
        file_path = self.session_file_path(session["id"])
        file_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        return file_path

    def load(self, session_id):
        return json.loads(self.session_file_path(session_id).read_text(encoding="utf-8"))

    def latest(self):
        session_files = sorted(self.sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        return session_files[-1].stem if session_files else None


def _new_session(workspace_root: str) -> dict:
    return {
        "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
        "created_at": now(),
        "workspace_root": workspace_root,
        "history": [],
        "memory": {"task": "", "files": [], "notes": []},
    }


class FakeModelClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def complete(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)


class OllamaModelClient:
    def __init__(self, model, host, temperature, top_p, timeout):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    def complete(self, prompt, max_new_tokens):
        request_body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        http_request = urllib.request.Request(
            self.host + "/api/generate",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ) from exc

        if response_data.get("error"):
            raise RuntimeError(f"Ollama error: {response_data['error']}")
        return response_data.get("response", "")


@dataclass(frozen=True)
class Tool:
    schema: dict
    description: str
    risky: bool
    run: Callable


class ClearCodingAgent:
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        approval_policy="ask",
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.session = session or _new_session(workspace.repo_root)
        self.tools = self._build_tools()
        self.prefix = self._build_prefix()
        self.session_path = self.session_store.save(self.session)

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    @staticmethod
    def remember(items, entry, max_size):
        if not entry:
            return
        if entry in items:
            items.remove(entry)
        items.append(entry)
        del items[:-max_size]

    ###############################################
    #### 3) Structured Tools And Permissions ######
    ###############################################
    def _build_tools(self) -> dict:
        tools = {
            "list_files": Tool(
                schema={"path": "str='.'"},
                risky=False,
                description="List files in the workspace.",
                run=self.tool_list_files,
            ),
            "read_file": Tool(
                schema={"path": "str", "start": "int=1", "end": "int=200"},
                risky=False,
                description="Read a UTF-8 file by line range.",
                run=self.tool_read_file,
            ),
            "search": Tool(
                schema={"pattern": "str", "path": "str='.'"},
                risky=False,
                description="Search the workspace with rg or a simple fallback.",
                run=self.tool_search,
            ),
            "run_shell": Tool(
                schema={"command": "str", "timeout": "int=20"},
                risky=True,
                description="Run a shell command in the repo root.",
                run=self.tool_run_shell,
            ),
            "write_file": Tool(
                schema={"path": "str", "content": "str"},
                risky=True,
                description="Write a text file.",
                run=self.tool_write_file,
            ),
            "patch_file": Tool(
                schema={"path": "str", "old_text": "str", "new_text": "str"},
                risky=True,
                description="Replace one exact text block in a file.",
                run=self.tool_patch_file,
            ),
        }
        if self.depth < self.max_depth:
            tools["delegate"] = Tool(
                schema={"task": "str", "max_steps": "int=3"},
                risky=False,
                description="Ask a bounded read-only child agent to investigate.",
                run=self.tool_delegate,
            )
        return tools

    ############################################
    #### 2) Prompt Shape And Cache Reuse #######
    ############################################
    def _build_prefix(self) -> str:
        tool_lines = []
        for name, tool in self.tools.items():
            schema_fields = ", ".join(f"{k}: {v}" for k, v in tool.schema.items())
            risk_label = "approval required" if tool.risky else "safe"
            tool_lines.append(f"- {name}({schema_fields}) [{risk_label}] {tool.description}")
        tools_text = "\n".join(tool_lines)
        examples = "\n".join(
            [
                '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
                '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
                '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
                '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
                "<final>Done.</final>",
            ]
        )
        rules = "\n".join(
            [
                "- Use tools instead of guessing about the workspace.",
                "- Return exactly one <tool>...</tool> or one <final>...</final>.",
                "- Tool calls must look like:",
                '  <tool>{"name":"tool_name","args":{...}}</tool>',
                "- For write_file and patch_file with multi-line text, prefer XML style:",
                '  <tool name="write_file" path="file.py"><content>...</content></tool>',
                "- Final answers must look like:",
                "  <final>your answer</final>",
                "- Never invent tool results.",
                "- Keep answers concise and concrete.",
                "- If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.",
                "- Before writing tests for existing code, read the implementation first.",
                "- When writing tests, match the current implementation unless the user explicitly asked you to change the code.",
                "- New files should be complete and runnable, including obvious imports.",
                "- Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.",
                "- Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={}.",
            ]
        )
        return "\n\n".join(
            [
                "You are Clear-Coding-Agent, a small local coding agent running through Ollama.",
                "Rules:\n" + rules,
                "Tools:\n" + tools_text,
                "Valid response examples:\n" + examples,
                self.workspace.text(),
            ]
        )

    def memory_text(self) -> str:
        memory = self.session["memory"]
        notes = "\n".join(f"- {note}" for note in memory["notes"]) or "- none"
        return "\n".join(
            [
                "Memory:",
                f"- task: {memory['task'] or '-'}",
                f"- files: {', '.join(memory['files']) or '-'}",
                "- notes:",
                notes,
            ]
        )

    #####################################################
    #### 4) Context Reduction And Output Management #####
    #####################################################
    def history_text(self) -> str:
        history = self.session["history"]
        if not history:
            return "- empty"

        output_lines = []
        seen_reads: set[str] = set()
        recent_start = max(0, len(history) - 6)

        for idx, entry in enumerate(history):
            is_recent = idx >= recent_start

            if entry["role"] == "tool" and entry["name"] in ("write_file", "patch_file"):
                file_path = str(entry["args"].get("path", ""))
                seen_reads.discard(file_path)

            if entry["role"] == "tool" and entry["name"] == "read_file" and not is_recent:
                file_path = str(entry["args"].get("path", ""))
                if file_path in seen_reads:
                    continue
                seen_reads.add(file_path)

            if entry["role"] == "tool":
                char_limit = 900 if is_recent else 180
                output_lines.append(f"[tool:{entry['name']}] {json.dumps(entry['args'], sort_keys=True)}")
                output_lines.append(clip(entry["content"], char_limit))
            else:
                char_limit = 900 if is_recent else 220
                output_lines.append(f"[{entry['role']}] {clip(entry['content'], char_limit)}")

        return clip("\n".join(output_lines), MAX_HISTORY_CHARS)

    ########################################################
    #### 2) Prompt Shape And Cache Reuse (Continued) #######
    ########################################################
    def prompt(self, user_message: str) -> str:
        return "\n\n".join(
            [
                self.prefix,
                self.memory_text(),
                "Transcript:\n" + self.history_text(),
                "Current user request:\n" + user_message,
            ]
        )

    ###############################################
    #### 5) Session Memory (Continued) ###########
    ###############################################
    def record(self, entry: dict) -> None:
        self.session["history"].append(entry)
        self.session_path = self.session_store.save(self.session)

    def note_tool(self, name: str, tool_args: dict, result: str) -> None:
        memory = self.session["memory"]
        file_path = tool_args.get("path")
        if name in {"read_file", "write_file", "patch_file"} and file_path:
            self.remember(memory["files"], str(file_path), 8)
        note = f"{name}: {clip(str(result).replace(chr(10), ' '), 220)}"
        self.remember(memory["notes"], note, 5)

    def ask(self, user_message: str) -> str:
        memory = self.session["memory"]
        if not memory["task"]:
            memory["task"] = clip(user_message.strip(), 300)
        self.record({"role": "user", "content": user_message, "created_at": now()})

        tool_call_count = 0
        attempt_count = 0
        max_total_attempts = max(self.max_steps * 3, self.max_steps + 4)

        while tool_call_count < self.max_steps and attempt_count < max_total_attempts:
            attempt_count += 1
            model_response = self.model_client.complete(self.prompt(user_message), self.max_new_tokens)
            response_type, parsed = self.parse(model_response)

            if response_type == "tool":
                tool_call_count += 1
                tool_name = parsed.get("name", "")
                tool_args = parsed.get("args", {})
                tool_result = self.run_tool(tool_name, tool_args)
                self.record(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "args": tool_args,
                        "content": tool_result,
                        "created_at": now(),
                    }
                )
                self.note_tool(tool_name, tool_args, tool_result)
                continue

            if response_type == "retry":
                self.record({"role": "assistant", "content": parsed, "created_at": now()})
                continue

            final_answer = (parsed or model_response).strip()
            self.record({"role": "assistant", "content": final_answer, "created_at": now()})
            self.remember(memory["notes"], clip(final_answer, 220), 5)
            return final_answer

        if attempt_count >= max_total_attempts and tool_call_count < self.max_steps:
            final_answer = "Stopped after too many malformed model responses without a valid tool call or final answer."
        else:
            final_answer = "Stopped after reaching the step limit without a final answer."
        self.record({"role": "assistant", "content": final_answer, "created_at": now()})
        return final_answer

    #############################################################
    #### 3) Structured Tools, Validation, And Permissions #######
    #############################################################
    def run_tool(self, name: str, tool_args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'"
        try:
            self.validate_tool(name, tool_args)
        except Exception as exc:
            usage_example = self.tool_example(name)
            error_message = f"error: invalid arguments for {name}: {exc}"
            if usage_example:
                error_message += f"\nexample: {usage_example}"
            return error_message
        if self.repeated_tool_call(name, tool_args):
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"
        if tool.risky and not self.approve(name, tool_args):
            return f"error: approval denied for {name}"
        try:
            return clip(tool.run(tool_args))
        except Exception as exc:
            return f"error: tool {name} failed: {exc}"

    def repeated_tool_call(self, name: str, tool_args: dict) -> bool:
        tool_events = [entry for entry in self.session["history"] if entry["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        last_two = tool_events[-2:]
        return all(entry["name"] == name and entry["args"] == tool_args for entry in last_two)

    def tool_example(self, name: str) -> str:
        examples = {
            "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
            "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
            "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
            "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
            "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
            "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
        }
        return examples.get(name, "")

    def validate_tool(self, name: str, tool_args: dict) -> None:
        tool_args = tool_args or {}

        if name == "list_files":
            target_path = self.resolve_path(tool_args.get("path", "."))
            if not target_path.is_dir():
                raise ValueError("path is not a directory")
            return

        if name == "read_file":
            target_path = self.resolve_path(tool_args["path"])
            if not target_path.is_file():
                raise ValueError("path is not a file")
            start = int(tool_args.get("start", 1))
            end = int(tool_args.get("end", 200))
            if start < 1 or end < start:
                raise ValueError("invalid line range")
            return

        if name == "search":
            pattern = str(tool_args.get("pattern", "")).strip()
            if not pattern:
                raise ValueError("pattern must not be empty")
            self.resolve_path(tool_args.get("path", "."))
            return

        if name == "run_shell":
            command = str(tool_args.get("command", "")).strip()
            if not command:
                raise ValueError("command must not be empty")
            timeout = int(tool_args.get("timeout", 20))
            if timeout < 1 or timeout > 120:
                raise ValueError("timeout must be in [1, 120]")
            return

        if name == "write_file":
            target_path = self.resolve_path(tool_args["path"])
            if target_path.exists() and target_path.is_dir():
                raise ValueError("path is a directory")
            if "content" not in tool_args:
                raise ValueError("missing content")
            return

        if name == "patch_file":
            target_path = self.resolve_path(tool_args["path"])
            if not target_path.is_file():
                raise ValueError("path is not a file")
            old_text = str(tool_args.get("old_text", ""))
            if not old_text:
                raise ValueError("old_text must not be empty")
            if "new_text" not in tool_args:
                raise ValueError("missing new_text")
            file_content = target_path.read_text(encoding="utf-8")
            occurrence_count = file_content.count(old_text)
            if occurrence_count != 1:
                raise ValueError(f"old_text must occur exactly once, found {occurrence_count}")
            return

        if name == "delegate":
            if self.depth >= self.max_depth:
                raise ValueError("delegate depth exceeded")
            task = str(tool_args.get("task", "")).strip()
            if not task:
                raise ValueError("task must not be empty")
            return

    def approve(self, name: str, tool_args: dict) -> bool:
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(tool_args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(response: str) -> tuple:
        response = str(response)
        if "<tool>" in response and ("<final>" not in response or response.find("<tool>") < response.find("<final>")):
            json_body = ClearCodingAgent.extract(response, "tool")
            try:
                parsed = json.loads(json_body)
            except Exception:
                return "retry", ClearCodingAgent.retry_notice("model returned malformed tool JSON")
            if not isinstance(parsed, dict):
                return "retry", ClearCodingAgent.retry_notice("tool payload must be a JSON object")
            if not str(parsed.get("name", "")).strip():
                return "retry", ClearCodingAgent.retry_notice("tool payload is missing a tool name")
            tool_args = parsed.get("args", {})
            if tool_args is None:
                parsed["args"] = {}
            elif not isinstance(tool_args, dict):
                return "retry", ClearCodingAgent.retry_notice()
            return "tool", parsed
        if "<tool" in response and ("<final>" not in response or response.find("<tool") < response.find("<final>")):
            parsed = ClearCodingAgent.parse_xml_tool(response)
            if parsed is not None:
                return "tool", parsed
            return "retry", ClearCodingAgent.retry_notice()
        if "<final>" in response:
            final_text = ClearCodingAgent.extract(response, "final").strip()
            if final_text:
                return "final", final_text
            return "retry", ClearCodingAgent.retry_notice("model returned an empty <final> answer")
        response = response.strip()
        if response:
            return "final", response
        return "retry", ClearCodingAgent.retry_notice("model returned an empty response")

    @staticmethod
    def retry_notice(problem: str | None = None) -> str:
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(response: str) -> dict | None:
        xml_match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", response, re.S)
        if not xml_match:
            return None
        tag_attrs = ClearCodingAgent.parse_attrs(xml_match.group("attrs"))
        tool_name = str(tag_attrs.pop("name", "")).strip()
        if not tool_name:
            return None

        tag_body = xml_match.group("body")
        extra_args = dict(tag_attrs)
        for field_key in ("content", "old_text", "new_text", "command", "task", "pattern", "path"):
            if f"<{field_key}>" in tag_body:
                extra_args[field_key] = ClearCodingAgent.extract_raw(tag_body, field_key)

        stripped_body = tag_body.strip("\n")
        if tool_name == "write_file" and "content" not in extra_args and stripped_body:
            extra_args["content"] = stripped_body
        if tool_name == "delegate" and "task" not in extra_args and stripped_body:
            extra_args["task"] = stripped_body.strip()
        return {"name": tool_name, "args": extra_args}

    @staticmethod
    def parse_attrs(attr_string: str) -> dict:
        attrs = {}
        for attr_match in re.finditer(
            r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", attr_string
        ):
            attrs[attr_match.group(1)] = (
                attr_match.group(2) if attr_match.group(2) is not None else attr_match.group(3)
            )
        return attrs

    @staticmethod
    def extract(text: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self) -> None:
        self.session["history"] = []
        self.session["memory"] = {"task": "", "files": [], "notes": []}
        self.session_store.save(self.session)

    def _path_is_within_root(self, resolved: Path) -> bool:
        check = resolved
        while not check.exists() and check.parent != check:
            check = check.parent
        for ancestor in (check, *check.parents):
            try:
                if ancestor.samefile(self.root):
                    return True
            except OSError:
                continue
        return False

    def resolve_path(self, raw_path: str) -> Path:
        resolved_path = Path(raw_path)
        resolved_path = resolved_path if resolved_path.is_absolute() else self.root / resolved_path
        resolved = resolved_path.resolve()
        if not self._path_is_within_root(resolved):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved

    def tool_list_files(self, tool_args: dict) -> str:
        dir_path = self.resolve_path(tool_args.get("path", "."))
        if not dir_path.is_dir():
            raise ValueError("path is not a directory")
        dir_entries = [
            entry
            for entry in sorted(dir_path.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
            if entry.name not in IGNORED_DIR_NAMES
        ]
        output_lines = []
        for entry in dir_entries[:200]:
            entry_kind = "[F]" if entry.is_file() else "[D]"
            output_lines.append(f"{entry_kind} {entry.relative_to(self.root)}")
        return "\n".join(output_lines) or "(empty)"

    def tool_read_file(self, tool_args: dict) -> str:
        file_path = self.resolve_path(tool_args["path"])
        if not file_path.is_file():
            raise ValueError("path is not a file")
        start = int(tool_args.get("start", 1))
        end = int(tool_args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(
            f"{line_num:>4}: {line_text}"
            for line_num, line_text in enumerate(lines[start - 1 : end], start=start)
        )
        return f"# {file_path.relative_to(self.root)}\n{body}"

    def tool_search(self, tool_args: dict) -> str:
        pattern = str(tool_args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        search_path = self.resolve_path(tool_args.get("path", "."))

        if shutil.which("rg"):
            proc = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(search_path)],
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            return proc.stdout.strip() or proc.stderr.strip() or "(no matches)"

        matches = []
        target_files = (
            [search_path]
            if search_path.is_file()
            else [
                f
                for f in search_path.rglob("*")
                if f.is_file()
                and not any(part in IGNORED_DIR_NAMES for part in f.relative_to(self.root).parts)
            ]
        )
        for file_path in target_files:
            for line_num, line_text in enumerate(
                file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                if pattern.lower() in line_text.lower():
                    matches.append(f"{file_path.relative_to(self.root)}:{line_num}:{line_text}")
                    if len(matches) >= 200:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def tool_run_shell(self, tool_args: dict) -> str:
        command = str(tool_args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(tool_args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        proc = subprocess.run(
            command,
            cwd=self.root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return "\n".join(
            [
                f"exit_code: {proc.returncode}",
                "stdout:",
                proc.stdout.strip() or "(empty)",
                "stderr:",
                proc.stderr.strip() or "(empty)",
            ]
        )

    def tool_write_file(self, tool_args: dict) -> str:
        file_path = self.resolve_path(tool_args["path"])
        content = str(tool_args["content"])
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"wrote {file_path.relative_to(self.root)} ({len(content)} chars)"

    def tool_patch_file(self, tool_args: dict) -> str:
        file_path = self.resolve_path(tool_args["path"])
        if not file_path.is_file():
            raise ValueError("path is not a file")
        old_text = str(tool_args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in tool_args:
            raise ValueError("missing new_text")
        file_content = file_path.read_text(encoding="utf-8")
        occurrence_count = file_content.count(old_text)
        if occurrence_count != 1:
            raise ValueError(f"old_text must occur exactly once, found {occurrence_count}")
        file_path.write_text(file_content.replace(old_text, str(tool_args["new_text"]), 1), encoding="utf-8")
        return f"patched {file_path.relative_to(self.root)}"

    ###################################################
    #### 6) Delegation And Bounded Subagents ##########
    ###################################################
    def tool_delegate(self, tool_args: dict) -> str:
        if self.depth >= self.max_depth:
            raise ValueError("delegate depth exceeded")
        task = str(tool_args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        child = ClearCodingAgent(
            model_client=self.model_client,
            workspace=self.workspace,
            session_store=self.session_store,
            approval_policy="never",
            max_steps=int(tool_args.get("max_steps", 3)),
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=True,
        )
        child.session["memory"]["task"] = task
        child.session["memory"]["notes"] = [clip(self.history_text(), 300)]
        return "delegate_result:\n" + child.ask(task)


# Backwards-compatible alias so external code importing MiniAgent still works.
MiniAgent = ClearCodingAgent


def build_welcome(agent, model: str, host: str) -> str:
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner_width = width - 4
    col_gap = 3
    left_col_width = (inner_width - col_gap) // 2
    right_col_width = inner_width - col_gap - left_col_width

    def row(text):
        clipped = middle(text, width - 4)
        return f"| {clipped.ljust(width - 4)} |"

    def divider(char="-"):
        return "+" + char * (width - 2) + "+"

    def center(text):
        clipped = middle(text, inner_width)
        return f"| {clipped.center(inner_width)} |"

    def cell(label, value, col_width):
        clipped = middle(f"{label:<9} {value}", col_width)
        return clipped.ljust(col_width)

    def pair(left_label, left_value, right_label, right_value):
        left_cell = cell(left_label, left_value, left_col_width)
        right_cell = cell(right_label, right_value, right_col_width)
        return f"| {left_cell}{' ' * col_gap}{right_cell} |"

    border = divider("=")
    banner_lines = [center(text) for text in WELCOME_ART]
    banner_lines.extend(
        [
            center("CLEAR CODING AGENT"),
            divider("-"),
            row(""),
            row("WORKSPACE  " + middle(agent.workspace.cwd, inner_width - 11)),
            pair("MODEL", model, "BRANCH", agent.workspace.branch),
            pair("APPROVAL", agent.approval_policy, "SESSION", agent.session["id"]),
            row(""),
        ]
    )
    return "\n".join([border, *banner_lines, border])


def create_agent(cli_args):
    workspace = WorkspaceContext.build(cli_args.cwd)
    session_store = SessionStore(Path(workspace.repo_root) / ".clear-coding-agent" / "sessions")
    model_client = OllamaModelClient(
        model=cli_args.model,
        host=cli_args.host,
        temperature=cli_args.temperature,
        top_p=cli_args.top_p,
        timeout=cli_args.ollama_timeout,
    )
    session_id = cli_args.resume
    if session_id == "latest":
        session_id = session_store.latest()
    if session_id:
        return ClearCodingAgent.from_session(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session_id=session_id,
            approval_policy=cli_args.approval,
            max_steps=cli_args.max_steps,
            max_new_tokens=cli_args.max_new_tokens,
        )
    return ClearCodingAgent(
        model_client=model_client,
        workspace=workspace,
        session_store=session_store,
        approval_policy=cli_args.approval,
        max_steps=cli_args.max_steps,
        max_new_tokens=cli_args.max_new_tokens,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Clear coding agent for Ollama models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--model", default="qwen3.5:4b", help="Ollama model name.")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama server URL.")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument(
        "--approval",
        choices=("ask", "auto", "never"),
        default="ask",
        help="Approval policy for risky tools; auto grants the model arbitrary command execution and file writes.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to Ollama.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value sent to Ollama.")
    return parser


def main(argv=None):
    cli_args = build_arg_parser().parse_args(argv)
    agent = create_agent(cli_args)

    print(build_welcome(agent, model=cli_args.model, host=cli_args.host))

    if cli_args.prompt:
        prompt_text = " ".join(cli_args.prompt).strip()
        if prompt_text:
            print()
            try:
                print(agent.ask(prompt_text))
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        return 0

    while True:
        try:
            user_input = input("\nclear-coding-agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            print(agent.memory_text())
            continue
        if user_input == "/session":
            print(agent.session_path)
            continue
        if user_input == "/reset":
            agent.reset()
            print("session reset")
            continue

        print()
        try:
            print(agent.ask(user_input))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

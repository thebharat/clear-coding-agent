&nbsp;
# Clear-Coding-Agent

A small standalone coding agent backed by Ollama.

- code: `clear_coding_agent.py`
- CLI: `clear-coding-agent`

It is a minimal local agent loop with:

- workspace snapshot collection
- stable prompt plus turn state
- structured tools
- approval handling for risky tools
- transcript and memory persistence
- bounded delegation

The model backend is Ollama.

&nbsp;
## Six Core Components

This coding harness is organised around six practical building blocks:

1. **Live repo context**
   The agent collects stable workspace facts upfront: repo layout, project docs, and git state.
2. **Prompt shape and cache reuse**
   A stable prompt prefix is kept separate from the changing request, transcript, and memory so repeated model calls can reuse the static parts efficiently.
3. **Structured tools, validation, and permissions**
   The model works through named tools with checked inputs, workspace path validation, and approval gates instead of free-form arbitrary actions.
4. **Context reduction and output management**
   Long outputs are clipped, repeated reads are deduplicated, and older transcript entries are compressed to keep prompt size under control.
5. **Transcripts, memory, and resumption**
   The runtime keeps both a full durable transcript and a smaller working memory so sessions can be resumed while preserving important state.
6. **Delegation and bounded subagents**
   Scoped subtasks can be delegated to read-only child agents that inherit enough context to help but operate within limits.

&nbsp;
## What changed from mini-coding-agent

This repo is a structural refactor of [rasbt/mini-coding-agent](https://github.com/rasbt/mini-coding-agent). The logic is identical — only naming and structure were improved:

- `MiniAgent` renamed to `ClearCodingAgent` (a `MiniAgent` alias is kept for compatibility)
- `Tool` frozen dataclass replaces the nested dict-of-dicts tool registry
- `_new_session()` helper centralises session dict initialisation
- Private methods prefixed with `_`: `_build_tools`, `_build_prefix`, `_path_is_within_root`
- `path()` renamed to `resolve_path()` to avoid shadowing local path variables
- `SessionStore.path()` renamed to `session_file_path()`
- `build_agent()` renamed to `create_agent()` with a clearer `cli_args` parameter
- Variable names clarified throughout: `raw` → `model_response`, `kind` → `response_type`, `args` → `tool_args`, `limit` → `max_chars`, `inner` → `inner_width`, `probe` → `check`, `entries` → `dir_entries`, etc.
- Constants renamed: `DOC_NAMES` → `PROJECT_DOC_NAMES`, `MAX_TOOL_OUTPUT` → `MAX_TOOL_OUTPUT_CHARS`, `IGNORED_PATH_NAMES` → `IGNORED_DIR_NAMES`

&nbsp;
## Requirements

- Python 3.10+
- Ollama installed and running
- An Ollama model pulled locally

Optional:

- `uv` for environment management and the `clear-coding-agent` CLI entry point

No Python runtime dependency beyond the standard library — run it directly with `python clear_coding_agent.py` if you do not want `uv`.

&nbsp;
## Install Ollama

Install Ollama from [ollama.com/download](https://ollama.com/download), then:

```bash
ollama serve
```

In another terminal, pull a model:

```bash
ollama pull qwen3.5:4b
```

The default model is `qwen3.5:4b`. A larger variant such as `qwen3.5:9b` will follow the tool-call format more reliably.

&nbsp;
## Project Setup

```bash
git clone https://github.com/thebharat/clear-coding-agent.git
cd clear-coding-agent
```

&nbsp;
## Basic Usage

Start the interactive agent:

```bash
uv run clear-coding-agent
```

Without `uv`:

```bash
python clear_coding_agent.py
```

Defaults:

- model: `qwen3.5:4b`
- approval: `ask`

&nbsp;
## Approval Modes

Risky tools (shell commands and file writes) are gated by an approval policy:

- `--approval ask` — prompts before risky actions (default)
- `--approval auto` — allows risky actions automatically; use only with trusted prompts and repos
- `--approval never` — denies all risky actions

```bash
uv run clear-coding-agent --approval auto
```

&nbsp;
## Resume Sessions

Sessions are saved under the target workspace root in:

```text
.clear-coding-agent/sessions/
```

Resume the latest session:

```bash
uv run clear-coding-agent --resume latest
```

Resume a specific session:

```bash
uv run clear-coding-agent --resume 20260401-144025-2dd0aa
```

&nbsp;
## Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/memory` | Print the distilled session memory: current task, tracked files, and notes |
| `/session` | Print the path to the current saved session JSON file |
| `/reset` | Clear history and memory but stay in the REPL |
| `/exit` | Exit the agent |
| `/quit` | Alias for `/exit` |

&nbsp;
## CLI Flags

```bash
uv run clear-coding-agent --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--cwd` | `.` | Workspace directory to inspect and modify |
| `--model` | `qwen3.5:4b` | Ollama model name |
| `--host` | `http://127.0.0.1:11434` | Ollama server URL |
| `--ollama-timeout` | `300` | Request timeout in seconds |
| `--resume` | — | Session id to resume or `latest` |
| `--approval` | `ask` | `ask`, `auto`, or `never` |
| `--max-steps` | `6` | Max tool/model iterations per request |
| `--max-new-tokens` | `512` | Max model output tokens per step |
| `--temperature` | `0.2` | Sampling temperature |
| `--top-p` | `0.9` | Nucleus sampling value |

&nbsp;
## Notes

- The agent expects the model to emit either `<tool>...</tool>` or `<final>...</final>`.
- Different Ollama models follow the format with different reliability.
- If the model does not follow the format well, switch to a stronger instruction-following model.
- The agent is intentionally small and optimised for readability, not robustness.

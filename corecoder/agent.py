"""Core agent loop.

This is the heart of CoreCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

import concurrent.futures
import inspect
import json
from typing import TYPE_CHECKING

from tools.log.logging import AgentLogger
from .context import ContextManager, estimate_tokens
from .llm import LLM
from .prompt import system_prompt
from .tools import ALL_TOOLS
from .tools.agent import AgentTool
from .tools.base import Tool

if TYPE_CHECKING:
    from tools.log.logstore import LogStore


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
        logstore: "LogStore | None" = None,
        session_id: str = "",
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)
        self.log = AgentLogger(logstore, session_id)
        self._turn_id = 0

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def _append_message(self, msg: dict, round_num: int = 0):
        """Append a message to history and persist it."""
        role = msg.get("role", "")
        if role == "assistant" and msg.get("tool_calls"):
            content = json.dumps(msg, ensure_ascii=False)
        else:
            content = msg.get("content", "") or ""
        self.log.message(self._turn_id, round_num,
                          len(self.messages), role, content,
                          msg.get("tool_call_id", ""))
        self.messages.append(msg)

    def _maybe_compress(self, round_num: int):
        before_tokens = estimate_tokens(self.messages)
        before_msgs = len(self.messages)
        if self.context.maybe_compress(self.messages, self.llm):
            self.log.compress(self._turn_id, round_num,
                              before_tokens, estimate_tokens(self.messages),
                              before_msgs, len(self.messages))

    def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self._turn_id += 1
        tid = self._turn_id

        self._append_message({"role": "user", "content": user_input})
        self._maybe_compress(round_num=0)

        for round_num in range(self.max_rounds):
            self.log.chat_request(tid, round_num,
                                  self._full_messages(),
                                  self._tool_schemas())
            resp = self.llm.chat(
                messages=self._full_messages(),
                tools=self._tool_schemas(),
                on_token=on_token,
            )

            # no tool calls -> LLM is done, return text
            if not resp.tool_calls:
                self._append_message(resp.message, round_num)
                self.log.chat_done(tid, round_num, len(resp.content))
                return resp.content

            # tool calls -> execute (parallel when multiple, like Claude Code's
            # StreamingToolExecutor which runs independent tools concurrently)
            self._append_message(resp.message, round_num)

            try:
                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    result = self._exec_tool(tc, tid, round_num)
                    self._append_message({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }, round_num)
                else:
                    # parallel execution for multiple tool calls
                    results = self._exec_tools_parallel(
                        resp.tool_calls, on_tool, tid, round_num)
                    for tc, result in zip(resp.tool_calls, results):
                        self._append_message({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }, round_num)
            except KeyboardInterrupt:
                self._answer_pending_tool_calls(resp.tool_calls)
                raise

            # compress if tool outputs are big
            self._maybe_compress(round_num)

        return "(reached maximum tool-call rounds)"

    def _exec_tool(self, tc, turn_id: int = 0, round_num: int = 0) -> str:
        """Execute a single tool call, returning the result string."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            self.log.tool_error(turn_id, round_num, tc.name,
                                 f"Unknown tool: {tc.name}")
            return f"Error: unknown tool '{tc.name}'"

        self.log.tool_start(turn_id, round_num, tc.name, tc.arguments)

        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            self.log.tool_error(turn_id, round_num, tc.name, str(e))
            return f"Error: bad arguments for {tc.name}: {e}"
        try:
            result = tool.execute(**tc.arguments)
            return result
        except Exception as e:
            self.log.tool_error(turn_id, round_num, tc.name, str(e))
            return f"Error executing {tc.name}: {e}"

    def _exec_tools_parallel(self, tool_calls, on_tool=None,
                             turn_id: int = 0, round_num: int = 0) -> list[str]:
        """Run multiple tool calls concurrently using threads."""
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self._exec_tool, tc, turn_id, round_num)
                       for tc in tool_calls]
            return [f.result() for f in futures]

    def _answer_pending_tool_calls(self, tool_calls):
        """Backfill a tool reply for every call that didn't get one.

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self._append_message({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()

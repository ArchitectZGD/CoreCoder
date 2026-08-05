"""Thin wrapper around LogStore that makes logging calls safe to call
even when no logstore is configured (no-op fallback).

This eliminates repetitive ``if self.logstore:`` guards from agent.py.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.log.logstore import LogStore


class AgentLogger:
    """Proxy that delegates to LogStore when available, no-op otherwise."""

    def __init__(self, logstore: "LogStore | None" = None, session_id: str = ""):
        self._ls = logstore
        self._sid = session_id

    @property
    def session_id(self) -> str:
        return self._sid

    @session_id.setter
    def session_id(self, value: str):
        self._sid = value

    # -- public API (mirrors LogStore) ------------------------------------

    def chat_request(self, turn: int, round_num: int,
                     messages: list[dict], tools: list[dict]):
        if self._ls:
            self._ls.chat_request(self._sid, turn, round_num,
                                  messages, tools)

    def chat_done(self, turn: int, round_num: int, content_len: int):
        if self._ls:
            self._ls.chat_done(self._sid, turn, round_num, content_len)

    def tool_start(self, turn: int, round_num: int, tool: str, args: dict):
        if self._ls:
            self._ls.tool_start(self._sid, turn, round_num, tool, args)

    def tool_error(self, turn: int, round_num: int, tool: str, error: str):
        if self._ls:
            self._ls.tool_error(self._sid, turn, round_num, tool, error)

    # -- message logging -------------------------------------------------

    def compress(self, turn: int, round_num: int,
                 before_tokens: int, after_tokens: int,
                 before_msgs: int, after_msgs: int):
        if self._ls:
            self._ls.compress(self._sid, turn, round_num,
                              before_tokens, after_tokens,
                              before_msgs, after_msgs)

    def message(self, turn: int, round_num: int,
                seq: int, role: str, content: str,
                tool_call_id: str = ""):
        if self._ls:
            self._ls.message(self._sid, turn, round_num,
                              seq, role, content or "",
                              tool_call_id)

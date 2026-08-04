"""Thin wrapper around LogStore that makes logging calls safe to call
even when no logstore is configured (no-op fallback).

This eliminates repetitive ``if self.logstore:`` guards from agent.py.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corecoder.logstore import LogStore


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

    def chat_start(self, turn: int, input_len: int, msg_count: int):
        if self._ls:
            self._ls.chat_start(self._sid, turn, input_len, msg_count)

    def chat_done(self, turn: int, round_num: int, content_len: int):
        if self._ls:
            self._ls.chat_done(self._sid, turn, round_num, content_len)

    def tool_calls(self, turn: int, round_num: int, names: list[str]):
        if self._ls:
            self._ls.tool_calls(self._sid, turn, round_num, names)

    def tool_start(self, turn: int, round_num: int, tool: str, args: dict):
        if self._ls:
            self._ls.tool_start(self._sid, turn, round_num, tool, args)

    def tool_result(self, turn: int, round_num: int, tool: str, result_len: int):
        if self._ls:
            self._ls.tool_result(self._sid, turn, round_num, tool, result_len)

    def tool_error(self, turn: int, round_num: int, tool: str, error: str):
        if self._ls:
            self._ls.tool_error(self._sid, turn, round_num, tool, error)

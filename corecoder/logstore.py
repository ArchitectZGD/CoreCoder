"""MySQL-backed event log for agent execution traces.

Each method maps to a specific agent lifecycle event.  The module handles
connection, table creation, and insertion so that agent.py stays clean.
"""

import json
import uuid
from datetime import datetime

import pymysql


class LogStore:
    def __init__(self, **kwargs):
        defaults = {
            "host": "localhost",
            "port": 3306,
            "user": "corecoder",
            "password": "corecoder",
            "database": "corecoder",
            "charset": "utf8mb4",
            "autocommit": True,
        }
        defaults.update(kwargs)
        self._conn = pymysql.connect(**defaults)
        self._init_db()

    def _log(self, session_id, turn_id, round_num, event_type, **data):
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO events (session_id, turn_id, round_num, "
                "event_type, event_data) VALUES (%s, %s, %s, %s, %s)",
                (session_id, turn_id, round_num, event_type,
                 json.dumps(data, ensure_ascii=False)),
            )

    # ── public API ────────────────────────────────────────────────────

    def new_session(self, model: str = "") -> str:
        sid = str(uuid.uuid4())[:8]
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (id, model) VALUES (%s, %s)",
                (sid, model),
            )
        return sid

    def chat_start(self, sid: str, turn: int, input_len: int, msg_count: int):
        self._log(sid, turn, 0, "chat_start",
                  input_len=input_len, msg_count=msg_count)

    def chat_done(self, sid: str, turn: int, round_num: int, content_len: int):
        self._log(sid, turn, round_num, "chat_done",
                  content_len=content_len)

    def tool_calls(self, sid: str, turn: int, round_num: int, names: list[str]):
        self._log(sid, turn, round_num, "tool_calls", names=names)

    def tool_start(self, sid: str, turn: int, round_num: int,
                   tool: str, args: dict):
        self._log(sid, turn, round_num, "tool_start",
                  tool=tool, args={k: str(v)[:200] for k, v in args.items()})

    def tool_result(self, sid: str, turn: int, round_num: int,
                    tool: str, result_len: int):
        self._log(sid, turn, round_num, "tool_result",
                  tool=tool, result_len=result_len)

    def tool_error(self, sid: str, turn: int, round_num: int,
                   tool: str, error: str):
        self._log(sid, turn, round_num, "tool_error",
                  tool=tool, error=error)

    def close(self):
        self._conn.close()

"""MySQL-backed event log for agent execution traces.

Each method maps to a specific agent lifecycle event.  The module handles
connection, table creation, and insertion so that agent.py stays clean.
"""

import json
import uuid

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

    # -- internal --------------------------------------------------------

    def _init_db(self):
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id          VARCHAR(8)  PRIMARY KEY COMMENT '会话唯一标识',
                    model       VARCHAR(255) NOT NULL DEFAULT '' COMMENT '使用的LLM模型',
                    started_at  DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '会话开始时间'
                ) ENGINE=InnoDB COMMENT='CoreCoder会话表'
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id          BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '事件自增主键',
                    session_id  VARCHAR(8) NOT NULL COMMENT '关联会话ID',
                    turn_id     INT NOT NULL COMMENT '第几轮用户对话',
                    round_num   INT NOT NULL COMMENT '同一轮内第几次LLM调用',
                    source      VARCHAR(16) NOT NULL DEFAULT '' COMMENT '事件来源: user/llm/agent/tool',
                    target      VARCHAR(16) NOT NULL DEFAULT '' COMMENT '事件目标: user/llm/agent/tool',
                    event_type  VARCHAR(32) NOT NULL COMMENT '事件类型: chat_request/chat_done/tool_start/tool_error/compress/message',
                    event_data  JSON NOT NULL COMMENT '事件明细(JSON)',
                    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '事件发生时间',
                    INDEX idx_events_session (session_id),
                    INDEX idx_events_type (event_type)
                ) ENGINE=InnoDB COMMENT='CoreCoder执行事件表'
            """)
            # migrate existing tables that lack source/target columns
            for col in ("source", "target"):
                try:
                    cur.execute(f"""
                        ALTER TABLE events ADD COLUMN {col}
                        VARCHAR(16) NOT NULL DEFAULT ''
                        COMMENT '事件{col}' AFTER round_num
                    """)
                except Exception:
                    pass

    def _log(self, session_id, turn_id, round_num, event_type,
             source="", target="", **data):
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO events (session_id, turn_id, round_num, "
                "source, target, event_type, event_data) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (session_id, turn_id, round_num, source, target,
                 event_type, json.dumps(data, ensure_ascii=False)),
            )

    # -- session ---------------------------------------------------------

    def new_session(self, model: str = "") -> str:
        sid = str(uuid.uuid4())[:8]
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (id, model) VALUES (%s, %s)",
                (sid, model),
            )
        return sid

    # -- events ----------------------------------------------------------

    def message(self, sid: str, turn: int, round_num: int,
                seq: int, role: str, content: str,
                tool_call_id: str = ""):
        src = {"user": "user", "assistant": "llm", "tool": "tool"}.get(role, "")
        self._log(sid, turn, round_num, "message",
                  source=src, target="agent",
                  seq=seq, role=role, content=content,
                  tool_call_id=tool_call_id or None)

    def chat_request(self, sid: str, turn: int, round_num: int,
                     messages: list[dict], tools: list[dict]):
        self._log(sid, turn, round_num, "chat_request",
                  source="agent", target="llm",
                  msg_count=len(messages), tool_count=len(tools),
                  messages=messages, tools=tools)

    def chat_done(self, sid: str, turn: int, round_num: int, content_len: int):
        self._log(sid, turn, round_num, "chat_done",
                  source="agent", target="user",
                  content_len=content_len)

    def tool_start(self, sid: str, turn: int, round_num: int,
                   tool: str, args: dict):
        self._log(sid, turn, round_num, "tool_start",
                  source="agent", target="tool",
                  tool=tool, args={k: str(v) for k, v in args.items()})

    def tool_error(self, sid: str, turn: int, round_num: int,
                   tool: str, error: str):
        self._log(sid, turn, round_num, "tool_error",
                  source="tool", target="agent",
                  tool=tool, error=error)

    def compress(self, sid: str, turn: int, round_num: int,
                 before_tokens: int, after_tokens: int,
                 before_msgs: int, after_msgs: int):
        self._log(sid, turn, round_num, "compress",
                  source="agent", target="agent",
                  before_tokens=before_tokens, after_tokens=after_tokens,
                  before_msgs=before_msgs, after_msgs=after_msgs)

    def close(self):
        self._conn.close()

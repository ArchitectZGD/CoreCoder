import time
from .base import Tool

class NowTool(Tool):
    name = "Now"
    description = "Get the current local date and time. Use this when the user asks about the current time or you need a timestamp."
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    def execute(self, command: str, timeout: int = 120):
        return time.strftime("%Y-%m-%d %H:%M:%S")
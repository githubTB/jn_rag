"""
services/mcp_service.py - 本地 MCP 服务入口。

不需要单独启动远程进程，业务代码可直接 import 并调用。
"""

from __future__ import annotations

from dataclasses import dataclass

from services.prompt_templates.default_extract_profile import (
    SERVICE_NAME as DEFAULT_EXTRACT_PROMPT_SERVICE,
    q as DEFAULT_EXTRACT_Q,
    system_prompt as DEFAULT_EXTRACT_SYSTEM_PROMPT,
    user_prompt as DEFAULT_EXTRACT_USER_PROMPT,
)


@dataclass(frozen=True)
class MCPService:
    name: str
    q: str
    system_prompt: str
    user_prompt: str


_SERVICES: dict[str, MCPService] = {
    DEFAULT_EXTRACT_PROMPT_SERVICE: MCPService(
        name=DEFAULT_EXTRACT_PROMPT_SERVICE,
        q=DEFAULT_EXTRACT_Q,
        system_prompt=DEFAULT_EXTRACT_SYSTEM_PROMPT,
        user_prompt=DEFAULT_EXTRACT_USER_PROMPT,
    ),
}


def get_service(service_name: str) -> MCPService:
    service = _SERVICES.get(service_name)
    if not service:
        raise KeyError(f"未知模板服务: {service_name}")
    return service


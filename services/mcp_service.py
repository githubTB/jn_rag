"""
services/mcp_service.py - 本地 MCP 服务入口。

不需要单独启动远程进程，业务代码可直接 import 并调用。
"""

from __future__ import annotations

from dataclasses import dataclass

from services.prompt_templates.base_extract_prompt_template import (
    SERVICE_NAME as BASE_EXTRACT_MCP_SERVICE,
    q as BASE_EXTRACT_MCP_Q,
    system_prompt as BASE_EXTRACT_MCP_SYSTEM_PROMPT,
    user_prompt as BASE_EXTRACT_MCP_USER_PROMPT,
)
from services.prompt_templates.bulk_solid_waste_extract_prompt_template import (
    SERVICE_NAME as BULK_SOLID_WASTE_EXTRACT_MCP_SERVICE,
    q as BULK_SOLID_WASTE_EXTRACT_MCP_Q,
    system_prompt as BULK_SOLID_WASTE_EXTRACT_MCP_SYSTEM_PROMPT,
    user_prompt as BULK_SOLID_WASTE_EXTRACT_MCP_USER_PROMPT,
)
from services.prompt_templates.energy_extract_prompt_template import (
    SERVICE_NAME as ENERGY_EXTRACT_MCP_SERVICE,
    q as ENERGY_EXTRACT_MCP_Q,
    system_prompt as ENERGY_EXTRACT_MCP_SYSTEM_PROMPT,
    user_prompt as ENERGY_EXTRACT_MCP_USER_PROMPT,
)
from services.prompt_templates.major_production_equipment_extract_prompt_template import (
    SERVICE_NAME as MAJOR_PRODUCTION_EQUIPMENT_EXTRACT_MCP_SERVICE,
    q as MAJOR_PRODUCTION_EQUIPMENT_EXTRACT_MCP_Q,
    system_prompt as MAJOR_PRODUCTION_EQUIPMENT_EXTRACT_MCP_SYSTEM_PROMPT,
    user_prompt as MAJOR_PRODUCTION_EQUIPMENT_EXTRACT_MCP_USER_PROMPT,
)
from services.prompt_templates.materials_extract_prompt_template import (
    SERVICE_NAME as MATERIALS_EXTRACT_MCP_SERVICE,
    q as MATERIALS_EXTRACT_MCP_Q,
    system_prompt as MATERIALS_EXTRACT_MCP_SYSTEM_PROMPT,
    user_prompt as MATERIALS_EXTRACT_MCP_USER_PROMPT,
)
from services.prompt_templates.pollutants_extract_prompt_template import (
    SERVICE_NAME as POLLUTANTS_EXTRACT_MCP_SERVICE,
    q as POLLUTANTS_EXTRACT_MCP_Q,
    system_prompt as POLLUTANTS_EXTRACT_MCP_SYSTEM_PROMPT,
    user_prompt as POLLUTANTS_EXTRACT_MCP_USER_PROMPT,
)
from services.prompt_templates.product_extract_prompt_template import (
    SERVICE_NAME as PRODUCT_EXTRACT_MCP_SERVICE,
    q as PRODUCT_EXTRACT_MCP_Q,
    system_prompt as PRODUCT_EXTRACT_MCP_SYSTEM_PROMPT,
    user_prompt as PRODUCT_EXTRACT_MCP_USER_PROMPT,
)


@dataclass(frozen=True)
class MCPService:
    name: str
    q: str
    system_prompt: str
    user_prompt: str


_SERVICES: dict[str, MCPService] = {
    BASE_EXTRACT_MCP_SERVICE: MCPService(
        name=BASE_EXTRACT_MCP_SERVICE,
        q=BASE_EXTRACT_MCP_Q,
        system_prompt=BASE_EXTRACT_MCP_SYSTEM_PROMPT,
        user_prompt=BASE_EXTRACT_MCP_USER_PROMPT,
    ),
    BULK_SOLID_WASTE_EXTRACT_MCP_SERVICE: MCPService(
        name=BULK_SOLID_WASTE_EXTRACT_MCP_SERVICE,
        q=BULK_SOLID_WASTE_EXTRACT_MCP_Q,
        system_prompt=BULK_SOLID_WASTE_EXTRACT_MCP_SYSTEM_PROMPT,
        user_prompt=BULK_SOLID_WASTE_EXTRACT_MCP_USER_PROMPT,
    ),
    ENERGY_EXTRACT_MCP_SERVICE: MCPService(
        name=ENERGY_EXTRACT_MCP_SERVICE,
        q=ENERGY_EXTRACT_MCP_Q,
        system_prompt=ENERGY_EXTRACT_MCP_SYSTEM_PROMPT,
        user_prompt=ENERGY_EXTRACT_MCP_USER_PROMPT,
    ),
    MAJOR_PRODUCTION_EQUIPMENT_EXTRACT_MCP_SERVICE: MCPService(
        name=MAJOR_PRODUCTION_EQUIPMENT_EXTRACT_MCP_SERVICE,
        q=MAJOR_PRODUCTION_EQUIPMENT_EXTRACT_MCP_Q,
        system_prompt=MAJOR_PRODUCTION_EQUIPMENT_EXTRACT_MCP_SYSTEM_PROMPT,
        user_prompt=MAJOR_PRODUCTION_EQUIPMENT_EXTRACT_MCP_USER_PROMPT,
    ),
    MATERIALS_EXTRACT_MCP_SERVICE: MCPService(
        name=MATERIALS_EXTRACT_MCP_SERVICE,
        q=MATERIALS_EXTRACT_MCP_Q,
        system_prompt=MATERIALS_EXTRACT_MCP_SYSTEM_PROMPT,
        user_prompt=MATERIALS_EXTRACT_MCP_USER_PROMPT,
    ),
    POLLUTANTS_EXTRACT_MCP_SERVICE: MCPService(
        name=POLLUTANTS_EXTRACT_MCP_SERVICE,
        q=POLLUTANTS_EXTRACT_MCP_Q,
        system_prompt=POLLUTANTS_EXTRACT_MCP_SYSTEM_PROMPT,
        user_prompt=POLLUTANTS_EXTRACT_MCP_USER_PROMPT,
    ),
    PRODUCT_EXTRACT_MCP_SERVICE: MCPService(
        name=PRODUCT_EXTRACT_MCP_SERVICE,
        q=PRODUCT_EXTRACT_MCP_Q,
        system_prompt=PRODUCT_EXTRACT_MCP_SYSTEM_PROMPT,
        user_prompt=PRODUCT_EXTRACT_MCP_USER_PROMPT,
    ),
}


def get_service(service_name: str) -> MCPService:
    service = _SERVICES.get(service_name)
    if not service:
        raise KeyError(f"未知模板服务: {service_name}")
    return service


def list_service_names() -> list[str]:
    return sorted(_SERVICES.keys())


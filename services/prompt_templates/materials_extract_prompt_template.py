"""
原材料结构化抽取配置（单文件三变量）。
"""

SERVICE_NAME = "materials_extract_prompt_template"

# 检索问题/抽取主题默认值
q = "提取企业原材料情况、原材料采购情况、原材料库存情况"

# system 提示词
system_prompt = "你是一个数据提取分析师，基于提供的文本，提取出其中原材料消耗的数值信息和关联数值信息。输出数据列表"

# user 提示词模板（必须包含 {text} 占位符）
user_prompt = """
任务：从文本中抽取企业信息并输出JSON。

字段：
    名称
    消耗量
    单位
    消耗环节
    单位产品消耗量
    备注

要求：
    - 仅输出 JSON
    - 缺失字段为 null
    - 不要解释

文本：
{text}
""".strip()


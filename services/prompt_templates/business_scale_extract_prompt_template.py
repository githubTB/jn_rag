"""
企业经营规模结构化抽取配置。
"""

SERVICE_NAME = "buiness_scale_extract_prompt_template"

# 检索问题/抽取主题默认值
q = "提取{company_name}的经营规模情况"

# system 提示词
system_prompt = "你是一个数据提取分析师，基于提供的文本，提取出其中的数值信息和关联数值信息。"

# user 提示词模板（必须包含 {text} 占位符）
user_prompt = """
任务：从文本中抽取企业信息并输出JSON。

字段：
    员工总人数
    生产人员数
    管理人员数
    注册资本
    固定资产总值

输出：
    JSON对象

要求：
    - 不要解释
    - 不要markdown
    - 不要换行格式化
    - 缺失字段为null

文本：
{text}
""".strip()
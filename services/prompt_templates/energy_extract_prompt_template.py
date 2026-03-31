"""
能源结构化抽取配置（单文件三变量）。
"""

SERVICE_NAME = "energy_extract_prompt_template"

# 检索问题/抽取主题默认值
q = "提取{company_name}企业能源能源消耗情况。列如：水、电、气、热、油、煤、天然气、石油等"

# system 提示词
system_prompt = "你是一个数据提取分析师，基于提供的文本，提取出其中能源消耗近三年的数值信息和关联数值信息。"

# user 提示词模板（必须包含 {text} 占位符）
user_prompt = """
任务：从文本中抽取企业信息并输出JSON。

字段：
    能源品种
	年度
	实物量
	实物量单位
	当量值折标系数
	当量值
	当量值单位
	等价值折标系数
	等价值
	等价值单位

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


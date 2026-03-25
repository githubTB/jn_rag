"""
产品结构化抽取配置（单文件三变量）。
"""

SERVICE_NAME = "product_extract_prompt_template"

# 检索问题/抽取主题默认值
q = "提取企业近三年的产品产量及相关数值情况"

# system 提示词
system_prompt = system_prompt = """
你是一个专业的数据结构化抽取引擎。

你的任务是：
从给定文本中识别表格或数值型信息，并严格按照指定JSON结构输出。

必须：
- 只输出合法JSON
- 不输出解释
- 不输出markdown
- 不输出多余文本
- 不补充文本中不存在的数据
"""
# user 提示词模板（必须包含 {text} 占位符）
user_prompt = """
产品产量可能包含字段：
    产品名称
    年份
    产品数量
    数量单位
    产值
    产值单位
    工业增加值
    工业增加值单位

输出JSON格式以字段映射为准：
    例子：产品名称:产品A

要求：
    - 不要解释
    - 不要markdown
    - 不要换行格式化
    - 缺失字段为null

文本：
{text}
""".strip()


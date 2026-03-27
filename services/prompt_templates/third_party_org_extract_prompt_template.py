"""
产品结构化抽取配置（单文件三变量）。
"""

SERVICE_NAME = "third_party_org_extract_prompt_template"

# 检索问题/抽取主题默认值
q = "提取{company_name}的关于绿色生产水平评估报告的第三方机构相关数值情况"

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
第三方评估机构可能包含字段：
    评估咨询机构
    评估负责人
    单位负责人
    评估机构负责人
    报告审核人
    报告负责人
    报告编制人
    评估时间

输出JSON格式以字段映射为准：
    例子：评估咨询机构:重庆节能利用监测中心

要求：
    - 不要解释
    - 不要markdown
    - 不要换行格式化
    - 缺失字段为null

文本：
{text}
""".strip()


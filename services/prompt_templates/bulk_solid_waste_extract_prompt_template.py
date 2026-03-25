"""
固体废物结构化抽取配置（单文件三变量）。
"""

SERVICE_NAME = "bulk_solid_waste_extract_prompt_template"

# 检索问题/抽取主题默认值
q = "提取企业大宗固废情况、废物处理情况、废物利用情况"

# system 提示词
system_prompt = "你是一个数据提取分析师，基于提供的文本，提取出其中的数值信息和关联数值信息，输出列表数据。"

# user 提示词模板（必须包含 {text} 占位符）
user_prompt = """
任务：从文本中抽取企业信息并输出JSON。

字段：
    年份
        固废名称
    	固废种类
    	固废单位
    	固废产生量
    	固废处置方式
    	固废回收利用量
    	固废综合利用率

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


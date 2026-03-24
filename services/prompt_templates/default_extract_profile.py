"""
默认结构化抽取配置（单文件三变量）。
"""

SERVICE_NAME = "default_extract_profile"

# 检索问题/抽取主题默认值
q = "提取企业基础信息与关键财务指标"

# system 提示词
system_prompt = "你是一个数据提取分析师，基于提供的文本，提取出其中的数值信息和关联数值信息。"

# user 提示词模板（必须包含 {text} 占位符）
user_prompt = """
任务：从文本中抽取企业信息并输出JSON。

字段：
    company_name 企业名称
    company_credit_code 统一社会信用代码
    company_registered_type 注册类型
    company_registered_capital 注册资本
    company_registered_address 注册地址
    company_establishment_date 成立日期
    company_legal_representative 企业法定代表人
    company_office_address 办公地址
    company_contact_phone 联系电话
    company_contact_email 联系邮箱
    company_industry 行业分类
    company_region_code 注册企业区名称
    company_remark 备注
    revenue 营业收入（数字）
    net_profit 净利润（数字）
    listed_status 是否上市（是/否）

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


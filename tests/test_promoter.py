import time
from openai import OpenAI

client = OpenAI(
    api_key="ssh_jn_46d26532ccc1acbb66d9dd8a160c27a683e925f93fd2b3ffc7516e00c493a96a",
    base_url="http://117.157.162.135:8000/v1"  # 换成你的实际地址
)

start_time = time.time()

system_prompt = """
你是一个专业的用地集约化潜力分析助手，负责根据企业提供的基本信息，撰写企业用地集约化潜力分析描述。
"""

user_prompt = """
请基于提供的企业，按照以下结构和要求，撰写企业用地集约化潜力分析描述：
【铝型材企业基本信息】

    主要生产设备信息：序号 名称 单位 数据
                    1   工厂用地面积 m2 77037
                    2   总计容建/构筑物建筑面积
                    （计容建筑面积）    m2 79379
                    3   总建/构筑占地面积  m2 32949
                    6   产值（2024年）  万元 29794

【核心任务】
            根据《机械行业绿色工厂评价 导则》（JB/T 14407-2023）、《工业项目建设用地控制指标》（2023版）对
            企业的单位面积产值、提升土地利用效率等方面进行潜力分析，将企业信息整理成标准格式的描述文本。

【输出结构】
           第一部分：用地集约化潜力分析

【内容要求】
        - 严格基于提供的资料，不添加虚构信息
        - 时间线索要清晰准确，使用具体年份
        - 专业术语使用准确（如生产线类型、产品规格等）
        - 数据要精确，包括产能、员工人数等具体数值
        - 语言风格：正式、客观、简洁
        - 段落之间逻辑连贯
        - 段落之间用换行分隔，但不要标注"第一部分"、"第二部分"

【格式要求】
        - 第一部分使用简洁的项目符号格式
        - 第一部分使用连贯的段落描述
        - 适当使用分段，避免长段落堆砌
        - 数字和单位使用规范（如“40万t/a”、“3000t/d”）

【示例参考】
        用地集约化潜力分析
        在企业建设设计过程中，严格遵循了用地集约化的原则。厂区建筑布局紧凑且合理，采用多层建筑设计，功能分区明确清晰，既全面满足了生产需求，
        又显著节约了土地资源。企业建筑容积率为1.03，建筑密度为42.8%，满足《工业项目建设用地控制指标》（2023版）中汽车制造业容积率不低于0.8、
        建筑密度不低于40%的要求，且达到先进值水平。企业2024年单位用地面积产值为38.7亿元/km2，低于重庆市2023年底制造业亩均产值734.17万元/亩
        （110.2亿元/km2）水平，具有71.5亿元/km2的差距，存在较大的提升潜力。若单位用地面积产值达制造业亩均产值水平，按照现有用地面积77037m2计算，
        产值预计将提升至8.49亿元。因此，评估组建议企业在后续生产中尽快按环评内容启动建设剩余的注塑生产线、铝型材生产线及压铸件生产线，进一步优化生产
        布局，提升土地利用效率，同时积极探索产品创新，提高产品附加值，以实现单位用地面积产值的提升。
        现在，请根据我提供的企业资料，按照上述要求撰写企业能源低碳化潜力分析。

"""

system_prompt = """
你是一个数据提取分析师，基于提供的文本，提取出其中的数值信息。
"""

user_prompt = f"""
任务：从文本中抽取企业信息并输出JSON。

字段：
    company_name 企业名称
    credit_code 统一社会信用代码
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
"""
response = client.chat.completions.create(
    model="any",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    max_tokens=2000,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)

print(response.choices[0].message.content)

end_time = time.time()
usage = response.usage
print(f"输入tokens: {usage.prompt_tokens}")
print(f"输出tokens: {usage.completion_tokens}")
print(f"总tokens:   {usage.total_tokens}")
print(f"耗时: {end_time - start_time} 秒")
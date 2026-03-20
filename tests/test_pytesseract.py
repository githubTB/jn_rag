from ollama import chat
import base64

#image_path = "uploads/净化工程-空调机组设备铭牌.jpg"
image_path = "uploads/2020年水电气汇总统计表.jpg"
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# 调用本地 qwen3.5:2b 模型
response = chat(
    model="glm-ocr:latest",
    messages=[
        {"role": "system", "content": "你是一个OCR助手，负责识别图片中的文字，表格，图表等。保持内容原样，如果是表格，空白可以用-占位"},
        {"role": "user", "content": f"你好，图片内容是{image_b64}"}
    ]
)

print("模型回答:", response.message.content)
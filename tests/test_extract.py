import os
import sys
import logging
logging.basicConfig(level=logging.DEBUG)

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["VL_BACKEND"] = "vllm-server"
os.environ["VL_SERVER_URL"] = "http://117.157.162.55:8118/v1"

# sys.path 必须在 import 之前设置
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extract_processor import ExtractProcessor

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_path = os.path.join(project_root, 'uploads', '2020年水电气汇总统计表.jpg')
    print(f"Testing with image: {image_path}")

    docs = ExtractProcessor.extract(image_path, use_layout_detection=False)
    for i, doc in enumerate(docs):
        print(f"\n--- 块 {i} ({doc.metadata.get('label', 'unknown')}) ---")
        print(doc.page_content)

"""
最小化调试脚本：直接调用 PaddleOCRVL，打印 result 的完整结构。
用法：python debug_vl.py /path/to/image.jpg
"""
import sys
import os
import json
import tempfile

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
print(f"图片: {image_path}")

from paddleocr import PaddleOCRVL

print("初始化 PaddleOCRVL...")
pipeline = PaddleOCRVL(device="cpu")
print(f"pipeline 类型: {type(pipeline).__name__}")
print(f"pipeline 方法: {[m for m in dir(pipeline) if not m.startswith('_')]}\n")

print("开始 predict...")
raw = list(pipeline.predict(image_path))
print(f"\npredict 返回: {len(raw)} 个结果\n")

for i, result in enumerate(raw):
    print(f"{'='*60}")
    print(f"result[{i}] 类型: {type(result).__name__}")
    print(f"result[{i}] 属性: {[a for a in dir(result) if not a.startswith('_')]}")

    # 检查 .res
    if hasattr(result, "res"):
        res = result.res
        print(f"\nresult.res 类型: {type(res).__name__}")
        if isinstance(res, dict):
            print(f"result.res keys: {list(res.keys())}")
            for k, v in res.items():
                preview = str(v)[:300]
                print(f"  res[{k!r}] ({type(v).__name__}): {preview}")
        else:
            print(f"result.res 值: {str(res)[:300]}")

    # 尝试 save_to_json
    if hasattr(result, "save_to_json"):
        tmp = tempfile.mktemp(suffix=".json")
        try:
            result.save_to_json(tmp)
            with open(tmp, encoding="utf-8") as f:
                data = json.load(f)
            print(f"\nsave_to_json 内容:")
            print(json.dumps(data, ensure_ascii=False, indent=2)[:1000])
        except Exception as e:
            print(f"save_to_json 失败: {e}")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # 尝试 save_to_markdown
    if hasattr(result, "save_to_markdown"):
        tmp = tempfile.mktemp(suffix=".md")
        try:
            result.save_to_markdown(tmp)
            with open(tmp, encoding="utf-8") as f:
                md = f.read()
            print(f"\nsave_to_markdown 内容 ({len(md)} 字符):")
            print(md[:500])
        except Exception as e:
            print(f"save_to_markdown 失败: {e}")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

if not raw:
    print("predict 返回空列表！检查以下可能原因：")
    print("  1. 图片路径不存在")
    print("  2. 图片格式不支持")
    print("  3. VLM worker 静默失败")
    print(f"\n图片是否存在: {os.path.exists(image_path)}")
    print(f"图片大小: {os.path.getsize(image_path) / 1024:.1f} KB")
    
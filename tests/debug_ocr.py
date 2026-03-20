import sys
import cv2
from paddleocr import PaddleOCR

def main():
    if len(sys.argv) < 2:
        print("用法：python debug_ocr.py 图片路径")
        return

    img_path = sys.argv[1]
    print("🔍 正在使用 PaddleOCR V5 识别...")

    # ============================
    # ✅ V5 
    # ============================
    ocr = PaddleOCR(lang="ch")
    img = cv2.imread(img_path)

    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化，增强文字
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 找表格竖线
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 13))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # 找轮廓，确定左侧列范围
    contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    right_x = img.shape[1]  # 默认整图宽度

    # 找最左侧竖线的 x 坐标作为左列右边界
    if contours:
        left_line = min(contours, key=lambda c: cv2.boundingRect(c)[0])
        x, y, w, h = cv2.boundingRect(left_line)
        right_x = x + w

    # 裁剪左侧列
    left_col = img[:, :right_x]

    # OCR 识别左列文字
    result = ocr.predict(left_col)

    # 输出识别结果
    print("🔍 输出识别结果...")
    print(result)

if __name__ == "__main__":
    main()
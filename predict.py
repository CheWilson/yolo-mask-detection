
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 口罩檢測預測腳本 (Windows 相容版本)
訓練完成後使用此腳本進行預測
執行前請先安裝: pip install ultralytics opencv-python
"""

from ultralytics import YOLO
import cv2
import os

# ==================== 配置區 ====================
# 模型路徑（訓練完成後會自動生成）
MODEL_PATH = os.path.join('runs', 'detect', 'runs', 'mask_detection', 'weights', 'best.pt')

# 預測模式（選擇一種）
# 'image'  - 單張圖片
# 'folder' - 整個資料夾
# 'camera' - 即時攝像頭
PREDICT_MODE = 'camera'

# 單張圖片路徑（PREDICT_MODE = 'image' 時使用）
IMAGE_PATH = 'test_image.jpg'   # C:\Users\User\Desktop\dataset\test_image.jpg

# 資料夾路徑（PREDICT_MODE = 'folder' 時使用）
FOLDER_PATH = os.path.join('yolo_dataset', 'images', 'test')

# 攝像頭設定（PREDICT_MODE = 'camera' 時使用）
CAMERA_ID = 0   # 默認攝像頭

# 信心值門檢（低於此值的檢測會被過慾）
CONFIDENCE = 0.5

# ================================================


def predict_image(model, image_path):
    """單張圖片預測"""
    if not os.path.exists(image_path):
        print(f" 找不到圖片: {image_path}")
        return

    print(f"\n 預測圖片: {image_path}")
    results = model(image_path, conf=CONFIDENCE)

    # 顯示結果
    for r in results:
        print(f"\n  檢測到 {len(r.boxes)} 個物體:")
        if len(r.boxes) == 0:
            print("    沒有檢測到任何物體")
        for box in r.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"    - {class_name}: {confidence:.2%}")

    # 保存標記圖片
    output_path = 'prediction_result.jpg'
    results[0].save(output_path)
    print(f"\n   結果已保存: {output_path}")


def predict_folder(model, folder_path):
    """批量預測資料夾中的圖片"""
    if not os.path.exists(folder_path):
        print(f" 找不到資料夾: {folder_path}")
        return

    print(f"\n 批量預測: {folder_path}")
    results = model(folder_path, conf=CONFIDENCE, save=True)
    print(f"\n   完成！結果已保存到 runs/predict/ 目錄")


def predict_camera(model, camera_id=0):
    """即時攝像頭預測"""
    print(f"\n 開啟攝像頭 (ID: {camera_id})...")
    print("   按下 'q' 鍵退出\n")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f" 無法開啟攝像頭 {camera_id}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" 無法讀取攝像頭畫面")
            break

        # 進行預測
        results = model(frame, conf=CONFIDENCE, verbose=False)

        # 在畫面上畫出結果
        annotated_frame = results[0].plot()

        # 顯示視窗
        cv2.imshow('Mask Detection', annotated_frame)

        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n   攝像頭已關閉")


def main():
    print("=" * 60)
    print(" YOLO 口罩檢測 - 預測工具")
    print("=" * 60)

    # 檢查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"\n 找不到模型: {MODEL_PATH}")
        print("請先執行 train_yolo.py 訓練模型")
        return

    # 載入模型
    print(f"\n 載入模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✓ 模型載入成功")

    # 根據模式執行預測
    if PREDICT_MODE == 'image':
        predict_image(model, IMAGE_PATH)

    elif PREDICT_MODE == 'folder':
        predict_folder(model, FOLDER_PATH)

    elif PREDICT_MODE == 'camera':
        predict_camera(model, CAMERA_ID)

    else:
        print(f" 未知的預測模式: {PREDICT_MODE}")
        print("請將 PREDICT_MODE 設為: 'image', 'folder', 或 'camera'")

    print("\n" + "=" * 60)
    print(" 預測完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
from ultralytics import YOLO
import os

# ==================== 配置區 ====================
# 模型選擇
# 'yolov8n.pt' - Nano : 最快，適合測試和邊緣設備
# 'yolov8s.pt' - Small: 平衡速度和精度（推薦入門）
# 'yolov8m.pt' - Medium: 更高精度
# 'yolov8l.pt' - Large : 高精度
# 'yolov8x.pt' - XLarge: 最高精度
MODEL_SIZE = 'yolov8n.pt'

# 訓練參數
EPOCHS     = 100    # 訓練輪數
IMG_SIZE   = 640    # 輸入圖片大小
BATCH_SIZE = 16     # 批次大小（顯存不足時改為 8 或 4）
DEVICE     = 0      # GPU編號；無GPU時改為 'cpu'
WORKERS    = 0      # 數據加載線程數（Windows 建議用 0）

# 學習率
LR0  = 0.01   # 初始學習率
LRF  = 0.01   # 最終學習率係數

# 早停
PATIENCE = 20   # 幾輪沒進展就停止

# 輸出
PROJECT = 'runs'              # 項目根目錄
NAME    = 'mask_detection'    # 實驗名稱

# ================================================


def get_yaml_path():
    """自動找到 dataset.yaml"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path  = os.path.join(script_dir, 'yolo_dataset', 'dataset.yaml')

    if not os.path.exists(yaml_path):
        print(f" 找不到配置文件: {yaml_path}")
        print("請先執行 convert_xml_to_yolo.py 生成數據集")
        return None
    return yaml_path


def train_model():
    """訓練YOLO模型"""
    print("=" * 60)
    print(" 開始訓練 YOLO 口罩檢測模型")
    print("=" * 60)

    yaml_path = get_yaml_path()
    if yaml_path is None:
        return None

    print(f"\n 訓練配置:")
    print(f"   模型       : {MODEL_SIZE}")
    print(f"   數據集     : {yaml_path}")
    print(f"   訓練輪數   : {EPOCHS}")
    print(f"   圖片大小   : {IMG_SIZE}")
    print(f"   批次大小   : {BATCH_SIZE}")
    print(f"   設備       : {DEVICE}")
    print(f"   輸出目錄   : {PROJECT}/{NAME}\n")

    try:
        print(f" 載入預訓練模型: {MODEL_SIZE}")
        model = YOLO(MODEL_SIZE)

        print("\n 開始訓練...\n")
        results = model.train(
            data=yaml_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
            optimizer='auto',
            lr0=LR0,
            lrf=LRF,
            patience=PATIENCE,
            project=PROJECT,
            name=NAME,
            save=True,
            plots=True,
            val=True,
            verbose=True,
        )

        best_path = os.path.join(PROJECT, NAME, 'weights', 'best.pt')
        last_path = os.path.join(PROJECT, NAME, 'weights', 'last.pt')

        print("\n" + "=" * 60)
        print(" 訓練完成！")
        print("=" * 60)
        print(f"\n 模型保存位置:")
        print(f"   最佳模型 : {best_path}")
        print(f"   最後模型 : {last_path}")
        print(f"\n 訓練結果目錄: {PROJECT}/{NAME}")
        print(f"   results.png          - 訓練曲線圖")
        print(f"   confusion_matrix.png - 混淆矩陣")
        print(f"\n 下一步: 執行 predict.py 進行預測")
        print("=" * 60)

        return results

    except Exception as e:
        print(f"\n 訓練錯誤: {str(e)}")
        print("\n請檢查:")
        print("  1. 數據集路徑是否正確")
        print("  2. GPU記憶體是否足夠（可以減小 BATCH_SIZE）")
        print("  3. ultralytics 是否已安裝")
        return None


if __name__ == '__main__':
    train_model()

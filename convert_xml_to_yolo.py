import xml.etree.ElementTree as ET
import os
import shutil
from pathlib import Path
import random

# ==================== 配置區 ====================
# 類別定義（請根據你的實際類別調整）
CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']

# 資料集分割比例
TRAIN_RATIO = 0.7   # 70% 訓練集
VAL_RATIO = 0.2     # 20% 驗證集
TEST_RATIO = 0.1    # 10% 測試集

# 路徑配置 - 使用相對路徑（相對於腳本所在目錄）
SOURCE_IMAGES_DIR = 'images'           # 原始圖片目錄
SOURCE_ANNOTATIONS_DIR = 'annotations' # 原始XML標註目錄
OUTPUT_DIR = 'yolo_dataset'            # 輸出目錄

# ================================================


def convert_bbox_to_yolo(size, box):
    """
    將邊界框從絕對座標轉換為YOLO格式（相對座標）
    
    Args:
        size: (width, height, depth) 圖片尺寸
        box: (xmin, ymin, xmax, ymax) 邊界框絕對座標
    
    Returns:
        (center_x, center_y, width, height) YOLO格式相對座標（0-1之間）
    """
    dw = 1.0 / size[0]  # width縮放因子
    dh = 1.0 / size[1]  # height縮放因子
    
    # 計算中心點和寬高
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]
    
    # 轉換為相對座標
    center_x = center_x * dw
    center_y = center_y * dh
    width = width * dw
    height = height * dh
    
    return (center_x, center_y, width, height)


def convert_xml_to_yolo(xml_file):
    """
    將單個XML文件轉換為YOLO格式
    
    Args:
        xml_file: XML文件路徑
    
    Returns:
        list: YOLO格式的標註行列表
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 獲取圖片尺寸
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    img_depth = int(size.find('depth').text)
    
    yolo_annotations = []
    
    # 處理每個object
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        
        # 檢查類別是否在定義的類別中
        if class_name not in CLASSES:
            print(f"警告: 未知類別 '{class_name}' 在文件 {xml_file}，已跳過")
            continue
        
        class_id = CLASSES.index(class_name)
        
        # 獲取邊界框座標
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # 轉換為YOLO格式
        yolo_bbox = convert_bbox_to_yolo(
            (img_width, img_height, img_depth),
            (xmin, ymin, xmax, ymax)
        )
        
        # 格式化輸出（保留6位小數）
        yolo_line = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
        yolo_annotations.append(yolo_line)
    
    return yolo_annotations


def get_image_filename_from_xml(xml_file):
    """從XML文件中獲取對應的圖片文件名"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root.find('filename').text


def create_directory_structure(output_dir):
    """創建YOLO數據集目錄結構"""
    dirs = [
        'images/train',
        'images/val',
        'images/test',
        'labels/train',
        'labels/val',
        'labels/test'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(output_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"✓ 創建目錄: {full_path}")


def split_dataset(xml_files):
    """分割數據集為訓練/驗證/測試集"""
    # 隨機打亂
    random.shuffle(xml_files)
    
    total = len(xml_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))
    
    train_files = xml_files[:train_end]
    val_files = xml_files[train_end:val_end]
    test_files = xml_files[val_end:]
    
    return {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }


def process_dataset():
    """主處理函數"""
    print("=" * 60)
    print("開始轉換XML標註為YOLO格式")
    print("=" * 60)
    
    # 獲取腳本所在目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n 當前工作目錄: {script_dir}")
    
    # 構建完整路徑
    source_images = os.path.join(script_dir, SOURCE_IMAGES_DIR)
    source_annotations = os.path.join(script_dir, SOURCE_ANNOTATIONS_DIR)
    output_path = os.path.join(script_dir, OUTPUT_DIR)
    
    print(f" 圖片目錄: {source_images}")
    print(f" 標註目錄: {source_annotations}")
    print(f" 輸出目錄: {output_path}")
    
    # 檢查源目錄是否存在
    if not os.path.exists(source_images):
        print(f"\n 錯誤: 圖片目錄不存在: {source_images}")
        print(f"請確認以下目錄結構:")
        print(f"{script_dir}/")
        print(f"  ├── images/")
        print(f"  ├── annotations/")
        print(f"  └── convert_xml_to_yolo.py")
        return
    
    if not os.path.exists(source_annotations):
        print(f"\n 錯誤: 標註目錄不存在: {source_annotations}")
        print(f"請確認以下目錄結構:")
        print(f"{script_dir}/")
        print(f"  ├── images/")
        print(f"  ├── annotations/")
        print(f"  └── convert_xml_to_yolo.py")
        return
    
    # 創建輸出目錄結構
    print("\n 創建目錄結構...")
    create_directory_structure(output_path)
    
    # 獲取所有XML文件
    xml_files = list(Path(source_annotations).glob('*.xml'))
    print(f"\n 找到 {len(xml_files)} 個XML標註文件")
    
    if len(xml_files) == 0:
        print(" 沒有找到XML文件，請檢查路徑")
        return
    
    # 分割數據集
    print("\n 分割數據集...")
    dataset_split = split_dataset(xml_files)
    print(f"  訓練集: {len(dataset_split['train'])} 個文件 ({TRAIN_RATIO*100}%)")
    print(f"  驗證集: {len(dataset_split['val'])} 個文件 ({VAL_RATIO*100}%)")
    print(f"  測試集: {len(dataset_split['test'])} 個文件 ({TEST_RATIO*100}%)")
    
    # 處理每個分割
    total_converted = 0
    total_errors = 0
    
    for split_name, xml_file_list in dataset_split.items():
        print(f"\n🔄 處理 {split_name} 集...")
        
        for xml_file in xml_file_list:
            try:
                # 轉換XML到YOLO格式
                yolo_annotations = convert_xml_to_yolo(xml_file)
                
                # 獲取對應的圖片文件名
                img_filename = get_image_filename_from_xml(xml_file)
                img_source_path = os.path.join(source_images, img_filename)
                
                # 檢查圖片是否存在
                if not os.path.exists(img_source_path):
                    print(f"    警告: 圖片不存在 {img_filename}，跳過此標註")
                    total_errors += 1
                    continue
                
                # 準備輸出路徑
                txt_filename = Path(img_filename).stem + '.txt'
                img_dest_path = os.path.join(output_path, 'images', split_name, img_filename)
                txt_dest_path = os.path.join(output_path, 'labels', split_name, txt_filename)
                
                # 複製圖片
                shutil.copy2(img_source_path, img_dest_path)
                
                # 寫入YOLO標註
                with open(txt_dest_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                total_converted += 1
                
                if total_converted % 50 == 0:
                    print(f"  已處理 {total_converted} 個文件...")
                
            except Exception as e:
                print(f"   處理文件 {xml_file.name} 時出錯: {str(e)}")
                total_errors += 1
    
    # 創建dataset.yaml配置文件
    print("\n 創建dataset.yaml配置文件...")
    
    # 使用絕對路徑
    yaml_content = f"""# YOLO Dataset Configuration
# 口罩檢測數據集

path: {output_path}  # 數據集根目錄
train: images/train  # 訓練集圖片路徑（相對於path）
val: images/val      # 驗證集圖片路徑
test: images/test    # 測試集圖片路徑

# 類別
nc: {len(CLASSES)}  # 類別數量
names: {CLASSES}  # 類別名稱列表

# 類別說明:
# 0: with_mask - 正確佩戴口罩
# 1: without_mask - 未佩戴口罩
# 2: mask_weared_incorrect - 口罩佩戴不正確
"""
    
    yaml_path = os.path.join(output_path, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✓ 配置文件已保存: {yaml_path}")
    
    # 顯示統計信息
    print("\n" + "=" * 60)
    print(" 轉換完成!")
    print("=" * 60)
    print(f"成功轉換: {total_converted} 個文件")
    print(f"錯誤/跳過: {total_errors} 個文件")
    print(f"\n輸出目錄: {output_path}")
    print(f"配置文件: {yaml_path}")
    print("\n下一步:")
    print("1. 檢查輸出目錄中的文件是否正確")
    print("2. 使用以下命令開始訓練:")
    print("   from ultralytics import YOLO")
    print(f"   model = YOLO('yolov8n.pt')")
    print(f"   model.train(data='{yaml_path}', epochs=100)")
    print("=" * 60)


if __name__ == '__main__':
    # 設置隨機種子以便結果可復現
    random.seed(42)
    
    # 執行轉換
    process_dataset()

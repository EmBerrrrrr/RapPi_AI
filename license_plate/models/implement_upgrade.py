from ultralytics import YOLO
import numpy as np
import cv2

# Load 2 model
lp_model = YOLO("LP_best.pt")
ocr_model = YOLO("Letters_detection.pt")

CHAR_MAP = {
    0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",
    10:"A",11:"B",12:"C",13:"D",14:"E",15:"F",16:"G",17:"H",
    18:"I",19:"J",20:"K",21:"L",22:"M",23:"N",24:"O",25:"P",
    26:"Q",27:"R",28:"S",29:"T",30:"U",31:"V",32:"W",33:"X",
    34:"Y",35:"Z"
}

def format_plate(lines):
    if len(lines) == 1:
        return lines[0]
    line1, line2 = lines
    return f"{line1}{line2}"

def ocr_read(ocr):
    chars = []

    # Lấy tất cả ký tự từ OCR
    for box in ocr.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        cls = int(box.cls.item())
        conf = float(box.conf.item())

        if conf < 0.40:
            continue
        
        chars.append(((x1, y1), CHAR_MAP.get(cls, "?")))

    if len(chars) == 0:
        return ""

    # Sort ký tự theo X (trái → phải) luôn
    chars = sorted(chars, key=lambda c: c[0][0])

    # Kiểm tra sự khác biệt Y để quyết định 1 dòng hay 2 dòng
    ys = [c[0][1] for c in chars]
    y_range = max(ys) - min(ys)

    # Ngưỡng phân biệt 1 dòng vs 2 dòng, có thể điều chỉnh theo dataset
    threshold_y = 15  # ví dụ pixel, tùy ảnh resize

    if y_range < threshold_y:
        # 1 dòng → ghép tất cả
        final = "".join([c[1] for c in chars])
    else:
        # 2 dòng → tách theo Y trung bình
        threshold = np.mean(ys)
        line1 = [c for c in chars if c[0][1] < threshold]
        line2 = [c for c in chars if c[0][1] >= threshold]

        # sort mỗi dòng theo X
        line1 = sorted(line1, key=lambda c: c[0][0])
        line2 = sorted(line2, key=lambda c: c[0][0])

        # đảm bảo dòng trên có Y nhỏ hơn dòng dưới
        if line1 and line2:
            if np.mean([c[0][1] for c in line1]) > np.mean([c[0][1] for c in line2]):
                line1, line2 = line2, line1

        text1 = "".join([c[1] for c in line1])
        text2 = "".join([c[1] for c in line2])

        final = format_plate([text for text in [text1, text2] if text != ""])

    return final



# ------------------------------
#   MULTI-PLATE PIPELINE
# ------------------------------
def read_multi_plates(img):
    result = lp_model(img)[0]
    plates = result.boxes

    final_outputs = []

    h, w = img.shape[:2]

    for p in plates:
        x1, y1, x2, y2 = map(int, p.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = img[y1:y2, x1:x2]

        # OCR từng biển số
        ocr_out = ocr_model(crop)[0]
        plate_text = ocr_read(ocr_out)
        final_outputs.append(plate_text)

        # vẽ bounding box + text
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, plate_text, (x1, y1-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    return final_outputs, img

# ------------------------------
#   TEST TRÊN 1 ẢNH NHIỀU BIỂN SỐ
# ------------------------------
if __name__ == "__main__":
    img_path = "557.png"  # ảnh có nhiều xe/biển
    img = cv2.imread(img_path)

    if img is None:
        print("Không đọc được ảnh, kiểm tra đường dẫn!")
    else:
        plates, out_img = read_multi_plates(img)
        print("Danh sách biển số nhận diện:", plates)
        cv2.imshow("Result", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

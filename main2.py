import cv2
import dlib
import time
import csv
import datetime
import numpy as np

# 初始化人臉偵測器與特徵點預測器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# AU 權重設定
AU_WEIGHTS = {
    "AU1": 2, "AU2": 1, "AU4": 3,
    "AU7": 3, "AU12": 2, "AU15": 2,
    "AU20": 3, "AU23": 3, "AU24": 3,
}

DECAY_THRESHOLD = 3  # 超過3秒沒偵測就扣掉該AU的分數

# 加強 AU 條件限制版本
def analyze_AUs(landmarks):
    detected = []

    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

    # AU1：兩邊眉毛顯著上抬
    if (points[21][1] < points[27][1] - 8) and (points[22][1] < points[27][1] - 8):
        detected.append("AU1")

    # AU4：眉毛內側明顯接近眼角
    if (dist(points[21], points[39]) < 25) and (dist(points[22], points[42]) < 25):
        detected.append("AU4")

    # AU7：眼睛外側明顯收縮
    if (dist(points[36], points[39]) < 15) and (dist(points[42], points[45]) < 15):
        detected.append("AU7")

    # AU12：嘴角兩邊都大幅上揚
    if (points[48][1] < points[57][1] - 8) and (points[54][1] < points[57][1] - 8):
        detected.append("AU12")

    # AU15：嘴角兩邊明顯下垂
    if (points[48][1] > points[57][1] + 3) and (points[54][1] > points[57][1] + 3):
        detected.append("AU15")

    # AU20：嘴巴寬度非常大
    if dist(points[48], points[54]) > 65:
        detected.append("AU20")

    # AU23：上下唇幾乎貼合
    if dist(points[62], points[66]) < 3:
        detected.append("AU23")

    return detected

# 初始化攝影機與資料儲存
cap = cv2.VideoCapture(0)
csv_file = open("facs_scores.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Detected_AUs", "Score"])

score = 0
last_update = time.time()
au_accumulator = []
au_counter = {}
au_last_seen = {}  # 記錄每個 AU 最後偵測時間

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        detected_aus = analyze_AUs(landmarks)
        au_accumulator.extend(detected_aus)

    current_time = time.time()
    if current_time - last_update >= 1:
        if au_accumulator:
            unique_aus = set(au_accumulator)

            for au in unique_aus:
                au_last_seen[au] = current_time  # 更新最後偵測時間
                if au in au_counter:
                    au_counter[au] += 1
                else:
                    au_counter[au] = 1

        # 動態計算分數（僅保留 X 秒內有偵測到的 AU）
        active_aus = []
        for au in AU_WEIGHTS.keys():
            if au in au_last_seen and current_time - au_last_seen[au] <= DECAY_THRESHOLD:
                active_aus.append(au)

        score = sum([AU_WEIGHTS[au] for au in active_aus])

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([timestamp, ", ".join(active_aus), score])
        csv_file.flush()

        au_accumulator.clear()
        last_update = current_time

    # 顯示分數
    text = f"Score: {score}"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(frame, text, (frame.shape[1] - text_width - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 重置提示
    reset_text = "Press 'r' to reset score"
    cv2.putText(frame, reset_text, (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("FACS Lie Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        score = 0
        au_accumulator.clear()
        au_counter.clear()
        au_last_seen.clear()

cap.release()
csv_file.close()
cv2.destroyAllWindows()

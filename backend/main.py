import io
import base64
from typing import Tuple

import cv2
import numpy as np
import mediapsipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ====== Настройка приложения ======
app = FastAPI(title="Hand Photo Processor")

# Разрешаем CORS для разработки (замени на конкретные хосты в продакшне)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Разрешить все источники для теста
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Инициализация MediaPipe (руки) ======
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Настройки MediaPipe: одна рука, средняя точность работы
hands_detector = mp_hands.Hands(static_image_mode=True,  # для отдельных кадров (фото)
                                max_num_hands=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)


# ====== Утилиты ======
def read_imagefile(file: UploadFile) -> np.ndarray:
    """
    Прочитать UploadFile (multipart/form-data) в OpenCV-изображение (BGR).
    """
    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")
    return img


def encode_image_to_dataurl(img_bgr: np.ndarray, quality: int = 90) -> str:
    """
    Кодирует изображение BGR в data:image/jpeg;base64,...
    """
    ret, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        raise RuntimeError("Failed to encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def count_fingers_from_landmarks(landmarks: mp_hands.HandLandmark, handedness: str = "Right") -> int:
    """
    Простейшая логика подсчёта поднятых пальцев на основе позиций ключевых точек.
    Возвращает количество пальцев (0..5).
    handedness указывается как 'Right' или 'Left' (если доступно).
    Алгоритм:
      - Большой палец: сравнение по X (вправо/влево) в зависимости от руки.
      - Остальные пальцы: сравнение по Y (кончик пальца выше/ниже среднего сустава).
    landmarks — объект mp.solutions.hands.NormalizedLandmarkList
    """
    tip_ids = [4, 8, 12, 16, 20]  # индексы кончиков пальцев (thumb, index, middle, ring, pinky)
    fingers = []

    # Большой палец: сравниваем x коорд. кончика и x предшествующей точки (в зависимости от ориентации руки)
    # У normalized landmarks x возрастаёт слева направо (в системе изображения после flip не учитывать).
    try:
        # Thumb logic: для правой руки кончик thumb (4).x > point(3).x когда палец открыт (в "зеркальном" отображении можно инвертировать)
        thumb_tip = landmarks.landmark[tip_ids[0]]
        thumb_ip = landmarks.landmark[tip_ids[0] - 1]

        if handedness == "Right":
            fingers.append(1 if thumb_tip.x > thumb_ip.x else 0)
        else:
            fingers.append(1 if thumb_tip.x < thumb_ip.x else 0)
    except Exception:
        fingers.append(0)

    # Остальные пальцы: кончик выше (меньше y) чем сустав (tip vs tip-2)
    for i in range(1, 5):
        try:
            tip = landmarks.landmark[tip_ids[i]]
            pip = landmarks.landmark[tip_ids[i] - 2]  # PIP joint для пальца
            fingers.append(1 if tip.y < pip.y else 0)
        except Exception:
            fingers.append(0)

    return sum(fingers)


# ====== Эндпоинты ======

@app.post("/process-photo")
async def process_photo(file: UploadFile = File(...)):
    """
    Принимает multipart/form-data с полем file (фото руки),
    возвращает JSON: { count: int, image: data_url }
    где image — аннотированное изображение в base64.
    """
    # Прочитать файл в OpenCV-формат
    img_bgr = read_imagefile(file)

    # Копия для рисования (чтобы не портить оригинал)
    output_img = img_bgr.copy()

    # Преобразуем BGR -> RGB для MediaPipe
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # MediaPipe ждёт нормализованные координаты; запускаем детектор на статическом изображении
    results = hands_detector.process(img_rgb)

    if results.multi_hand_landmarks is None or len(results.multi_hand_landmarks) == 0:
        # Нет рук на фото
        annotated = encode_image_to_dataurl(output_img)
        return JSONResponse({"count": 0, "image": annotated, "message": "No hand detected"})

    # Берём первую руку
    hand_landmarks = results.multi_hand_landmarks[0]

    # Попробуем определить сторону (handedness) — MediaPipe даёт информацию в multi_handedness (если включено)
    # В static_image_mode True multi_handedness может быть пустым; обработаем безопасно.
    handedness_label = None
    try:
        # results.multi_handedness — список Classification, берем 0-й
        if results.multi_handedness and len(results.multi_handedness) > 0:
            handedness_label = results.multi_handedness[0].classification[0].label  # 'Left' или 'Right'
    except Exception:
        handedness_label = None

    # Нарисуем ключевые точки и связи ладони
    mp_drawing.draw_landmarks(
        output_img,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
    )

    # Подсчёт пальцев
    count = count_fingers_from_landmarks(hand_landmarks, handedness=handedness_label or "Right")

    # Наложим текст на изображение
    cv2.putText(output_img, f"Fingers: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    # Опционально: можно нарисовать bounding box (вычисляем по всем landmark-координатам)
    h, w, _ = output_img.shape
    xs = [int(lmk.x * w) for lmk in hand_landmarks.landmark]
    ys = [int(lmk.y * h) for lmk in hand_landmarks.landmark]
    x1, x2 = max(min(xs) - 10, 0), min(max(xs) + 10, w)
    y1, y2 = max(min(ys) - 10, 0), min(max(ys) + 10, h)
    cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Кодируем изображение в data-url
    data_url = encode_image_to_dataurl(output_img)

    # Возвращаем JSON с количеством и аннотированным изображением
    return JSONResponse({"count": int(count), "image": data_url})


@app.get("/")
async def root():
    return {"status": "ok", "message": "Hand processor ready. POST /process-photo with form file field."}


# ====== Завершение ======
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:main", host="0.0.0.0", port=8000, reload=True)
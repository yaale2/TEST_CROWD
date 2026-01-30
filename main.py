import argparse
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import numpy as np
import torch

def parse_arguments():
    """
    Парсит аргументы командной строки.
    
    Returns:
        argparse.Namespace: Объект с аргументами.
    """
    parser = argparse.ArgumentParser(description="Детекция людей на видео с использованием YOLO.")
    parser.add_argument('--input', default='crowd.mp4', type=str, help='Путь к входному видео файлу.')
    parser.add_argument('--output', default='output.mp4', type=str, help='Путь к выходному видео файлу.')
    parser.add_argument('--confidence', default=0.4, type=float, help='Порог уверенности для детекции.')
    parser.add_argument('--use_tracker', action='store_true', help='Включить трекинг объектов.')
    return parser.parse_args()

def load_model(model_name):
    """
    Загружает модель YOLO.
    
    Args:
        model_name (str): Имя файла с весами.
    
    Returns:
        YOLO: Загруженная модель.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_name)
    model.to(device)
    print(f"Модель {model_name} загружена на {device}.")
    return model

def process_video(input_path, output_path, model, conf_threshold, use_tracker=False):
    """
    Обрабатывает видео: детектирует людей, рисует bounding boxes и сохраняет результат.
    
    Args:
        input_path (str): Путь к входному видео.
        output_path (str): Путь к выходному видео.
        model (YOLO): Модель для детекции.
        conf_threshold (float): Порог уверенности.
        use_tracker (bool): Включить трекинг.
    """
    # Чтение видео
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {input_path}")
    
    # Получаем свойства видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Создаем writer для выходного видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    missed_frames = defaultdict(int)
    max_missed_frames = 15
    box_history = defaultdict(lambda: deque(maxlen=15))
   
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        if use_tracker:
            results = model.track(frame, conf=conf_threshold, iou=0.25, classes=[0], persist=True)
            current_boxes = {}
            for box in results[0].boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                track_id = int(box.id[0])
                if track_id == -1:
                    continue
                current_boxes[track_id] = (x1, y1, x2, y2, conf)
                box_history[track_id].append((x1, y1, x2, y2))
                missed_frames[track_id] = 0  

            to_remove = []
            for track_id in list(box_history.keys()):
                if track_id not in current_boxes:
                    missed_frames[track_id] += 1
                    if missed_frames[track_id] > max_missed_frames:
                        to_remove.append(track_id)
            
            for track_id in to_remove:
                del box_history[track_id]
                del missed_frames[track_id]

            for track_id, hist in box_history.items():
                if len(hist) == 0:
                    continue
                if track_id in current_boxes:
                    x1, y1, x2, y2, conf = current_boxes[track_id]
                else:
                    avg_box = tuple(map(int, smooth_box(hist)))
                    x1, y1, x2, y2 = avg_box
                    conf = 0.0
                
                label = f'person {conf:.2f} ID:{track_id}'
                draw_box(frame, x1, y1, x2, y2, label)
        else:
            results = model.predict(frame, conf=conf_threshold, iou=0.25, classes=[0])
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                label = f'person {conf:.2f}'
                draw_box(frame, x1, y1, x2, y2, label)
        
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Обработанное видео сохранено в {output_path}")

def draw_box(frame, x1, y1, x2, y2, label):
    """
    Рисует bounding box с лейблом на кадре.
    
    Args:
        frame (np.array): Кадр видео.
        x1, y1, x2, y2 (int): Координаты bounding box.
        label (str): Текст лейбла (класс + conf + ID если трекинг).
    """
    # Рисуем прямоугольник (зеленый, толщина 2, полупрозрачный)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    alpha = 0.6  # Прозрачность
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Рисуем текст (над box)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def smooth_box(history, alpha=0.6):
    """
    Выполняет экспоненциальное сглаживание координат bounding box, 
    чтобы уменьшить дрожание объектов при визуализации.

    Args:
        history (collections.deque | list): Последовательность предыдущих координат
            прямоугольников в формате (x1, y1, x2, y2).
        alpha (float, optional): Коэффициент сглаживания.
            Значение ближе к 1 делает сглаживание сильнее.

    Returns:
        np.ndarray: Сглаженные координаты прямоугольника в формате (x1, y1, x2, y2).
    """
    history = list(history)
    smoothed = np.array(history[0])
    for box in history[1:]:
        smoothed = alpha * np.array(box) + (1 - alpha) * smoothed
    return smoothed

if __name__ == "__main__":
    args = parse_arguments()
    model = load_model('best.pt')
    process_video(args.input, args.output, model, args.confidence, args.use_tracker)
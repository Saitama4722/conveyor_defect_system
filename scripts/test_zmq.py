import zmq
import cv2
import numpy as np
import sys

ZMQ_PORT = 5555
ZMQ_ADDR = f"tcp://localhost:{ZMQ_PORT}"
TOPIC_FRONT = b"cam_front"
TOPIC_SIDE  = b"cam_side"

def main() -> None:
    print(f"[INFO] Подключение к {ZMQ_ADDR}")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVTIMEO, 3000)
    socket.setsockopt(zmq.SUBSCRIBE, TOPIC_FRONT)
    socket.setsockopt(zmq.SUBSCRIBE, TOPIC_SIDE)
    socket.connect(ZMQ_ADDR)
    print("[INFO] Ожидание кадров (3 секунды таймаут)...")
    print("[INFO] Убедитесь что Webots запущен и симуляция идёт (Play)")

    received = {b"cam_front": 0, b"cam_side": 0}

    for _ in range(50):
        try:
            parts = socket.recv_multipart()
            topic, data = parts[0], parts[1]
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                received[topic] += 1
                name = topic.decode()
                h, w = frame.shape[:2]
                print(f"[OK] {name}: кадр {received[topic]}, размер {w}x{h}")
                # Сохранить кадр в файл (нет GUI у OpenCV)
                cv2.imwrite(f"scripts/test_frame_{name}.jpg", frame)
            else:
                print(f"[WARN] Не удалось декодировать кадр от {topic.decode()}")
        except zmq.Again:
            print("[WARN] Таймаут — кадры не приходят")
            break

    socket.close()
    context.term()

    print(f"\n[ИТОГ] cam_front: {received[TOPIC_FRONT]} кадров")
    print(f"[ИТОГ] cam_side:  {received[TOPIC_SIDE]} кадров")
    if received[TOPIC_FRONT] > 0 and received[TOPIC_SIDE] > 0:
        print("[SUCCESS] ZMQ работает — обе камеры передают кадры")
    else:
        print("[FAIL] Кадры не получены — проверьте что Webots запущен")

if __name__ == "__main__":
    main()

"""ZMQ-приёмник кадров от контроллера Webots.

Подписывается на топики ``cam_front`` и ``cam_side``, декодирует JPEG в BGR
numpy-массивы и складывает их в две независимые очереди фиксированного
размера. Чтение очередей — неблокирующее, при переполнении старый кадр
выкидывается в пользу нового.
"""

from __future__ import annotations

import queue
import threading

import cv2
import numpy as np
import zmq


ZMQ_PORT: int = 5555
ZMQ_CONNECT_ADDR: str = f"tcp://localhost:{ZMQ_PORT}"
TOPIC_CAM_FRONT: bytes = b"cam_front"
TOPIC_CAM_SIDE: bytes = b"cam_side"
RECONNECT_DELAY: float = 1.0   # секунд между попытками переподключения
MAX_QUEUE_SIZE: int = 5        # максимум кадров в очереди каждой камеры

RECV_TIMEOUT_MS: int = 500
CAMERA_FRONT: str = "front"
CAMERA_SIDE: str = "side"


class FrameReceiver:
    """Фоновый приёмник кадров с двух камер."""

    def __init__(self, port: int = ZMQ_PORT) -> None:
        self.port: int = port
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._queue_front: queue.Queue[np.ndarray] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._queue_side: queue.Queue[np.ndarray] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._lock: threading.Lock = threading.Lock()

    def connect(self) -> None:
        """Открыть ZMQ-контекст и подключить SUB-сокет к контроллеру."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, TOPIC_CAM_FRONT)
        self._socket.setsockopt(zmq.SUBSCRIBE, TOPIC_CAM_SIDE)
        self._socket.setsockopt(zmq.RCVTIMEO, RECV_TIMEOUT_MS)
        addr = f"tcp://localhost:{self.port}"
        self._socket.connect(addr)

    def _decode_frame(self, raw: bytes) -> np.ndarray | None:
        """Декодировать JPEG-байты в BGR ndarray."""
        if not raw:
            return None
        buf = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return frame

    def _enqueue(self, q: queue.Queue[np.ndarray], frame: np.ndarray) -> None:
        """Положить кадр в очередь, при переполнении — выкинуть самый старый."""
        with self._lock:
            try:
                q.put_nowait(frame)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    pass

    def _receive_loop(self) -> None:
        """Главный цикл фонового потока приёма."""
        while self._running:
            if self._socket is None:
                break
            try:
                parts = self._socket.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                continue

            if len(parts) < 2:
                continue

            topic, frame_bytes = parts[0], parts[1]
            frame = self._decode_frame(frame_bytes)
            if frame is None:
                continue

            if topic == TOPIC_CAM_FRONT:
                self._enqueue(self._queue_front, frame)
            elif topic == TOPIC_CAM_SIDE:
                self._enqueue(self._queue_side, frame)

    def start(self) -> None:
        """Подключиться и запустить фоновый поток приёма."""
        self.connect()
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Остановить поток и закрыть сокет/контекст."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def get_frame(self, camera: str) -> np.ndarray | None:
        """Достать самый свежий кадр для камеры ``"front"`` или ``"side"``."""
        if camera == CAMERA_FRONT:
            q = self._queue_front
        elif camera == CAMERA_SIDE:
            q = self._queue_side
        else:
            raise ValueError(f"Неизвестная камера: {camera!r}")
        try:
            return q.get_nowait()
        except queue.Empty:
            return None

    def is_connected(self) -> bool:
        """Признак активного приёма."""
        return self._running and self._socket is not None

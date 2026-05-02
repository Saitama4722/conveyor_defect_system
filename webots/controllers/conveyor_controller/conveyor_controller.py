"""Контроллер конвейера для Webots R2025a (режим Supervisor)."""

from __future__ import annotations
import random
import cv2
import numpy as np
import zmq
from controller import Supervisor, Camera, Motor  # type: ignore

TIMESTEP: int = 16
CONVEYOR_SPEED_DEFAULT: float = 0.1
ZMQ_PORT: int = 5555
JPEG_QUALITY: int = 85
FRAME_SKIP: int = 2
CAMERA_FRONT_NAME: str = "camera_front"
CAMERA_SIDE_NAME: str = "camera_side"
BELT_MOTOR_NAME: str = "belt_motor"
TOPIC_CAM_FRONT: bytes = b"cam_front"
TOPIC_CAM_SIDE: bytes = b"cam_side"
ZMQ_BIND_ADDRESS: str = f"tcp://*:{ZMQ_PORT}"
ZMQ_CMD_PORT: int = 5556
ZMQ_CMD_ADDR: str = f"tcp://*:{ZMQ_CMD_PORT}"
SPEED_MIN: float = 0.0
SPEED_MAX: float = 1.0

# Пресеты позиций камер (Z-up)
CAM_PRESETS: dict[str, dict[str, list]] = {
    "front_top": {
        "translation": [0.118, -0.621, 2.55],
        "rotation":    [-0.525, 0.519, 0.674, 2.02],
    },
    "front_side": {
        "translation": [0.116, -2.57, 0.862],
        "rotation":    [-0.12, 0.144, 0.982, 1.62],
    },
    "side_top": {
        "translation": [0.118, -0.621, 2.55],
        "rotation":    [-0.525, 0.519, 0.674, 2.02],
    },
    "side_side": {
        "translation": [0.116, -2.57, 0.862],
        "rotation":    [-0.12, 0.144, 0.982, 1.62],
    },
}

PRODUCT_NAMES: list[str] = ["metal_nut_0", "screw_0", "bracket_0"]

# Z-up: (X — длина ленты, Y — поперёк, Z — высота)
PRODUCT_SPAWN: dict[str, tuple[float, float, float]] = {
    "metal_nut_0": (-0.8, 0.0, 0.18),
    "screw_0":     (-0.5, 0.0, 0.18),
    "bracket_0":   (-0.2, 0.0, 0.18),
}


class ConveyorController:
    def __init__(self) -> None:
        self.robot: Supervisor = Supervisor()
        self.timestep: int = int(self.robot.getBasicTimeStep())
        print(f"[DEBUG] basicTimeStep = {self.timestep} мс")
        print(f"[DEBUG] dx за шаг = {CONVEYOR_SPEED_DEFAULT * (self.timestep / 1000.0):.6f} м")

        self.camera_front: Camera = self.robot.getDevice(CAMERA_FRONT_NAME)
        self.camera_side: Camera = self.robot.getDevice(CAMERA_SIDE_NAME)
        self.camera_front.enable(self.timestep)
        self.camera_side.enable(self.timestep)

        self.belt_motor = self.robot.getDevice(BELT_MOTOR_NAME)
        if self.belt_motor:
            self.belt_motor.setPosition(float("inf"))
            self.belt_motor.setVelocity(CONVEYOR_SPEED_DEFAULT)
        else:
            print("[WARN] belt_motor не найден")

        self.zmq_context: zmq.Context = zmq.Context()
        self.zmq_socket: zmq.Socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind(ZMQ_BIND_ADDRESS)

        # Сокет команд (REP) для управления камерами из GUI
        self._cmd_context: zmq.Context = zmq.Context()
        self._cmd_socket: zmq.Socket = self._cmd_context.socket(zmq.REP)
        self._cmd_socket.setsockopt(zmq.RCVTIMEO, 0)
        self._cmd_socket.bind(ZMQ_CMD_ADDR)
        print(f"[INFO] ZMQ CMD сокет готов на порту {ZMQ_CMD_PORT}")

        self.frame_count: int = 0
        self.current_speed: float = CONVEYOR_SPEED_DEFAULT

        self._product_nodes: dict = {}
        for name in PRODUCT_NAMES:
            node = self.robot.getFromDef(name.upper())
            if node:
                self._product_nodes[name] = node
                node.getField("translation").setSFVec3f(
                    list(PRODUCT_SPAWN[name]))
                node.resetPhysics()
            else:
                print(f"[WARN] {name.upper()} не найден")
        self._step: int = 0
        self._conveyor_running: bool = True
        print(f"[DEBUG] Найдено объектов: {len(self._product_nodes)}")

        self._active: bool = True
        print("[INFO] Режим Supervisor активен — объекты управляются контроллером")

    def _get_frame(self, camera: Camera) -> np.ndarray | None:
        raw = camera.getImage()
        if raw is None:
            return None
        width = camera.getWidth()
        height = camera.getHeight()
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
        return bgra[:, :, :3]

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else b""

    def _send_frames(self) -> None:
        frame_front = self._get_frame(self.camera_front)
        frame_side = self._get_frame(self.camera_side)
        if frame_front is not None:
            jpeg = self._encode_frame(frame_front)
            if jpeg:
                self.zmq_socket.send_multipart([TOPIC_CAM_FRONT, jpeg])
        if frame_side is not None:
            jpeg = self._encode_frame(frame_side)
            if jpeg:
                self.zmq_socket.send_multipart([TOPIC_CAM_SIDE, jpeg])

    def _process_commands(self) -> None:
        # неблокирующий приём команд управления камерами
        try:
            msg = self._cmd_socket.recv_string()
        except zmq.Again:
            return
        # формат: "set_camera:<cam>:<preset>" или "conveyor:start|stop"
        if msg == "conveyor:start":
            self._step = 0
            self._conveyor_running = True
            for name, node in self._product_nodes.items():
                node.getField("translation").setSFVec3f(
                    list(PRODUCT_SPAWN[name]))
                node.resetPhysics()
            self._cmd_socket.send_string("OK")
            return
        if msg == "conveyor:stop":
            self._conveyor_running = False
            self._cmd_socket.send_string("OK")
            return
        parts = msg.split(":")
        if parts[0] == "spawn_product" and len(parts) == 2:
            product_name = parts[1]
            if product_name in self._product_nodes:
                # скрыть все объекты
                for name, node in self._product_nodes.items():
                    node.getField("translation").setSFVec3f(
                        [-0.5, 0.0, -1.0])
                    node.resetPhysics()
                # показать только выбранный
                node = self._product_nodes[product_name]
                spawn = PRODUCT_SPAWN[product_name]
                node.getField("translation").setSFVec3f(list(spawn))
                node.resetPhysics()
                print(f"[INFO] Деталь на ленте: {product_name}")
                self._cmd_socket.send_string("OK")
            else:
                self._cmd_socket.send_string(
                    f"ERROR: unknown product {product_name}")
            return
        if parts[0] == "set_camera" and len(parts) == 3:
            cam_name = parts[1]
            preset = parts[2]
            preset_key = f"{cam_name}_{preset}"
            if preset_key in CAM_PRESETS:
                p = CAM_PRESETS[preset_key]
                def_name = "CAMERA_FRONT" if cam_name == "front" else "CAMERA_SIDE"
                node = self.robot.getFromDef(def_name)
                if node:
                    node.getField("translation").setSFVec3f(p["translation"])
                    node.getField("rotation").setSFRotation(p["rotation"])
                    print(f"[INFO] Камера {def_name} → пресет {preset_key}")
                    self._cmd_socket.send_string("OK")
                else:
                    self._cmd_socket.send_string("ERROR: node not found")
            else:
                self._cmd_socket.send_string("ERROR: unknown preset")
        else:
            self._cmd_socket.send_string("ERROR: bad format")

    def _move_products(self) -> None:
        self._step += 1
        if not self._conveyor_running:
            return
        if self._step == 120:
            print("[INFO] Конвейер несёт объекты")
            for name, node in self._product_nodes.items():
                pos = node.getField("translation").getSFVec3f()
                print(f"[Z] {name} Z={pos[2]:.3f}")
        # сброс упавших объектов
        for name, node in self._product_nodes.items():
            pos = node.getField("translation").getSFVec3f()
            if pos[0] > 1.1 or pos[2] < 0.03:
                node.getField("translation").setSFVec3f(
                    list(PRODUCT_SPAWN[name]))
                node.resetPhysics()
                print(f"[INFO] {name} сброшен")

    def set_speed(self, speed: float) -> None:
        clamped = max(SPEED_MIN, min(SPEED_MAX, speed))
        if self.belt_motor:
            self.belt_motor.setVelocity(clamped)
        self.current_speed = clamped

    def run(self) -> None:
        while self.robot.step(self.timestep) != -1:
            self._process_commands()
            self._move_products()
            if self.frame_count % FRAME_SKIP == 0:
                self._send_frames()
            self.frame_count += 1


if __name__ == "__main__":
    controller = ConveyorController()
    controller.run()

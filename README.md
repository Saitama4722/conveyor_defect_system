# Neural Network Visual Inspection System for Industrial Product Quality Control

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Webots R2025a](https://img.shields.io/badge/Webots-R2025a-007ACC?logo=robotframework&logoColor=white)](https://cyberbotics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-OBB-00FFFF?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

End-to-end visual defect detection pipeline for a simulated industrial conveyor line: **Webots R2025a** physics scene → **ZeroMQ** dual-camera stream → **YOLOv8n-OBB + GhostConv** detector → **Tkinter** operator HMI with live analytics.

<!-- TODO: insert demo.gif here (e.g. docs/demo.gif) — recommended 10–15 s capture of GUI with both cameras streaming live detections -->

---

## 🚀 Overview

`conveyor_defect_system` is a research prototype of an automated optical inspection (AOI) station that classifies surface defects on industrial parts moving along a conveyor belt. The simulator (Webots R2025a) hosts a physically-driven conveyor with two cameras (front and side); a Python detection service consumes the JPEG stream over ZMQ and runs an oriented-bounding-box detector based on YOLOv8n with a GhostConv backbone; an operator-facing Tkinter GUI displays the live feed, defect log, and real-time defect statistics.

The system was developed as a bachelor's thesis project at **RTU MIREA** (РТУ МИРЭА), study group **КРБО-03-22**, direction **15.03.06 — Mechatronics and Robotics**, defended in 2026.

## 🎯 Key Results

All technical-specification requirements were exceeded by a wide margin:

| Metric | Requirement | Achieved | Status |
| --- | --- | --- | --- |
| Detection quality (mAP@0.5) | ≥ 0.75 | **0.955** | ✅ |
| Precision | — | **0.970** | ✅ |
| Recall | — | **0.874** | ✅ |
| Frame processing speed | ≤ 100 ms | **3.1 ms** | ✅ |
| Video stream resolution | ≥ 640×480 | **1920×1080** | ✅ |
| Number of product types | ≥ 3 | **3** (cylinder, bar/box, L-bracket) | ✅ |
| Number of cameras | 2 | **2** (front + side) | ✅ |
| Training time (RTX 4070, 100 epochs) | — | **~11 min** | ✅ |

The detector recognizes **8 defect classes**: `bent`, `scratch`, `color`, `broken_large`, `broken_small`, `contamination`, `thread_side`, `thread_top`.

## 🛠️ Technology Stack

| Layer | Technology |
| --- | --- |
| 3D simulation | Webots R2025a (conveyor scene, supervisor controller) |
| Detection model | YOLOv8n-OBB + GhostConv (Han et al., CVPR 2020) |
| DL framework | PyTorch 2.x + CUDA 12.x |
| Training / inference | Ultralytics |
| Image processing | OpenCV 4.x |
| Inter-process transport | ZeroMQ (pyzmq) — PUB/SUB + REQ/REP |
| GUI | Tkinter + Matplotlib (TkAgg backend) |
| Training data | MVTec AD dataset (converted to YOLO OBB format) |
| Language | Python 3.10+ |

## 📁 Project Structure

```text
conveyor_defect_system/
├── detection/
│   ├── model/
│   │   └── best.pt              ← trained YOLOv8n-OBB weights
│   ├── ghost_conv.py            ← GhostConv module
│   ├── inference.py             ← DefectDetector class
│   └── zmq_receiver.py          ← dual-camera ZMQ frame receiver
├── gui/
│   └── app.py                   ← Tkinter GUI + Matplotlib charts
├── scripts/
│   ├── train.py                 ← YOLOv8n-OBB training pipeline
│   ├── convert_masks_to_yolo.py ← MVTec mask → YOLO OBB converter
│   └── test_zmq.py              ← ZMQ connection test utility
├── webots/
│   ├── worlds/
│   │   └── conveyor.wbt         ← Webots scene file
│   └── controllers/
│       └── conveyor_controller/
│           └── conveyor_controller.py ← Supervisor controller
├── requirements.txt
└── README.md
```

## 🧠 Architecture

```text
        ┌────────────────────────┐
        │  Webots R2025a Sim     │
        │  (conveyor + 2 cams)   │
        └────────────┬───────────┘
                     │ JPEG frames
                     ▼
        ┌────────────────────────┐         ┌─────────────────────┐
        │  ZMQ PUB :5555         │────────▶│  ZMQ SUB (Python)   │
        │  topics: cam_front,    │         │  zmq_receiver.py    │
        │          cam_side      │         └──────────┬──────────┘
        └────────────────────────┘                    │ frames
                     ▲                                ▼
                     │                     ┌─────────────────────┐
                     │                     │  DefectDetector     │
                     │                     │  YOLOv8n-OBB +      │
                     │                     │  GhostConv          │
                     │                     └──────────┬──────────┘
                     │                                │ detections
                     │                                ▼
        ┌────────────────────────┐         ┌─────────────────────┐
        │  Webots Controller     │◀────────│  Tkinter GUI        │
        │  (camera + conveyor    │  cmds   │  app.py             │
        │   control)             │         │  + Matplotlib       │
        └────────────────────────┘         └─────────────────────┘
                     ▲                                ▲
                     │       ZMQ REQ :5556            │
                     └────────────────────────────────┘
```

## ⚡ Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt
```

```bash
# 2. Place pretrained weights
#    Copy best.pt to:  detection/model/best.pt
```

```bash
# 3. Launch the simulator
#    Open Webots → File → Open World → webots/worlds/conveyor.wbt
#    Press ▶ (Play) to start the simulation
```

```bash
# 4. Wait for the Webots console to print:
#    [INFO] Конвейер несёт объекты
```

```bash
# 5. Start the operator GUI
python gui/app.py
#    then click "Старт" inside the GUI window
```

## 📦 Installation

### System requirements

| Component | Minimum | Recommended |
| --- | --- | --- |
| OS | Windows 10/11 or Ubuntu 20.04+ | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 | 3.11 |
| Webots | R2025a | R2025a |
| GPU | — (CPU works) | NVIDIA + CUDA 12.x |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB | 10 GB (with datasets) |

### Step-by-step

```bash
# Clone the repository
git clone https://github.com/Saitama4722/conveyor_defect_system.git
cd conveyor_defect_system

# Create virtual environment
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: PyTorch with CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install Webots R2025a from [cyberbotics.com](https://cyberbotics.com) and, in `Tools → Preferences → General`, point the Python command to your venv interpreter (`venv/Scripts/python.exe` on Windows or `venv/bin/python` on Linux).

## 🎮 GUI Reference

The operator interface (`gui/app.py`) is built around a single dark-themed Tkinter window:

- **Video panel** — 1460×480 live feed with overlaid OBB detections.
- **Camera toggle** — switch between *Camera 1 (front)* and *Camera 2 (side)*.
- **Control buttons** — `Старт` (start conveyor) and `Стоп` (stop conveyor).
- **Object dropdown** — `Деталь на ленте`: select which part the simulator spawns next.
- **Camera preset buttons** — `⬆ Сверху` (top-down view) and `↔ Сбоку` (side view) for each camera.
- **Fullscreen** — double-click the video panel to enter 1280×720 fullscreen.
- **Stats panel** — running total of detections and the currently dominant defect class.
- **Detection table** — chronological log: *Time / Camera / Class / Confidence*.
- **Chart 1** — horizontal bar chart of the top-3 defects, updated live.
- **Chart 2** — detection-rate dynamics line chart, updated live.

## 📡 ZMQ Protocol

The Webots controller and the Python services communicate over two ZMQ sockets:

### Port 5555 — PUB/SUB (video)

JPEG-encoded frames published on two topics:

```python
b"cam_front"   # front camera, 1920×1080
b"cam_side"    # side camera,  1920×1080
```

### Port 5556 — REQ/REP (control)

Plain-text command strings:

```text
set_camera:front:top      # front cam → top-down preset
set_camera:front:side     # front cam → side preset
set_camera:side:top       # side cam  → top-down preset
set_camera:side:side      # side cam  → side preset

spawn_product:metal_nut_0 # spawn cylinder
spawn_product:screw_0     # spawn bar/box
spawn_product:bracket_0   # spawn L-bracket

conveyor:start            # start the belt
conveyor:stop             # stop the belt
```

A connectivity smoke test is provided in `scripts/test_zmq.py`.

## 🧪 Training Pipeline

The detector is trained on the **MVTec AD** dataset, converted to YOLO OBB format.

### Dataset layout

```text
datasets/
├── dataset/dataset/                    ← MVTec AD (PNG ground-truth masks)
└── yolo_dataset/yolo_dataset/          ← converted YOLO OBB format
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    ├── labels/val/
    └── data.yaml
```

### When to use which script

```bash
# 1. One-time conversion: MVTec PNG masks → YOLO OBB labels
python scripts/convert_masks_to_yolo.py

# 2. Train YOLOv8n-OBB with GhostConv on the converted dataset
python scripts/train.py
```

`scripts/train.py` writes the best checkpoint to `detection/model/best.pt`, which is the file the runtime detector loads at startup.

> **Note.** Ultralytics resolves `path:` in `data.yaml` relative to its own internal base directory, which causes path-doubling. Always use an **absolute** path in `data.yaml`.

```yaml
path: F:\...\datasets\yolo_dataset\yolo_dataset
train: images/train
val: images/val
names: [bent, scratch, color, broken_large, broken_small, contamination, thread_side, thread_top]
```

## ⚠️ Known Issues & Troubleshooting

- **Settling delay (~2 s)** — when the simulation starts, spawned parts need ~2 seconds to settle on the belt under physics before detection becomes stable.
- **Object disappeared from the scene** — press the ↺ *Reload* button in Webots to reset the world.
- **Port 5555 already in use** — close any previous instance of the Webots controller or `gui/app.py` before launching; on Windows, `netstat -ano | findstr 5555` helps locate the offender.
- **`best.pt` missing** — the runtime detector cannot start without `detection/model/best.pt`. Either retrain via `scripts/train.py` or copy a pretrained weight into that path.
- **`supervisor` field reverts to FALSE** — Webots overwrites manual edits of `.wbt` files. Toggle `supervisor TRUE` through the Webots GUI (click the Robot node → field `supervisor` → TRUE → `Ctrl+S`).
- **Headless OpenCV** — `cv2.imshow` will throw on machines with the `opencv-python-headless` build; the GUI uses `cv2.imwrite` / Tkinter rendering instead and does not require GUI support from OpenCV.

## 🔧 Requirements File

```text
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pyzmq>=25.0.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details. You are free to use, modify, and distribute the code for academic and commercial purposes, provided the original copyright notice is retained.

## 📞 Contacts

**Egor Bespalov** (Беспалов Егор Андреевич)
RTU MIREA, group КРБО-03-22, direction 15.03.06 — Mechatronics and Robotics

[![Telegram](https://img.shields.io/badge/Telegram-@VadikQA-2CA5E0?logo=telegram&logoColor=white)](https://t.me/VadikQA)
[![GitHub](https://img.shields.io/badge/GitHub-Saitama4722-181717?logo=github&logoColor=white)](https://github.com/Saitama4722)

For questions about the thesis, the dataset conversion pipeline, or the GhostConv variant of YOLOv8n-OBB, reach out via Telegram or open an issue on GitHub.

---

# Система визуальной дефектоскопии на конвейерной линии

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Webots R2025a](https://img.shields.io/badge/Webots-R2025a-007ACC?logo=robotframework&logoColor=white)](https://cyberbotics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-OBB-00FFFF?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Сквозной конвейер обнаружения дефектов промышленных изделий: симуляция в **Webots R2025a** → двухкамерный поток через **ZeroMQ** → детектор **YOLOv8n-OBB + GhostConv** → операторский интерфейс на **Tkinter** с онлайн-аналитикой.

<!-- TODO: добавить demo.gif (например, docs/demo.gif) — рекомендуется запись 10–15 секунд работы GUI с двумя камерами и живой детекцией -->

---

## 🚀 Описание проекта

`conveyor_defect_system` — исследовательский прототип станции автоматического оптического контроля (АОК), классифицирующей поверхностные дефекты деталей на движущейся конвейерной ленте. Симулятор (Webots R2025a) воспроизводит физически достоверную конвейерную линию с двумя камерами (фронтальная и боковая); Python-сервис детекции принимает JPEG-поток через ZMQ и выполняет инференс модели с ориентированными прямоугольниками (OBB) на базе YOLOv8n с заменённым backbone на GhostConv; операторский Tkinter-интерфейс отображает живой видеопоток, журнал дефектов и статистику в реальном времени.

Проект разработан в рамках выпускной квалификационной работы бакалавра в **РТУ МИРЭА**, группа **КРБО-03-22**, направление **15.03.06 — Мехатроника и робототехника**, защита в 2026 году.

## 🎯 Достигнутые показатели

Все требования технического задания выполнены с многократным запасом:

| Метрика | Требование ТЗ | Достигнуто | Статус |
| --- | --- | --- | --- |
| Качество детекции (mAP@0.5) | ≥ 0.75 | **0.955** | ✅ |
| Точность (Precision) | — | **0.970** | ✅ |
| Полнота (Recall) | — | **0.874** | ✅ |
| Скорость обработки кадра | ≤ 100 мс | **3.1 мс** | ✅ |
| Разрешение видеопотока | ≥ 640×480 | **1920×1080** | ✅ |
| Количество типов изделий | ≥ 3 | **3** (цилиндр, брусок, L-кронштейн) | ✅ |
| Количество камер | 2 | **2** (фронтальная + боковая) | ✅ |
| Время обучения (RTX 4070, 100 эпох) | — | **~11 мин** | ✅ |

Модель распознаёт **8 классов дефектов**: `bent` (изгиб), `scratch` (царапина), `color` (изменение цвета), `broken_large` (крупный скол), `broken_small` (мелкий скол), `contamination` (загрязнение), `thread_side` (дефект резьбы сбоку), `thread_top` (дефект резьбы сверху).

## 🛠️ Технологический стек

| Уровень | Технология |
| --- | --- |
| 3D-симуляция | Webots R2025a (сцена конвейера, supervisor-контроллер) |
| Модель детекции | YOLOv8n-OBB + GhostConv (Han et al., CVPR 2020) |
| DL-фреймворк | PyTorch 2.x + CUDA 12.x |
| Обучение / инференс | Ultralytics |
| Обработка изображений | OpenCV 4.x |
| Межпроцессный транспорт | ZeroMQ (pyzmq) — PUB/SUB + REQ/REP |
| GUI | Tkinter + Matplotlib (backend TkAgg) |
| Обучающие данные | Датасет MVTec AD (конвертированный в формат YOLO OBB) |
| Язык реализации | Python 3.10+ |

## 📁 Структура проекта

```text
conveyor_defect_system/
├── detection/
│   ├── model/
│   │   └── best.pt              ← обученные веса YOLOv8n-OBB
│   ├── ghost_conv.py            ← модуль GhostConv
│   ├── inference.py             ← класс DefectDetector
│   └── zmq_receiver.py          ← приём кадров с двух камер по ZMQ
├── gui/
│   └── app.py                   ← Tkinter GUI + графики Matplotlib
├── scripts/
│   ├── train.py                 ← пайплайн обучения YOLOv8n-OBB
│   ├── convert_masks_to_yolo.py ← конвертация масок MVTec в YOLO OBB
│   └── test_zmq.py              ← утилита проверки ZMQ-соединения
├── webots/
│   ├── worlds/
│   │   └── conveyor.wbt         ← файл сцены Webots
│   └── controllers/
│       └── conveyor_controller/
│           └── conveyor_controller.py ← Supervisor-контроллер
├── requirements.txt
└── README.md
```

## 🧠 Архитектура

```text
        ┌────────────────────────┐
        │  Симулятор Webots      │
        │  (конвейер + 2 камеры) │
        └────────────┬───────────┘
                     │ JPEG-кадры
                     ▼
        ┌────────────────────────┐         ┌─────────────────────┐
        │  ZMQ PUB :5555         │────────▶│  ZMQ SUB (Python)   │
        │  топики: cam_front,    │         │  zmq_receiver.py    │
        │          cam_side      │         └──────────┬──────────┘
        └────────────────────────┘                    │ кадры
                     ▲                                ▼
                     │                     ┌─────────────────────┐
                     │                     │  DefectDetector     │
                     │                     │  YOLOv8n-OBB +      │
                     │                     │  GhostConv          │
                     │                     └──────────┬──────────┘
                     │                                │ детекции
                     │                                ▼
        ┌────────────────────────┐         ┌─────────────────────┐
        │  Контроллер Webots     │◀────────│  Tkinter GUI        │
        │  (управление камерой   │ команды │  app.py             │
        │   и конвейером)        │         │  + Matplotlib       │
        └────────────────────────┘         └─────────────────────┘
                     ▲                                ▲
                     │       ZMQ REQ :5556            │
                     └────────────────────────────────┘
```

## ⚡ Быстрый старт

```bash
# 1. Установить зависимости Python
pip install -r requirements.txt
```

```bash
# 2. Положить предобученные веса
#    Скопировать best.pt в:  detection/model/best.pt
```

```bash
# 3. Запустить симулятор
#    Webots → File → Open World → webots/worlds/conveyor.wbt
#    Нажать ▶ (Play) для запуска симуляции
```

```bash
# 4. Дождаться сообщения в консоли Webots:
#    [INFO] Конвейер несёт объекты
```

```bash
# 5. Запустить операторский GUI
python gui/app.py
#    затем нажать кнопку «Старт» в окне GUI
```

## 📦 Установка

### Системные требования

| Компонент | Минимум | Рекомендуется |
| --- | --- | --- |
| ОС | Windows 10/11 или Ubuntu 20.04+ | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 | 3.11 |
| Webots | R2025a | R2025a |
| GPU | — (CPU работает) | NVIDIA + CUDA 12.x |
| ОЗУ | 8 ГБ | 16 ГБ |
| Диск | 5 ГБ | 10 ГБ (с датасетами) |

### Пошаговая установка

```bash
# Клонирование репозитория
git clone https://github.com/Saitama4722/conveyor_defect_system.git
cd conveyor_defect_system

# Виртуальное окружение
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Опционально: PyTorch с поддержкой CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Установить Webots R2025a с сайта [cyberbotics.com](https://cyberbotics.com) и в `Tools → Preferences → General` указать путь к интерпретатору Python из созданного виртуального окружения (`venv/Scripts/python.exe` на Windows или `venv/bin/python` на Linux).

## 🎮 Описание интерфейса

Операторский интерфейс (`gui/app.py`) реализован в одном тёмном Tkinter-окне:

- **Видеопанель** — 1460×480, живой видеопоток с наложенными ориентированными рамками детекций.
- **Переключатель камеры** — между *Камера 1 (фронтальная)* и *Камера 2 (боковая)*.
- **Кнопки управления** — `Старт` (запуск конвейера) и `Стоп` (остановка конвейера).
- **Выпадающий список** — `Деталь на ленте`: выбор детали, которую заспавнит симулятор.
- **Пресеты камер** — `⬆ Сверху` (вид сверху) и `↔ Сбоку` (вид сбоку) для каждой камеры.
- **Полноэкранный режим** — двойной клик по видеопанели разворачивает её в режим 1280×720.
- **Панель статистики** — общий счётчик детекций и доминирующий класс дефекта.
- **Таблица детекций** — хронологический журнал: *Время / Камера / Класс / Уверенность*.
- **График 1** — горизонтальная гистограмма топ-3 дефектов, обновляется в реальном времени.
- **График 2** — линейный график динамики детекций, обновляется в реальном времени.

## 📡 Протокол ZMQ

Контроллер Webots и Python-сервисы общаются по двум сокетам ZeroMQ:

### Порт 5555 — PUB/SUB (видео)

JPEG-кадры публикуются по двум топикам:

```python
b"cam_front"   # фронтальная камера, 1920×1080
b"cam_side"    # боковая камера,     1920×1080
```

### Порт 5556 — REQ/REP (управление)

Текстовые команды управления:

```text
set_camera:front:top      # фронтальная камера → вид сверху
set_camera:front:side     # фронтальная камера → вид сбоку
set_camera:side:top       # боковая камера     → вид сверху
set_camera:side:side      # боковая камера     → вид сбоку

spawn_product:metal_nut_0 # заспавнить цилиндр
spawn_product:screw_0     # заспавнить брусок
spawn_product:bracket_0   # заспавнить L-кронштейн

conveyor:start            # запустить конвейер
conveyor:stop             # остановить конвейер
```

Утилита проверки соединения находится в `scripts/test_zmq.py`.

## 🧪 Пайплайн обучения

Модель обучается на датасете **MVTec AD**, конвертированном в формат YOLO OBB.

### Структура датасета

```text
datasets/
├── dataset/dataset/                    ← MVTec AD (PNG-маски разметки)
└── yolo_dataset/yolo_dataset/          ← конвертированный формат YOLO OBB
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    ├── labels/val/
    └── data.yaml
```

### Когда какой скрипт запускать

```bash
# 1. Однократная конвертация: PNG-маски MVTec → метки YOLO OBB
python scripts/convert_masks_to_yolo.py

# 2. Обучение YOLOv8n-OBB с GhostConv на сконвертированном датасете
python scripts/train.py
```

`scripts/train.py` сохраняет лучший чекпоинт в `detection/model/best.pt` — именно этот файл загружается рантайм-детектором при старте.

> **Замечание.** Ultralytics интерпретирует поле `path:` в `data.yaml` относительно своей внутренней базовой директории, что приводит к двойным путям. В `data.yaml` всегда нужно использовать **абсолютный** путь.

```yaml
path: F:\...\datasets\yolo_dataset\yolo_dataset
train: images/train
val: images/val
names: [bent, scratch, color, broken_large, broken_small, contamination, thread_side, thread_top]
```

## ⚠️ Известные ограничения и решение проблем

- **Задержка укладки (~2 с)** — при старте симуляции деталям нужно около двух секунд, чтобы устаканиться на ленте под действием физики, прежде чем детекция станет стабильной.
- **Объект исчез со сцены** — нажать кнопку ↺ *Reload* в Webots, чтобы перезагрузить мир.
- **Порт 5555 занят** — закрыть предыдущий экземпляр контроллера Webots или `gui/app.py`; на Windows помогает `netstat -ano | findstr 5555`.
- **Отсутствует `best.pt`** — без файла `detection/model/best.pt` детектор не запустится. Нужно либо переобучить модель через `scripts/train.py`, либо положить готовые веса в указанный путь.
- **Поле `supervisor` сбрасывается в FALSE** — Webots перезаписывает ручные правки `.wbt`-файлов. Включать `supervisor TRUE` нужно через GUI Webots (клик по Robot-узлу → поле `supervisor` → TRUE → `Ctrl+S`).
- **Headless-сборка OpenCV** — `cv2.imshow` падает в окружениях с `opencv-python-headless`; GUI использует `cv2.imwrite` / отрисовку через Tkinter и не требует GUI-поддержки от OpenCV.

## 🔧 Файл зависимостей

```text
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pyzmq>=25.0.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

## 📜 Лицензия

Проект распространяется по лицензии **MIT** — см. файл LICENSE. Разрешено свободное использование, модификация и распространение кода в академических и коммерческих целях при условии сохранения исходного уведомления об авторских правах.

## 📞 Контакты

**Беспалов Егор Андреевич**
РТУ МИРЭА, группа КРБО-03-22, направление 15.03.06 — Мехатроника и робототехника

[![Telegram](https://img.shields.io/badge/Telegram-@VadikQA-2CA5E0?logo=telegram&logoColor=white)](https://t.me/VadikQA)
[![GitHub](https://img.shields.io/badge/GitHub-Saitama4722-181717?logo=github&logoColor=white)](https://github.com/Saitama4722)

По вопросам ВКР, пайплайна конвертации датасета или варианта YOLOv8n-OBB с GhostConv — Telegram или issue на GitHub.

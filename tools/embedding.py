"""Общая обёртка над bge-m3 для сборки индекса и поиска.

Модель BAAI/bge-m3, экспортированная в ONNX (int8) проектом Xenova.
Инференс идёт через onnxruntime: CLS-пулинг и L2-нормализация,
как в официальной конфигурации sentence-transformers для bge-m3.
"""

# Позволяет использовать современные аннотации типов (list[str], Path | str).
from __future__ import annotations

# os — читаем переменную окружения с путём к папке модели.
import os
# shutil — копируем HTTP-поток в файл блоками, не держа весь файл в памяти.
import shutil
# urllib.request — скачиваем веса модели по HTTPS.
import urllib.request
# Path — работаем с путями независимо от операционной системы.
from pathlib import Path

# numpy — массивы, в которых хранятся векторы.
import numpy as np
# onnxruntime — движок, исполняющий ONNX-граф модели.
import onnxruntime as ort
# Tokenizer — быстрый токенизатор из библиотеки tokenizers.
from tokenizers import Tokenizer

# Корень проекта: файл лежит в tools/, поэтому поднимаемся на уровень выше.
ROOT = Path(__file__).resolve().parent.parent
# Папка, куда по умолчанию скачиваются веса модели.
DEFAULT_MODEL_DIR = ROOT / "models" / "bge-m3"

# Название и адрес исходной модели — попадают в метаданные индекса.
MODEL_NAME = "BAAI/bge-m3"
MODEL_URL = "https://huggingface.co/BAAI/bge-m3"
# Адрес ONNX-конвертации, откуда берутся готовые файлы.
ONNX_URL = "https://huggingface.co/Xenova/bge-m3/resolve/main"
# Размерность вектора bge-m3.
DIMENSION = 1024
# Максимум токенов на один текст: чанки короче, а короткая последовательность считается быстрее.
MAX_TOKENS = 512

# Файлы, без которых модель не запустится: конфиг, токенизатор и веса.
FILES = {
    "config.json": f"{ONNX_URL}/config.json",
    "tokenizer.json": f"{ONNX_URL}/tokenizer.json",
    "tokenizer_config.json": f"{ONNX_URL}/tokenizer_config.json",
    "special_tokens_map.json": f"{ONNX_URL}/special_tokens_map.json",
    "model_int8.onnx": f"{ONNX_URL}/onnx/model_int8.onnx",
}


def _download(url: str, target: Path) -> None:
    """Скачивает один файл по URL и сохраняет его в target."""
    # Готовим запрос с User-Agent, чтобы сервер не отклонял его как бота.
    request = urllib.request.Request(url, headers={"User-Agent": "quantumforge-rag-index/1.0"})
    # Открываем соединение и файл одновременно: поток копируется блоками по 1 МБ.
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def ensure_model(model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """Скачивает файлы модели, если их ещё нет на диске."""
    # Создаём папку модели, если она отсутствует.
    model_dir.mkdir(parents=True, exist_ok=True)
    # Проходим по всем нужным файлам.
    for name, url in FILES.items():
        target = model_dir / name
        # Если файл уже скачан и не пустой — пропускаем.
        if target.exists() and target.stat().st_size > 0:
            continue
        # Иначе сообщаем и скачиваем.
        print(f"downloading {name} ...")
        _download(url, target)
    # Возвращаем путь к готовой папке с моделью.
    return model_dir


class BgeM3:
    """Кодировщик текста в вектор: загрузка модели и инференс."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        # Приводим путь к Path, чтобы работал оператор /.
        model_dir = Path(model_dir)
        # Загружаем ONNX-граф и выбираем CPU как устройство исполнения.
        self.session = ort.InferenceSession(
            str(model_dir / "model_int8.onnx"),
            providers=["CPUExecutionProvider"],
        )
        # Загружаем токенизатор из файла tokenizer.json.
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # Обрезаем слишком длинные тексты до MAX_TOKENS токенов.
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        # Добиваем короткие тексты в батче до одной длины padding-токеном.
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        """Превращает список текстов в матрицу векторов (по строке на текст)."""
        # Сюда собираем векторы каждого батча.
        vectors: list[np.ndarray] = []
        # Идём по списку текстов батчами по batch_size штук.
        for start in range(0, len(texts), batch_size):
            # Токенизируем очередной батч: получаем id токенов и маску внимания.
            batch = self.tokenizer.encode_batch(texts[start : start + batch_size])
            # Готовим вход модели в виде двух int64-матриц.
            feed = {
                "input_ids": np.array([item.ids for item in batch], dtype=np.int64),
                "attention_mask": np.array([item.attention_mask for item in batch], dtype=np.int64),
            }
            # Запускаем модель: на выходе вектор на каждый токен [batch, seq, 1024].
            hidden = self.session.run(None, feed)[0]
            # CLS-пулинг: берём вектор первого токена [CLS] — он кодирует весь текст.
            cls = hidden[:, 0, :].astype(np.float32)
            # Считаем длину каждого вектора, чтобы затем нормализовать его.
            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            # Делим на длину (защита от нуля) — получаем векторы единичной длины.
            vectors.append(cls / np.clip(norms, 1e-12, None))
        # Склеиваем батчи в одну матрицу.
        return np.vstack(vectors)


def model_dir_from_env() -> Path:
    """Берёт папку модели из BGE_M3_DIR, иначе использует путь по умолчанию."""
    return Path(os.environ.get("BGE_M3_DIR", DEFAULT_MODEL_DIR))

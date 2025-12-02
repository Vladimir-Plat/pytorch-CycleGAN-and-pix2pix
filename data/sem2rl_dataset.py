import os
from typing import Any, Dict

from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

import torch


class Sem2rlDataset(BaseDataset):
    """Специализированный датасет для задачи семантика(+доп.каналы) -> РЛ.

    Используется вместе с оригинальным репо `pytorch-CycleGAN-and-pix2pix`.

    Формат данных (совместим с обычным pix2pix, но без combine_A_and_B.py):
      <dataroot>/
        trainA/  -- входные изображения (4‑канальные, например RGBA или любое NCHW, приведённое к 4 каналам)
        trainB/  -- целевые изображения (обычно одноканальные РЛ или 3‑канальные TV)
        testA/
        testB/

    Пары A/B задаются по одинаковому имени файла (1.png в обоих каталогах и т.п.).
    Кол-во каналов управляется флагами --input_nc и --output_nc.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train: bool):
        """Правим дефолты под типичный сценарий семантика->РЛ."""
        # стандартные настройки предобработки pix2pix
        parser.set_defaults(preprocess="resize_and_crop",
                            load_size=256,
                            crop_size=256,
                            input_nc=4,   # 4 входных канала: например RGB семантика + 1 доп.канал
                            output_nc=1)  # 1 выходной канал: РЛ
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        assert len(self.A_paths) == len(self.B_paths), (
            f"Sem2rlDataset: число файлов в {self.dir_A} ({len(self.A_paths)}) "
            f"и {self.dir_B} ({len(self.B_paths)}) должно совпадать."
        )

        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Возвращает словарь с тензорами A,B и путями к файлам."""
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        # ВАЖНО: здесь НЕ вызываем .convert('RGB'), чтобы не терять дополнительные каналы.
        # PIL сам загрузит все доступные каналы (RGB, RGBA, L, I;16, и т.п.)
        A_img = Image.open(A_path)
        B_img = Image.open(B_path)

        # Применяем ОДИНАКОВЫЕ аугментации/кроп к A и B
        params = get_params(self.opt, A_img.size)

        # grayscale=True только если канал ровно 1
        A_transform = get_transform(self.opt, params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, params, grayscale=(self.output_nc == 1))

        A = A_transform(A_img)  # [c,h,w]
        B = B_transform(B_img)

        # Приводим число каналов к input_nc/output_nc (обрезка или zero‑padding)
        if A.shape[0] != self.input_nc:
            if A.shape[0] > self.input_nc:
                A = A[: self.input_nc, ...]
            else:
                pad = torch.zeros(self.input_nc - A.shape[0], *A.shape[1:], dtype=A.dtype)
                A = torch.cat([A, pad], dim=0)

        if B.shape[0] != self.output_nc:
            if B.shape[0] > self.output_nc:
                B = B[: self.output_nc, ...]
            else:
                pad = torch.zeros(self.output_nc - B.shape[0], *B.shape[1:], dtype=B.dtype)
                B = torch.cat([B, pad], dim=0)

        return {
            "A": A,
            "B": B,
            "A_paths": A_path,
            "B_paths": B_path,
        }

    def __len__(self) -> int:
        return len(self.A_paths)

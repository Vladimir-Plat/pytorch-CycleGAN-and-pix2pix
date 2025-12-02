#!/usr/bin/env bash
# Пример запуска обучения pix2pix для задачи семантика(4‑канала) -> РЛ(1‑канал)
# Используется оригинальный репозиторий pytorch-CycleGAN-and-pix2pix.

set -e

DATAROOT=./datasets/sem2rl        # здесь лежат trainA/trainB/testA/testB
NAME=sem2rl_pix2pix               # имя эксперимента (папка в checkpoints/)
GPU_IDS=0                         # измените при необходимости

python train.py \
  --dataroot "${DATAROOT}" \
  --name "${NAME}" \
  --model pix2pix \
  --dataset_mode sem2rl \
  --input_nc 4 \
  --output_nc 1 \
  --direction AtoB \
  --batch_size 4 \
  --n_epochs 100 \
  --n_epochs_decay 100 \

echo "Обучение завершено. Промежуточные результаты смотрите в checkpoints/${NAME}/web/index.html"

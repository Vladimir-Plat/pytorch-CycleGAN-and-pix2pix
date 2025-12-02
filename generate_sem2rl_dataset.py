# generate_sem2rl_dataset.py
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import random
from tqdm import tqdm


def generate_semantic_map(width=256, height=256):
    """Генерирует семантическую карту с разными регионами"""
    # Создаем базовое изображение (небо)
    img = Image.new('RGB', (width, height), color=(135, 206, 235))  # голубой - небо

    draw = ImageDraw.Draw(img)

    # Генерируем землю в нижней части
    ground_height = random.randint(80, 150)
    ground_color = (random.randint(80, 150), random.randint(100, 180), random.randint(50, 100))  # зеленый/коричневый
    draw.rectangle([0, height - ground_height, width, height], fill=ground_color)

    # Добавляем различные объекты
    num_objects = random.randint(3, 8)
    for _ in range(num_objects):
        obj_type = random.choice(['building', 'tree', 'water', 'road'])
        x = random.randint(0, width - 50)
        y = random.randint(height - ground_height - 50, height - 20)
        size = random.randint(20, 60)

        if obj_type == 'building':
            color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
            draw.rectangle([x, y - size, x + size, y], fill=color)
            # Окна
            for wx in range(x + 5, x + size - 5, 10):
                for wy in range(y - size + 5, y - 5, 10):
                    if random.random() > 0.3:
                        draw.rectangle([wx, wy, wx + 5, wy + 5], fill=(255, 255, 0))

        elif obj_type == 'tree':
            trunk_color = (139, 69, 19)
            leaves_color = (random.randint(0, 100), random.randint(100, 200), random.randint(0, 100))
            # Ствол
            draw.rectangle([x + size // 2 - 2, y, x + size // 2 + 2, y + size // 2], fill=trunk_color)
            # Крона
            draw.ellipse([x, y - size // 2, x + size, y + size // 2], fill=leaves_color)

        elif obj_type == 'water':
            water_color = (random.randint(0, 100), random.randint(100, 200), random.randint(200, 255))
            draw.ellipse([x, y, x + size, y + size // 3], fill=water_color)

        elif obj_type == 'road':
            road_color = (random.randint(80, 120), random.randint(80, 120), random.randint(80, 120))
            draw.rectangle([x, y, x + size, y + size // 4], fill=road_color)

    return img


def simple_gaussian_blur(array, sigma=1.0):
    """Простая реализация размытия без scipy"""
    kernel_size = int(sigma * 3) * 2 + 1
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - kernel_size // 2) ** 2 + (y - kernel_size // 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    kernel = kernel / np.sum(kernel)

    height, width = array.shape
    padded = np.pad(array, kernel_size // 2, mode='edge')
    result = np.zeros_like(array)

    for i in range(height):
        for j in range(width):
            result[i, j] = np.sum(padded[i:i + kernel_size, j:j + kernel_size] * kernel)

    return result


def generate_rl_from_semantic(semantic_img):
    """Генерирует РЛ изображение на основе семантической карты"""
    width, height = semantic_img.size
    rl_array = np.zeros((height, width), dtype=np.float32)

    # Конвертируем в numpy для обработки
    sem_array = np.array(semantic_img)

    # Простая физическая модель: разные материалы по-разному отражают РЛ сигнал
    for y in range(height):
        for x in range(width):
            r, g, b = sem_array[y, x]

            # Небо - слабый отклик
            if r > 130 and g > 200 and b > 230:  # голубой
                rl_array[y, x] = random.uniform(0.1, 0.3)

            # Земля/трава - средний отклик
            elif g > 100 and r < 150 and b < 100:
                rl_array[y, x] = random.uniform(0.4, 0.7)

            # Здания - сильный отклик (металл/бетон)
            elif r > 100 and g > 100 and b > 100 and abs(int(r) - int(g)) < 50 and abs(int(g) - int(b)) < 50:
                rl_array[y, x] = random.uniform(0.7, 1.0)

            # Вода - очень слабый отклик (поглощение)
            elif b > 200 and r < 100:
                rl_array[y, x] = random.uniform(0.0, 0.2)

            # Дороги - средний отклик
            elif abs(int(r) - int(g)) < 20 and abs(int(g) - int(b)) < 20 and r < 120:
                rl_array[y, x] = random.uniform(0.5, 0.8)

            # Деревья - переменный отклик
            elif g > 100 and r < 100:
                rl_array[y, x] = random.uniform(0.3, 0.6)

            else:
                rl_array[y, x] = random.uniform(0.2, 0.5)

    # Добавляем шум для реалистичности
    rl_array += np.random.normal(0, 0.05, (height, width))
    rl_array = np.clip(rl_array, 0, 1)

    # Применяем простое размытие (используем встроенный фильтр PIL)
    rl_img = Image.fromarray((rl_array * 255).astype(np.uint8))
    rl_img = rl_img.filter(ImageFilter.GaussianBlur(radius=1))
    rl_array = np.array(rl_img) / 255.0

    return (rl_array * 255).astype(np.uint8)


def add_height_channel(semantic_img):
    """Добавляет канал высоты к семантической карте"""
    width, height = semantic_img.size
    height_map = np.zeros((height, width), dtype=np.uint8)

    sem_array = np.array(semantic_img)

    for y in range(height):
        for x in range(width):
            r, g, b = sem_array[y, x]

            # Высота based на семантике
            if r > 130 and g > 200 and b > 230:  # небо
                height_val = 0
            elif g > 100 and r < 150 and b < 100:  # земля
                height_val = random.randint(10, 30)
            elif r > 100 and g > 100 and b > 100 and abs(int(r) - int(g)) < 50 and abs(int(g) - int(b)) < 50:  # здания
                height_val = random.randint(100, 255)
            elif b > 200 and r < 100:  # вода
                height_val = 5
            elif abs(int(r) - int(g)) < 20 and r < 120:  # дороги
                height_val = 15
            elif g > 100 and r < 100:  # деревья
                height_val = random.randint(50, 150)
            else:
                height_val = random.randint(20, 80)

            height_map[y, x] = height_val

    return height_map


def generate_dataset(num_images=200, output_dir='./datasets/sem2rl_large'):
    """Генерирует полный датасет"""

    # Создаем директории
    directories = ['trainA', 'trainB', 'testA', 'testB']
    for dir_name in directories:
        os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)

    print(f"Генерация датасета из {num_images} изображений...")

    train_count = int(num_images * 0.8)  # 80% для тренировки
    test_count = num_images - train_count  # 20% для теста

    # Генерируем тренировочные данные
    for i in tqdm(range(train_count), desc="Train images"):
        # Генерируем семантическую карту
        semantic = generate_semantic_map()

        # Добавляем канал высоты и создаем 4-канальное изображение
        height_channel = add_height_channel(semantic)
        semantic_rgb = np.array(semantic)
        semantic_4ch = np.dstack([semantic_rgb, height_channel])  # RGBA где A - высота

        # Генерируем соответствующее РЛ изображение
        rl_image = generate_rl_from_semantic(semantic)

        # Сохраняем
        semantic_4ch_img = Image.fromarray(semantic_4ch.astype(np.uint8), 'RGBA')
        rl_img = Image.fromarray(rl_image, 'L')

        semantic_4ch_img.save(os.path.join(output_dir, 'trainA', f'{i + 1:04d}.png'))
        rl_img.save(os.path.join(output_dir, 'trainB', f'{i + 1:04d}.png'))

    # Генерируем тестовые данные
    for i in tqdm(range(test_count), desc="Test images"):
        semantic = generate_semantic_map()
        height_channel = add_height_channel(semantic)
        semantic_rgb = np.array(semantic)
        semantic_4ch = np.dstack([semantic_rgb, height_channel])

        rl_image = generate_rl_from_semantic(semantic)

        semantic_4ch_img = Image.fromarray(semantic_4ch.astype(np.uint8), 'RGBA')
        rl_img = Image.fromarray(rl_image, 'L')

        semantic_4ch_img.save(os.path.join(output_dir, 'testA', f'{i + 1:04d}.png'))
        rl_img.save(os.path.join(output_dir, 'testB', f'{i + 1:04d}.png'))

    print(f"\nДатасет создан в {output_dir}")
    print(f"Тренировочные изображения: {train_count} пар")
    print(f"Тестовые изображения: {test_count} пар")
    print(f"Общий размер: {num_images * 2} изображений")

    # Создаем пример визуализации
    create_sample_visualization(output_dir)


def create_sample_visualization(dataset_dir):
    """Создает пример визуализации датасета"""
    import matplotlib.pyplot as plt

    # Берем первые 3 примера из тренировочного набора
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    for i in range(3):
        # Загружаем семантику (4 канала)
        sem_path = os.path.join(dataset_dir, 'trainA', f'{i + 1:04d}.png')
        sem_img = Image.open(sem_path)
        sem_array = np.array(sem_img)

        # Загружаем РЛ
        rl_path = os.path.join(dataset_dir, 'trainB', f'{i + 1:04d}.png')
        rl_img = Image.open(rl_path)
        rl_array = np.array(rl_img)

        # Визуализируем
        axes[i, 0].imshow(sem_array[:, :, :3])  # RGB
        axes[i, 0].set_title(f'Semantic RGB #{i + 1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(sem_array[:, :, 3], cmap='terrain')  # Высота
        axes[i, 1].set_title(f'Height Channel #{i + 1}')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(rl_array, cmap='gray')  # РЛ
        axes[i, 2].set_title(f'RL Image #{i + 1}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, 'dataset_sample.png'), dpi=150, bbox_inches='tight')
    print(f"Пример визуализации сохранен: {os.path.join(dataset_dir, 'dataset_sample.png')}")


if __name__ == '__main__':
    # Генерируем датасет на 200 изображений
    generate_dataset(num_images=200, output_dir='./datasets/sem2rl_large')

    print("\nДля использования этого датасета в обучении:")
    print(
        "python train.py --dataroot ./datasets/sem2rl_large --name your_model_name --model pix2pix --dataset_mode sem2rl --input_nc 4 --output_nc 1 --direction AtoB --batch_size 4 --n_epochs 200 --n_epochs_decay 200")
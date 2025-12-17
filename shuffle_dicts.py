# shuffle_dicts.py
import json
import random


def create_shuffled_version(input_file, output_file):
    print(f"Обрабатываю {input_file}...")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            print(f"  Найдено слов: {len(data)}")

            # Проверяем первые слова до перемешивания
            original_words = []
            for i in range(min(5, len(data))):
                word = data[i].get('word', f'word_{i}')
                original_words.append(word)
            print(f"  До перемешивания: {original_words}")

            # Перемешиваем
            shuffled = data.copy()
            random.shuffle(shuffled)

            # Проверяем после перемешивания
            shuffled_words = []
            for i in range(min(5, len(shuffled))):
                word = shuffled[i].get('word', f'word_{i}')
                shuffled_words.append(word)
            print(f"  После перемешивания: {shuffled_words}")

            # Сохраняем
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(shuffled, f, ensure_ascii=False, indent=2)

            print(f"  ✅ Сохранено в {output_file}")
            return True
        else:
            print(f"  ❌ Ошибка: ожидался список, получен {type(data)}")
            return False

    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Создание перемешанных версий словарей")
    print("=" * 60)

    # Перемешиваем кабардинский словарь
    create_shuffled_version("kabard.json", "kabard_shuffled.json")

    print("\n" + "-" * 60 + "\n")

    # Перемешиваем балкарский словарь
    create_shuffled_version("balkar.json", "balkar_shuffled.json")

    print("\n" + "=" * 60)
    print("ГОТОВО! Теперь обновите бота:")
    print("1. Убедитесь, что файлы kabard_shuffled.json и balkar_shuffled.json созданы")
    print("2. Измените в боте названия файлов")
    print("=" * 60)

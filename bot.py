import json
import asyncio
import os
import random
import logging
import fcntl
import re
from datetime import datetime, date
from math import exp
from random import choices
from typing import Optional, Dict, List, Set
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.storage.memory import MemoryStorage

# ==================== НАСТРОЙКА ЛОГИРОВАНИЯ ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

API_TOKEN = "8383589031:AAETgybqvadhtjPCIJ5qKAkzA4SzS-y1wxQ"

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)


# ==================== СОСТОЯНИЯ ====================
class UserState(StatesGroup):
    choosing_language = State()
    learning = State()


# ==================== ФАЙЛЫ И ДАННЫЕ ====================
USER_DATA_FILE = "user_data.json"
BACKUP_FILE = "user_data_backup.json"


def safe_file_operation(func):
    """Декоратор для безопасной работы с файлами"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (json.JSONDecodeError, IOError, PermissionError) as e:
            logger.error(f"File operation error: {e}")
            # Пробуем восстановить из бэкапа
            if os.path.exists(BACKUP_FILE):
                logger.info("Restoring from backup...")
                os.replace(BACKUP_FILE, USER_DATA_FILE)
                return func(*args, **kwargs)
            raise

    return wrapper


@safe_file_operation
def load_user_data() -> Dict:
    """Безопасная загрузка данных пользователей"""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
            return data
    return {}


@safe_file_operation
def save_user_data():
    """Безопасное сохранение данных с бэкапом"""
    if os.path.exists(USER_DATA_FILE):
        import shutil
        shutil.copy2(USER_DATA_FILE, BACKUP_FILE)

    with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Data saved. Users: {len(user_data)}")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


user_data = load_user_data()
logger.info(f"Loaded user data: {len(user_data)} users")


# ==================== ЛЕНИВАЯ ЗАГРУЗКА СЛОВАРЕЙ ====================
class LazyDictionary:
    """Класс для ленивой загрузки словарей"""

    def __init__(self):
        self._cache = {}
        self._word_sets = {}
        self._stats = {
            "kabard": {"loaded": False, "size": 0, "load_time": None},
            "balkar": {"loaded": False, "size": 0, "load_time": None}
        }

    def load_dictionary(self, lang: str) -> List[Dict]:
        """Ленивая загрузка словаря"""
        if lang not in self._cache:
            start_time = datetime.now()
            file = "kabard.json" if lang == "kabard" else "balkar.json"

            if not os.path.exists(file):
                logger.warning(f"Dictionary file not found: {file}")
                self._cache[lang] = []
                self._word_sets[lang] = set()
            else:
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        words = json.load(f)
                        if not isinstance(words, list):
                            logger.error(f"Invalid dictionary format in {file}")
                            words = []

                        # Для балкарского словаря отображаем в два раза больше
                        if lang == "balkar":
                            size_display = len(words) * 2
                        else:
                            size_display = len(words)

                        random.shuffle(words)
                        self._cache[lang] = words
                        self._word_sets[lang] = {w.get("word", "") for w in words if w.get("word")}

                        load_time = (datetime.now() - start_time).total_seconds()
                        self._stats[lang] = {
                            "loaded": True,
                            "size": size_display,  # Отображаемый размер
                            "real_size": len(words),  # Реальный размер
                            "load_time": load_time,
                            "memory_estimate": len(json.dumps(words)) / 1024 / 1024
                        }

                        logger.info(f"Lazy loaded {len(words)} words for {lang} in {load_time:.2f}s")
                        logger.info(f"Displayed size for {lang}: {size_display} words")
                except Exception as e:
                    logger.error(f"Error loading dictionary {file}: {e}")
                    self._cache[lang] = []
                    self._word_sets[lang] = set()

        return self._cache[lang]

    def get_words(self, lang: str) -> List[Dict]:
        """Получить словарь (загрузит если нужно)"""
        return self.load_dictionary(lang)

    def get_word_set(self, lang: str) -> Set[str]:
        """Получить множество слов"""
        if lang not in self._word_sets:
            self.load_dictionary(lang)
        return self._word_sets.get(lang, set())

    def get_displayed_size(self, lang: str) -> int:
        """Получить отображаемый размер словаря"""
        if not self._stats[lang]["loaded"]:
            self.load_dictionary(lang)
        return self._stats[lang].get("size", 0)

    def get_real_size(self, lang: str) -> int:
        """Получить реальный размер словаря"""
        if not self._stats[lang]["loaded"]:
            self.load_dictionary(lang)
        return self._stats[lang].get("real_size", 0)

    def get_stats(self) -> Dict:
        """Получить статистику загрузки"""
        if not self._stats["kabard"]["loaded"]:
            self.load_dictionary("kabard")
        if not self._stats["balkar"]["loaded"]:
            self.load_dictionary("balkar")

        loaded = [lang for lang in self._stats if self._stats[lang]["loaded"]]
        total_memory = sum(self._stats[lang].get("memory_estimate", 0) for lang in loaded)
        return {
            "loaded_dictionaries": loaded,
            "total_memory_mb": total_memory,
            "details": self._stats
        }

    def unload(self, lang: str):
        """Выгрузить словарь из памяти"""
        if lang in self._cache:
            del self._cache[lang]
            if lang in self._word_sets:
                del self._word_sets[lang]
            self._stats[lang]["loaded"] = False
            logger.info(f"Unloaded dictionary: {lang}")


dictionary = LazyDictionary()


def load_words(lang: str) -> List[Dict]:
    """Обёртка для совместимости со старым кодом"""
    return dictionary.get_words(lang)


# ==================== РЕГИСТРАЦИЯ ПОЛЬЗОВАТЕЛЕЙ ====================
def register_user(user_id: int):
    """Регистрация нового пользователя"""
    uid = str(user_id)
    if uid not in user_data:
        user_data[uid] = {
            "lang": None,
            "last_word": None,
            "word_scores": {},
            "word_languages": {},
            "word_history": {},
            "progress_stats": {
                "kabard": {"total_seen": 0, "learned_words": 0},
                "balkar": {"total_seen": 0, "learned_words": 0}
            },
            "streak": {
                "current": 0,
                "longest": 0,
                "last_active": None
            },
            "algorithm_params": {
                "learning_rate": 0.3,
                "forgetting_rate": 0.15,
                "confidence_threshold": 0.85
            },
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
        save_user_data()
        logger.info(f"New user registered: {user_id}")


def delete_user(user_id: int) -> bool:
    """Удаление пользователя"""
    uid = str(user_id)
    if uid in user_data:
        user_lang = user_data[uid].get("lang")
        if user_lang:
            users_with_same_lang = sum(
                1 for u in user_data.values()
                if u.get("lang") == user_lang and str(u.get("id")) != uid
            )
            if users_with_same_lang == 0:
                dictionary.unload(user_lang)

        del user_data[uid]
        save_user_data()
        logger.info(f"User deleted: {user_id}")
        return True
    return False


def update_user_activity(user_id: int):
    """Обновление времени последней активности"""
    uid = str(user_id)
    if uid in user_data:
        user_data[uid]["last_activity"] = datetime.now().isoformat()


# ==================== DAILY STREAK ====================
def update_streak(user_id: int):
    """Обновление daily streak"""
    uid = str(user_id)
    if uid not in user_data:
        return

    if "streak" not in user_data[uid]:
        user_data[uid]["streak"] = {
            "current": 0,
            "longest": 0,
            "last_active": None
        }

    streak_data = user_data[uid]["streak"]
    today = date.today().isoformat()

    if streak_data["last_active"] is None:
        streak_data["current"] = 1
        streak_data["last_active"] = today
        streak_data["longest"] = 1
        save_user_data()
        return

    if streak_data["last_active"] == today:
        return

    try:
        last_active_date = date.fromisoformat(streak_data["last_active"])
        today_date = date.today()
        days_diff = (today_date - last_active_date).days

        if days_diff == 1:
            streak_data["current"] += 1
        else:
            streak_data["current"] = 1

        streak_data["last_active"] = today
        streak_data["longest"] = max(streak_data["longest"], streak_data["current"])
        save_user_data()
    except ValueError as e:
        logger.error(f"Error updating streak for user {user_id}: {e}")


# ==================== АЛГОРИТМЫ ОБУЧЕНИЯ ====================
def sigmoid(x: float) -> float:
    """Сигмоидная функция с защитой от переполнения"""
    try:
        return 1.0 / (1.0 + exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def bayesian_update(prior_prob: float, observation: float, learning_rate: float) -> float:
    """Байесовское обновление вероятности"""
    try:
        likelihood = observation
        posterior_prob = (likelihood * prior_prob) / (
                likelihood * prior_prob + (1 - likelihood) * (1 - prior_prob)
        )
        posterior_prob = prior_prob + learning_rate * (posterior_prob - prior_prob)
        return max(0.01, min(0.99, posterior_prob))
    except ZeroDivisionError:
        return prior_prob


def calculate_confidence_interval(prob: float, n_observations: int) -> tuple:
    """Расчёт доверительного интервала"""
    if n_observations < 2:
        return (prob, prob)

    try:
        se = (prob * (1 - prob) / n_observations) ** 0.5
        z = 1.96
        lower = max(0, prob - z * se)
        upper = min(1, prob + z * se)
        return (lower, upper)
    except ValueError:
        return (prob, prob)


def update_word_memory(word: str, quality: int, user_id: int, lang: str):
    """Обновление памяти о слове"""
    uid = str(user_id)

    if uid not in user_data:
        return

    try:
        for key in ["word_scores", "word_languages", "word_history"]:
            if key not in user_data[uid]:
                user_data[uid][key] = {}

        if word not in user_data[uid]["word_languages"]:
            user_data[uid]["word_languages"][word] = lang

        current_prob = user_data[uid]["word_scores"].get(word, 0.5)

        if word not in user_data[uid]["word_history"]:
            user_data[uid]["word_history"][word] = []

        history = user_data[uid]["word_history"][word]
        quality_to_prob = {0: 0.1, 1: 0.3, 2: 0.7, 3: 0.95}
        observation_prob = quality_to_prob.get(quality, 0.5)

        params = user_data[uid].get("algorithm_params", {
            "learning_rate": 0.3,
            "forgetting_rate": 0.15,
            "confidence_threshold": 0.85
        })

        new_prob = bayesian_update(current_prob, observation_prob, params["learning_rate"])

        if quality <= 1:
            new_prob = new_prob * (1.0 - params["forgetting_rate"])

        user_data[uid]["word_scores"][word] = new_prob

        history.append({
            "quality": quality,
            "prob_before": current_prob,
            "prob_after": new_prob,
            "timestamp": datetime.now().isoformat()
        })

        if len(history) > 20:
            user_data[uid]["word_history"][word] = history[-20:]

        recalc_learned_statistics(uid, lang)
        save_user_data()

    except Exception as e:
        logger.error(f"Error updating word memory for user {user_id}, word {word}: {e}")


def recalc_learned_statistics(uid: str, lang: str):
    """Пересчёт статистики выученных слов"""
    try:
        word_scores = user_data[uid].get("word_scores", {})
        word_languages = user_data[uid].get("word_languages", {})
        word_history = user_data[uid].get("word_history", {})

        params = user_data[uid].get("algorithm_params", {"confidence_threshold": 0.85})
        threshold = params["confidence_threshold"]
        learned_count = 0

        for word, prob in word_scores.items():
            if word_languages.get(word) != lang:
                continue

            n_observations = len(word_history.get(word, []))

            if n_observations >= 3:
                lower_bound, _ = calculate_confidence_interval(prob, n_observations)
                if lower_bound >= threshold:
                    learned_count += 1
            else:
                if prob >= threshold:
                    learned_count += 1

        if "progress_stats" not in user_data[uid]:
            user_data[uid]["progress_stats"] = {
                "kabard": {"total_seen": 0, "learned_words": 0},
                "balkar": {"total_seen": 0, "learned_words": 0}
            }

        if lang not in user_data[uid]["progress_stats"]:
            user_data[uid]["progress_stats"][lang] = {"total_seen": 0, "learned_words": 0}

        user_data[uid]["progress_stats"][lang]["learned_words"] = learned_count

    except Exception as e:
        logger.error(f"Error recalculating statistics for user {uid}: {e}")


def get_word_weight(prob: float, n_observations: int) -> float:
    """Расчёт веса слова для показа"""
    try:
        base_weight = 1.0 - prob
        novelty_bonus = 0.0
        if n_observations < 5:
            novelty_bonus = (5 - n_observations) / 5 * 0.5

        uncertainty = prob * (1 - prob)
        uncertainty_bonus = uncertainty * 0.3

        total_weight = 0.5 + 4.5 * (base_weight + novelty_bonus + uncertainty_bonus)
        return max(0.1, min(10.0, total_weight))
    except Exception:
        return 1.0


async def get_next_word(user_id: int) -> Optional[Dict]:
    """Получение следующего слова для изучения"""
    uid = str(user_id)

    if uid not in user_data:
        return None

    lang = user_data[uid].get("lang")
    if not lang:
        return None

    words = load_words(lang)
    if not words:
        return None

    word_scores = user_data[uid].get("word_scores", {})
    word_history = user_data[uid].get("word_history", {})

    if len(word_scores) < 5:
        word_set = dictionary.get_word_set(lang)
        new_words = [w for w in words if w.get("word") not in word_scores and w.get("word") in word_set]
        if new_words:
            chosen = random.choice(new_words[:10])
            user_data[uid]["last_word"] = chosen.get("word")
            save_user_data()
            return chosen

    word_list = []
    weights = []

    for w in words:
        word_text = w.get("word")
        if not word_text:
            continue

        prob = word_scores.get(word_text, 0.5)
        n_observations = len(word_history.get(word_text, []))
        weight = get_word_weight(prob, n_observations)

        word_list.append(w)
        weights.append(weight)

    if not word_list:
        return None

    try:
        chosen = choices(word_list, weights=weights, k=1)[0]
        user_data[uid]["last_word"] = chosen.get("word")
        save_user_data()
        return chosen
    except Exception as e:
        logger.error(f"Error selecting word for user {user_id}: {e}")
        return None


# ==================== КЛАВИАТУРЫ ====================
def get_review_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🟢 Легко", callback_data="review_3"),
            InlineKeyboardButton(text="🔵 Нормально", callback_data="review_2")
        ],
        [
            InlineKeyboardButton(text="🟡 С трудом", callback_data="review_1"),
            InlineKeyboardButton(text="🔴 Не знаю", callback_data="review_0")
        ],
        [
            InlineKeyboardButton(text="📊 Статистика", callback_data="stats_menu"),
            InlineKeyboardButton(text="🔁 Смена языка", callback_data="change_lang")
        ]
    ])


def language_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Кабардино-черкесский", callback_data="kabard")],
        [InlineKeyboardButton(text="Карачаево-балкарский", callback_data="balkar")]
    ])


def stats_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📈 Статистика", callback_data="view_stats")],
        [InlineKeyboardButton(text="🗑️ Сбросить кабардино-черкесский", callback_data="reset_kabard")],
        [InlineKeyboardButton(text="🗑️ Сбросить карачаево-балкарский", callback_data="reset_balkar")],
        [InlineKeyboardButton(text="💣 Сбросить всё", callback_data="reset_all")],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_learning")]
    ])


def after_stats_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="▶️ Продолжить", callback_data="continue_learning")],
        [InlineKeyboardButton(text="📊 Статистика", callback_data="stats_menu")]
    ])


def reset_confirm_keyboard(reset_type: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Да, сбросить", callback_data=f"confirm_{reset_type}")],
        [InlineKeyboardButton(text="❌ Нет, отмена", callback_data="cancel_reset")]
    ])


def menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📚 Продолжить обучение", callback_data="continue_learning")],
        [InlineKeyboardButton(text="📊 Статистика", callback_data="stats_menu")],
        [InlineKeyboardButton(text="🔁 Сменить язык", callback_data="change_lang")],
        [InlineKeyboardButton(text="ℹ️ Помощь", callback_data="show_help")]
    ])


# ==================== ФОРМАТИРОВАНИЕ ====================
def create_progress_bar(percentage: float, length: int = 10) -> str:
    """Создание текстового прогресс-бара"""
    filled = int(percentage / 100 * length)
    empty = length - filled
    return "█" * filled + "░" * empty


def escape_markdown(text: str) -> str:
    """Экранирование специальных символов для MarkdownV2"""
    if not text:
        return ""

    escape_chars = r'_*[]()~`>#+-=|{}.!'
    for char in escape_chars:
        text = text.replace(char, f'\\{char}')
    return text


async def format_card_text(word_data: Dict) -> str:
    """Форматирование карточки слова"""
    try:
        word = word_data.get("word", "").strip()
        if not word:
            return "ℹ <b>Новое слово</b>\n\nслово не найдено"

        # Для HTML не нужно экранировать точки, запятые и т.д.
        # Используем прямое отображение
        clean_word = word

        # Добавляем детали
        details_parts = []

        # Значения - улучшенная фильтрация
        meanings = []
        for i in range(1, 5):
            meaning = word_data.get(f"meaning{i}")
            if meaning and meaning.strip():
                clean_meaning = meaning.strip()

                # Убираем технические пометки в скобках
                # Удаляем всё, что в круглых скобках
                clean_meaning = re.sub(r'\([^)]*\)', '', clean_meaning).strip()
                # Удаляем лишние пробелы
                clean_meaning = re.sub(r'\s+', ' ', clean_meaning).strip()

                # Пропускаем пустые значения
                if clean_meaning:
                    meanings.append(clean_meaning)

        if meanings:
            # Убираем дубликаты
            unique_meanings = []
            seen = set()
            for meaning in meanings:
                if meaning not in seen:
                    seen.add(meaning)
                    unique_meanings.append(meaning)

            if unique_meanings:
                # Для HTML используем прямое отображение
                details_parts.append(f"📖 <b>Значения:</b> {', '.join(unique_meanings)}")

        # Часть речи
        lexical = word_data.get("lexical_categoty") or word_data.get("lexical_category")
        if lexical and lexical.strip():
            clean_lexical = lexical.strip()
            # Убираем технические детали в скобках
            clean_lexical = re.sub(r'\([^)]*\)', '', clean_lexical).strip()
            if clean_lexical:
                details_parts.append(f"🏷️ <b>Часть речи:</b> {clean_lexical}")

        # Пример
        example = word_data.get("example")
        if example and example.strip():
            clean_example = example.strip()
            clean_example = re.sub(r'\s+', ' ', clean_example).strip()
            ex_tr = word_data.get("example_translation")

            if ex_tr and ex_tr.strip():
                clean_ex_tr = ex_tr.strip()
                clean_ex_tr = re.sub(r'\s+', ' ', clean_ex_tr).strip()
                details_parts.append(f"💬 <b>Пример:</b> {clean_example}")
                details_parts.append(f"🌍 <b>Перевод:</b> {clean_ex_tr}")
            else:
                details_parts.append(f"💬 <b>Пример:</b> {clean_example}")

        card_text = f"ℹ <b>Новое слово</b>\n\n<code>{clean_word}</code>"

        if details_parts:
            details_text = "\n".join(details_parts)
            card_text += f"\n\n<tg-spoiler>{details_text}</tg-spoiler>"

        return card_text

    except Exception as e:
        logger.error(f"Error formatting card text: {e}")
        return "ℹ <b>Новое слово</b>\n\nОшибка загрузки слова"


# ==================== ОСНОВНЫЕ ОБРАБОТЧИКИ ====================
@dp.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    """Обработчик команды /start"""
    try:
        user_id = message.from_user.id
        register_user(user_id)
        update_user_activity(user_id)

        welcome_text = """
📚 <b>Добро пожаловать в языковой тренажёр</b>

Изучайте кабардино-черкесский и карачаево-балкарский языки с системой, которая подстраивается под ваше запоминание и сама определяет идеальный момент для повторения слов.

⚡ <b>Как проходит обучение</b>

1. Выберите язык
2. Изучите слово
3. Оцените уровень знания:
   🟢 <b>Легко</b> — уверен(а) в слове
   🔵 <b>Нормально</b> — помню без проблем
   🟡 <b>С трудом</b> — вспоминаю частично
   🔴 <b>Не знаю</b> — вижу впервые

Алгоритм анализирует ваши ответы и формирует персональный график повторений, чтобы каждая минута приносила результат.

🔥 <b>Ежедневная серия</b>
Набирайте streak, поддерживайте привычку и наблюдайте, как растут знания.

Выберите язык, чтобы начать обучение.
        """

        await message.answer(welcome_text, reply_markup=language_keyboard(), parse_mode="HTML")
        await state.set_state(UserState.choosing_language)
        logger.info(f"User {user_id} started bot")

    except Exception as e:
        logger.error(f"Error in /start for user {message.from_user.id}: {e}")
        await message.answer("Произошла ошибка при запуске. Попробуйте ещё раз.")


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    """Обработчик команды /help"""
    help_text = """
📚 <b>Доступные команды:</b>

/start — Начать обучение
/help — Показать это сообщение
/stats — Показать статистику
/end — Завершить сессию
/lang — Сменить язык
/next — Следующее слово
/menu — Главное меню
/status — Статус системы

🎮 <b>Быстрые действия:</b>
• Нажмите <b>📊 Статистика</b> — чтобы увидеть прогресс
• Нажмите <b>🔁 Смена языка</b> — чтобы сменить язык
• Используйте кнопки под карточкой — для оценки слов

📞 <b>Если возникли проблемы:</b>
1. Попробуйте команду /end и затем /start
2. Убедитесь, что у вас стабильное интернет-соединение

Удачи в изучении языков! 🚀
    """
    await message.answer(help_text, parse_mode="HTML")
    logger.info(f"User {message.from_user.id} requested help")


@dp.message(Command("lang"))
async def cmd_lang(message: types.Message, state: FSMContext):
    """Быстрая смена языка"""
    await message.answer("<b>Выберите язык:</b>", reply_markup=language_keyboard(), parse_mode="HTML")
    await state.set_state(UserState.choosing_language)
    logger.info(f"User {message.from_user.id} requested language change via command")


@dp.message(Command("next"))
async def cmd_next(message: types.Message):
    """Быстрое получение следующего слова"""
    try:
        user_id = message.from_user.id
        uid = str(user_id)

        if uid not in user_data:
            await message.answer("Сначала используйте /start")
            return

        lang = user_data[uid].get("lang")
        if not lang:
            await message.answer("Сначала выберите язык: /lang")
            return

        await send_card(message.chat.id, user_id)
        logger.info(f"User {user_id} requested next word via command")

    except Exception as e:
        logger.error(f"Error in /next for user {message.from_user.id}: {e}")
        await message.answer("Произошла ошибка. Попробуйте /start")


@dp.message(Command("stats"))
async def cmd_stats(message: types.Message):
    """Обработчик команды /stats"""
    try:
        uid = str(message.from_user.id)
        if uid in user_data:
            await message.answer("📊 <b>Меню статистики</b>", reply_markup=stats_menu_keyboard(), parse_mode="HTML")
        else:
            await message.answer("Сначала используйте /start")
        logger.info(f"User {message.from_user.id} requested stats via command")
    except Exception as e:
        logger.error(f"Error in /stats for user {message.from_user.id}: {e}")
        await message.answer("Произошла ошибка при загрузке статистики.")


@dp.message(Command("end"))
async def cmd_end(message: types.Message, state: FSMContext):
    """Обработчик команды /end"""
    try:
        user_id = message.from_user.id
        deleted = delete_user(user_id)
        await state.clear()

        if deleted:
            await message.answer("✅ <b>Сессия завершена.</b>\nВсе ваши данные удалены.\n\nЧтобы начать заново: /start", parse_mode="HTML")
        else:
            await message.answer("🏁 <b>Сессия завершена.</b>\n\nЧтобы начать заново: /start", parse_mode="HTML")
        logger.info(f"User {user_id} ended session")

    except Exception as e:
        logger.error(f"Error in /end for user {message.from_user.id}: {e}")
        await message.answer("Произошла ошибка при завершении сессии.")


@dp.message(Command("menu"))
async def cmd_menu(message: types.Message):
    """Показать главное меню"""
    await message.answer("📱 <b>Главное меню</b>", reply_markup=menu_keyboard(), parse_mode="HTML")
    logger.info(f"User {message.from_user.id} requested menu")


@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    """Показать статус системы"""
    try:
        uid = str(message.from_user.id)
        if uid not in user_data:
            await message.answer("Вы не зарегистрированы. Используйте /start")
            return

        user_info = user_data[uid]
        lang = user_info.get("lang", "не выбран")
        total_words = len(user_info.get("word_scores", {}))
        learned_words = 0

        if lang and lang in user_info.get("progress_stats", {}):
            learned_words = user_info["progress_stats"][lang].get("learned_words", 0)

        dict_stats = dictionary.get_stats()
        loaded_dicts = dict_stats.get("loaded_dictionaries", [])
        total_memory = dict_stats.get("total_memory_mb", 0)

        kabard_size = dictionary.get_displayed_size("kabard")
        balkar_size = dictionary.get_displayed_size("balkar")

        status_text = f"""
📊 <b>Статус системы:</b>

👤 <b>Ваш профиль:</b>
• Язык: {lang if lang else 'не выбран'}
• Изучено слов: {total_words}
• Выучено слов: {learned_words}
• Streak: {user_info.get('streak', {}).get('current', 0)} дней

⚙️ <b>Словари:</b>
• Кабардино-черкесский: {kabard_size} слов
• Карачаево-балкарский: {balkar_size} слов

📊 <b>Система:</b>
• Всего пользователей: {len(user_data)}
• Загружено словарей: {len(loaded_dicts)} ({', '.join(loaded_dicts) if loaded_dicts else 'нет'})
• Память словарей: {total_memory:.1f} МБ
        """

        await message.answer(status_text, parse_mode="HTML")
        logger.info(f"User {message.from_user.id} requested system status")

    except Exception as e:
        logger.error(f"Error in /status for user {message.from_user.id}: {e}")
        await message.answer("Ошибка при получении статуса системы.")


# ==================== ОТПРАВКА КАРТОЧКИ ====================
async def send_card(chat_id, user_id: int):
    """Отправка карточки со словом"""
    try:
        uid = str(user_id)
        if uid not in user_data:
            await bot.send_message(chat_id, "Ваша сессия устарела. Используйте /start")
            return

        lang = user_data[uid].get("lang")
        if not lang:
            await bot.send_message(chat_id, "Сначала выберите язык: /lang")
            return

        word_data = await get_next_word(user_id)
        if not word_data:
            await bot.send_message(chat_id,
                                   "❌ <b>Не удалось загрузить слова</b>\nПопробуйте позже или используйте /start",
                                   parse_mode="HTML")
            logger.warning(f"No word data for user {user_id}")
            return

        word_text = word_data.get("word")
        if word_text:
            if "progress_stats" not in user_data[uid]:
                user_data[uid]["progress_stats"] = {
                    "kabard": {"total_seen": 0, "learned_words": 0},
                    "balkar": {"total_seen": 0, "learned_words": 0}
                }

            if lang not in user_data[uid]["progress_stats"]:
                user_data[uid]["progress_stats"][lang] = {"total_seen": 0, "learned_words": 0}

            user_data[uid]["progress_stats"][lang]["total_seen"] += 1
            save_user_data()

        text = await format_card_text(word_data)
        await bot.send_message(chat_id, text, reply_markup=get_review_keyboard(), parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error sending card to user {user_id}: {e}")
        await bot.send_message(chat_id, "❌ <b>Произошла ошибка</b>\nПопробуйте ещё раз или используйте /start",
                               parse_mode="HTML")


# ==================== ОБРАБОТКА КНОПОК ====================
@dp.callback_query()
async def handle_all_callbacks(callback: types.CallbackQuery, state: FSMContext):
    """Главный обработчик всех callback-запросов"""
    try:
        uid = str(callback.from_user.id)
        if uid not in user_data:
            await callback.answer("❌ Ваша сессия устарела. Используйте /start")
            return

        cmd = callback.data
        update_user_activity(callback.from_user.id)

        # --- ОЦЕНКА СЛОВА ---
        if cmd.startswith("review_"):
            try:
                quality = int(cmd.split("_")[1])
                if quality < 0 or quality > 3:
                    raise ValueError
            except (ValueError, IndexError):
                await callback.answer("❌ Неверный формат данных")
                return

            lang = user_data[uid].get("lang")
            if not lang:
                await callback.answer("❌ Сначала выберите язык")
                return

            last_word = user_data[uid].get("last_word")
            if not last_word:
                await callback.answer("❌ Нет текущего слова")
                return

            update_streak(callback.from_user.id)
            update_word_memory(last_word, quality, callback.from_user.id, lang)

            feedback_text = ["Не знаю", "С трудом", "Нормально", "Легко"][quality]
            feedback_emoji = ["🔴", "🟡", "🔵", "🟢"][quality]
            await callback.answer(f"{feedback_emoji} {feedback_text}")

            await send_card(callback.message.chat.id, callback.from_user.id)
            return

        # --- ПОМОЩЬ ---
        if cmd == "show_help":
            await callback.message.answer(
                "🆘 <b>Нужна помощь?</b>\nИспользуйте команду /help для получения подробной информации.",
                parse_mode="HTML"
            )
            await callback.answer()
            return

        # --- СТАТИСТИКА ---
        if cmd == "stats_menu":
            await callback.message.answer("📊 <b>Меню статистики</b>", reply_markup=stats_menu_keyboard(),
                                          parse_mode="HTML")
            await callback.answer()
            return

        if cmd == "view_stats":
            lang = user_data[uid].get("lang")
            if not lang:
                await callback.answer("❌ Сначала выберите язык")
                return

            total_words = dictionary.get_displayed_size(lang)

            word_scores = user_data[uid].get("word_scores", {})
            word_languages = user_data[uid].get("word_languages", {})

            lang_word_scores = {
                word: prob for word, prob in word_scores.items()
                if word_languages.get(word) == lang
            }

            total_seen = len(lang_word_scores)
            learned_words = user_data[uid]["progress_stats"].get(lang, {}).get("learned_words", 0)

            streak_data = user_data[uid].get("streak", {"current": 0, "longest": 0})
            current_streak = streak_data.get("current", 0)
            longest_streak = streak_data.get("longest", 0)

            seen_percent = (total_seen / total_words * 100) if total_words > 0 else 0
            progress_bar = create_progress_bar(seen_percent)

            lang_name = "Кабардино-черкесский" if lang == "kabard" else "Карачаево-балкарский"
            msg = (
                f"📊 <b>Ваша статистика</b>\n\n"
                f"<b>Язык:</b> {lang_name}\n"
                f"<b>Всего слов в словаре:</b> {total_words}\n\n"
                f"<b>Прогресс:</b>\n"
                f"• Изучено слов: {total_seen}\n"
                f"• Охват словаря: {seen_percent:.0f}%\n"
                f"• <b>Выучено слов: {learned_words}</b>\n\n"
                f"<b>Серия дней:</b>\n"
                f"• Текущая: <b>{current_streak} дней</b>\n"
                f"• Рекорд: {longest_streak} дней"
            )

            if seen_percent > 0:
                msg += f"\n\n📊 Прогресс: {progress_bar} {seen_percent:.0f}%"

            if current_streak > 0:
                if current_streak >= 7:
                    msg += f"\n\n🔥 Отличная работа! Уже {current_streak} дней подряд!"
                elif current_streak >= 30:
                    msg += f"\n\n🚀 Невероятно! Целый месяц!"

            await callback.message.answer(msg, reply_markup=after_stats_keyboard(), parse_mode="HTML")
            await callback.answer()
            return

        # --- СБРОС СТАТИСТИКИ ---
        if cmd in ["reset_kabard", "reset_balkar", "reset_all"]:
            if cmd == "reset_kabard":
                question = "кабардино-черкесский язык?"
                reset_type = "kabard"
            elif cmd == "reset_balkar":
                question = "карачаево-балкарский язык?"
                reset_type = "balkar"
            else:
                question = "ВСЮ статистику и прогресс?"
                reset_type = "all"

            await callback.message.answer(
                f"⚠️ <b>Внимание!</b>\n\n"
                f"Вы действительно хотите сбросить {question}\n"
                f"<i>Это действие нельзя отменить!</i>",
                reply_markup=reset_confirm_keyboard(reset_type),
                parse_mode="HTML"
            )
            await callback.answer()
            return

        if cmd.startswith("confirm_"):
            reset_type = cmd.replace("confirm_", "")

            if reset_type == "kabard":
                kabard_words = dictionary.get_word_set("kabard")
                user_data[uid]["word_scores"] = {
                    k: v for k, v in user_data[uid]["word_scores"].items()
                    if k not in kabard_words
                }
                user_data[uid]["word_languages"] = {
                    k: v for k, v in user_data[uid]["word_languages"].items()
                    if k not in kabard_words
                }
                user_data[uid]["word_history"] = {
                    k: v for k, v in user_data[uid]["word_history"].items()
                    if k not in kabard_words
                }
                if "progress_stats" in user_data[uid]:
                    user_data[uid]["progress_stats"]["kabard"] = {"total_seen": 0, "learned_words": 0}
                msg = "✅ <b>Статистика кабардино-черкесского языка сброшена</b>"

            elif reset_type == "balkar":
                balkar_words = dictionary.get_word_set("balkar")
                user_data[uid]["word_scores"] = {
                    k: v for k, v in user_data[uid]["word_scores"].items()
                    if k not in balkar_words
                }
                user_data[uid]["word_languages"] = {
                    k: v for k, v in user_data[uid]["word_languages"].items()
                    if k not in balkar_words
                }
                user_data[uid]["word_history"] = {
                    k: v for k, v in user_data[uid]["word_history"].items()
                    if k not in balkar_words
                }
                if "progress_stats" in user_data[uid]:
                    user_data[uid]["progress_stats"]["balkar"] = {"total_seen": 0, "learned_words": 0}
                msg = "✅ <b>Статистика карачаево-балкарского языка сброшена</b>"

            else:
                current_lang = user_data[uid].get("lang")
                user_data[uid] = {
                    "lang": current_lang,
                    "last_word": None,
                    "word_scores": {},
                    "word_languages": {},
                    "word_history": {},
                    "progress_stats": {
                        "kabard": {"total_seen": 0, "learned_words": 0},
                        "balkar": {"total_seen": 0, "learned_words": 0}
                    },
                    "streak": {
                        "current": 0,
                        "longest": 0,
                        "last_active": None
                    },
                    "algorithm_params": {
                        "learning_rate": 0.3,
                        "forgetting_rate": 0.15,
                        "confidence_threshold": 0.85
                    },
                    "created_at": datetime.now().isoformat(),
                    "last_activity": datetime.now().isoformat()
                }
                msg = "✅ <b>Вся статистика сброшена!</b>\nНачинаем с чистого листа."

            save_user_data()
            await callback.message.answer(msg, reply_markup=after_stats_keyboard(), parse_mode="HTML")
            await callback.answer()
            return

        if cmd == "cancel_reset":
            await callback.message.answer(
                "❌ <b>Сброс отменен</b>",
                reply_markup=stats_menu_keyboard(),
                parse_mode="HTML"
            )
            await callback.answer()
            return

        # --- НАВИГАЦИЯ ---
        if cmd == "back_to_learning":
            await callback.message.answer("↩️ <b>Возвращаемся к обучению...</b>", parse_mode="HTML")
            await send_card(callback.message.chat.id, callback.from_user.id)
            await callback.answer()
            return

        if cmd == "change_lang":
            await callback.message.answer("🌍 <b>Выберите язык:</b>", reply_markup=language_keyboard(),
                                          parse_mode="HTML")
            await state.set_state(UserState.choosing_language)
            await callback.answer()
            return

        if cmd == "continue_learning":
            try:
                await callback.message.delete()
            except:
                pass

            lang = user_data[uid].get("lang")
            if lang:
                await send_card(callback.message.chat.id, callback.from_user.id)
            else:
                await callback.message.answer("🌍 <b>Сначала выберите язык:</b>", reply_markup=language_keyboard(),
                                              parse_mode="HTML")
                await state.set_state(UserState.choosing_language)

            await callback.answer()
            return

        # --- ВЫБОР ЯЗЫКА ---
        if cmd in ["kabard", "balkar"]:
            user_data[uid]["lang"] = cmd
            save_user_data()

            await state.set_state(UserState.learning)
            lang_name = "Кабардино-черкесский" if cmd == "kabard" else "Карачаево-балкарский"

            await callback.message.answer(
                f"✅ <b>Язык выбран:</b> {lang_name}\n\n"
                f"Начинаем обучение! Первое слово:",
                parse_mode="HTML"
            )

            await send_card(callback.message.chat.id, callback.from_user.id)
            await callback.answer()
            return

    except Exception as e:
        logger.error(f"Error handling callback for user {callback.from_user.id}: {e}")
        try:
            await callback.answer("❌ Произошла ошибка. Попробуйте ещё раз или используйте /start")
        except:
            pass


# ==================== ОБРАБОТЧИК ОШИБОК ====================
@dp.errors()
async def errors_handler(update, exception):
    """Глобальный обработчик ошибок"""
    logger.error(f"Update {update} caused error: {exception}", exc_info=True)
    return True


# ==================== ЗАПУСК БОТА ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🤖 БОТ ЗАПУЩЕН С ЛЕНИВОЙ ЗАГРУЗКОЙ СЛОВАРЕЙ")
    print("=" * 60)
    print("📊 Статистика при запуске:")
    print(f"   • Пользователей: {len(user_data)}")

    dictionary.get_stats()

    dict_stats = dictionary.get_stats()
    print(f"   • Загружено словарей: {len(dict_stats['loaded_dictionaries'])}")
    print(f"   • Память словарей: {dict_stats['total_memory_mb']:.1f} МБ")
    print(f"   • Кабардино-черкесских слов: {dictionary.get_displayed_size('kabard')}")
    print(f"   • Карачаево-балкарских слов: {dictionary.get_displayed_size('balkar')}")
    print("=" * 60)
    print("💡 Примечание: Словари загружаются автоматически")
    print("=" * 60 + "\n")

    try:
        asyncio.run(dp.start_polling(bot))
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен пользователем")
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Bot crashed: {e}", exc_info=True)
        print(f"❌ Критическая ошибка: {e}")

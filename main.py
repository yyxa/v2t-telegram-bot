import logging
import os
import threading
import wave
import json
import asyncio
import time
import numpy as np
from jiwer import wer
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from pydub import AudioSegment
import whisper
from vosk import Model as VoskModel, KaldiRecognizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = ""

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

current_model = "whisper"
all_mode = False

logger.info("Загрузка моделей распознавания речи...")

# Whisper
whisper_model = whisper.load_model("small")

# Vosk
vosk_model = VoskModel("./vosk-model-small-ru-0.22/vosk-model-small-ru-0.22")

# Wav2Vec2
wav2vec_processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

logger.info("Модели загружены.")

@dp.message(Command("start"))
async def start(message: types.Message):
    global all_mode
    all_mode = False
    await message.reply(
        "Привет! Я бот для транскрибации голосовых сообщений.\n"
        "Текущая модель: Whisper\n"
        "Доступные команды:\n"
        "/model_whisper - Переключиться на Whisper\n"
        "/model_vosk - Переключиться на Vosk\n"
        "/model_wav2vec2 - Переключиться на Wav2Vec2\n"
        "/all - Переключиться на режим обработки всеми моделями\n"
        "/help - Получить инструкции по использованию бота"
    )

@dp.message(Command("help"))
async def help_command(message: types.Message):
    await message.reply(
        "Инструкции по использованию бота:\n"
        "1. Отправьте голосовое сообщение, и бот транскрибирует его в текст.\n"
        "2. Используйте команды для переключения модели распознавания речи:\n"
        "   - /model_whisper: Использовать модель Whisper (по умолчанию).\n"
        "   - /model_vosk: Использовать модель Vosk.\n"
        "   - /model_wav2vec2: Использовать модель Wav2Vec2.\n"
        "3. Используйте /all для переключения на режим обработки всеми моделями.\n"
        "4. Если есть вопросы, начните с команды /start."
    )

@dp.message(Command("model_whisper"))
async def model_whisper_command(message: types.Message):
    global current_model, all_mode
    all_mode = False
    current_model = "whisper"
    await message.reply("Текущая модель изменена на **Whisper**.", parse_mode='Markdown')

@dp.message(Command("model_vosk"))
async def model_vosk_command(message: types.Message):
    global current_model, all_mode
    all_mode = False
    if vosk_model is None:
        await message.reply("Модель Vosk не загружена или не найдена.")
        return
    current_model = "vosk"
    await message.reply("Текущая модель изменена на **Vosk**.", parse_mode='Markdown')

@dp.message(Command("model_wav2vec2"))
async def model_wav2vec2_command(message: types.Message):
    global current_model, all_mode
    all_mode = False
    if wav2vec_model is None or wav2vec_processor is None:
        await message.reply("Модель Wav2Vec2 не загружена или не найдена.")
        return
    current_model = "wav2vec2"
    await message.reply("Текущая модель изменена на **Wav2Vec2**.", parse_mode='Markdown')

@dp.message(Command("all"))
async def all_models_command(message: types.Message):
    global all_mode
    all_mode = True
    await message.reply("Режим обработки всеми моделями активирован.")

@dp.message(lambda message: message.content_type == types.ContentType.VOICE)
async def voice_handler(message: types.Message):
    global all_mode
    voice = message.voice
    file = await bot.get_file(voice.file_id)

    ogg_path = f"{voice.file_id}.ogg"
    wav_path = f"{voice.file_id}.wav"
    with open(ogg_path, "wb") as f:
        await bot.download_file(file.file_path, f)

    convert_to_wav(ogg_path, wav_path)

    if all_mode:
        metrics = []
        for model_name in ["whisper", "vosk", "wav2vec2"]:
            text, duration, cer, rtf = calculate_metrics(model_name, wav_path)
            metrics.append(f"Модель: {model_name}\nТекст: {text}\nВремя обработки: {duration:.2f} сек\nCER: {cer:.2f}\nRTF: {rtf:.2f}\n")

        await message.reply("\n".join(metrics))
    else:
        text = transcribe_speech(wav_path)
        if text:
            await message.reply(f"Распознанный текст ({current_model}):\n{text}")
        else:
            await message.reply("Извините, не удалось распознать голосовое сообщение.")

    clean_up([ogg_path, wav_path])

def convert_to_wav(input_path: str, output_path: str):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(output_path, format="wav")
    except Exception as e:
        logger.error(f"Ошибка при конвертации аудио: {e}")

def calculate_metrics(model_name: str, audio_path: str):
    wf = wave.open(audio_path, "rb")
    audio_duration = wf.getnframes() / wf.getframerate()
    wf.close()

    start_time = time.time()
    text = transcribe_speech(audio_path, model_name)
    end_time = time.time()

    duration = end_time - start_time

    reference_text = "пример текста для расчета метрик"

    cer = calculate_cer(reference_text, text) if text else float('inf')
    rtf = duration / audio_duration if audio_duration > 0 else float('inf')

    return text, duration, cer, rtf

def calculate_cer(reference: str, hypothesis: str):
    ref_len = len(reference)
    if ref_len == 0:
        return float('inf')
    return wer(reference, hypothesis) * ref_len / len(hypothesis)

def transcribe_speech(wav_path: str, model_name=None) -> str:
    global current_model
    if model_name:
        current_model = model_name

    text = ""
    try:
        if current_model == "whisper":
            result = whisper_model.transcribe(wav_path, language='ru')
            text = result["text"].strip()
        elif current_model == "vosk" and vosk_model is not None:
            wf = wave.open(wav_path, "rb")
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                logger.error("Некорректный формат WAV файла для Vosk.")
                return "Ошибка: WAV файл должен быть 16kHz, моно, 16-бит."
            rec = KaldiRecognizer(vosk_model, wf.getframerate())
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    pass
            result = rec.FinalResult()
            result_dict = json.loads(result)
            text = result_dict.get("text", "").strip()
            wf.close()
        elif current_model == "wav2vec2" and wav2vec_model is not None and wav2vec_processor is not None:
            wf = wave.open(wav_path, "rb")
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16) / 32768.0
            input_values = wav2vec_processor(audio_data, return_tensors="pt", sampling_rate=16000).input_values
            logits = wav2vec_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            text = wav2vec_processor.batch_decode(predicted_ids)[0]
            wf.close()
        else:
            logger.error
    except Exception as e:
        logger.error(f"Ошибка при транскрибации: {e}")
    return text

def clean_up(files_list):
    for f in files_list:
        if os.path.exists(f):
            os.remove(f)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

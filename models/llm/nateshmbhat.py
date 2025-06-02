import os
from datetime import datetime

import pyttsx3

def tts(text):
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"./gradio_temp/{formatted_time}.wav"

    engine = pyttsx3.init()
    # 设置语速
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 50)

    # 设置音调
    volume = engine.getProperty('volume')
    engine.setProperty('volume', volume + 0.25)
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename



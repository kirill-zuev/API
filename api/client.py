import requests
from loguru import logger

import snappy
import io
from pydub import AudioSegment

# def predict(data0: str, filter: str, lang: str) -> AudioSegment:
def predict(data0: str, lang: str) -> AudioSegment:
    """API
    :param data0: string for translate into audio
    :param filter: filter
    :param lang: language
    :return: 
    """

    server_ip = '0.0.0.0:9001'
    response = requests.post(
        f'http://{server_ip}/api/tts',
        json={'text': data0},
        # params={'filter': filter,
        params={
            'lang': lang}
        )
    response = response_handler(response)
    with open("./output/compressed_audio.snappy", "wb") as f:
        f.write(response.content)

    with open("./output/compressed_audio.snappy", "rb") as f:
        compressed_data = f.read()
    audio_bytes = snappy.uncompress(compressed_data)
    return AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

def response_handler(response):
    status_code = response.status_code
    if status_code == 500:
        raise RuntimeError(response.text)
    elif status_code == 415:
        logger.exception("Unsupported Data Type.")
        raise RuntimeError(response.text)
    elif status_code == 400:
        raise RuntimeError(response.text)
    elif status_code == 200:
        return response
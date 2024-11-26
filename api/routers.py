import torch
import pickle
import snappy
import io

from preprocessing import made_audio
from filter import compres
import schemas

from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import StreamingResponse

from TTS.api import TTS
# from TTS.bin.synthesize_new import main_tts
# from synthesize_new import main_tts
from transformers import VitsModel, VitsTokenizer

router = APIRouter()

@router.post('/load')
async def load_file(file: UploadFile):
    file_path = "./TTS/tests/data/ssml/input.ssml"
    
    buffer = io.BytesIO(await file.read())
    
    with open(file_path, "wb") as out_file:
        out_file.write(buffer.getvalue())
    
    return {"file_path": file_path}

@router.post('/tts')
async def main(request: schemas.Item, params: schemas.TTSParams = Depends()) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = ''
    file_path = ''
    if not params.ssml:
        data = request.text
    else:
        file_path = './TTS/tests/data/ssml/input.ssml'

    audio = []
    sample_rate = 24000
    if params.lang == 'ja':
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        s_new = data.split("。")
        if s_new[-1] == '':
            s_new.pop()
        for i in range(len(s_new)):
            wav = tts.tts(text=s_new[i], speaker_wav="./output.wav", language='ja')
            # wav = main_tts(s_new[i], "tts_models/multilingual/multi-dataset/xtts_v2", "server.wav", "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/A7-RU.mp3", 'ja')
            audio += wav
    elif params.lang == 'zh-cn':
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        s_new = data.split("。")
        if s_new[-1] == '':
            s_new.pop()
        for i in range(len(s_new)):
            wav = tts.tts(text=s_new[i], speaker_wav="./output.wav", language='zh-cn')
            # wav = main_tts(s_new[i], "tts_models/multilingual/multi-dataset/xtts_v2", "server.wav", "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/A7-RU.mp3", 'zh-cn')
            audio += wav
    elif params.lang == 'kaz':
        sample_rate = 16000
        model1 = VitsModel.from_pretrained("facebook/mms-tts-kaz")
        tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kaz")
        s_new = data.split(".")
        if s_new[-1] == '':
            s_new.pop()
        for i in range(len(s_new)):
            inputs = tokenizer(s_new[i], return_tensors="pt")
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                outputs = model1(input_ids)
            audio += outputs.waveform[0]
    elif params.lang == 'grc':
        sample_rate = 16000
        model1 = VitsModel.from_pretrained("facebook/mms-tts-grc")
        tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-grc")
        s_new = data.split(".")
        if s_new[-1] == '':
            s_new.pop()
        for i in range(len(s_new)):
            inputs = tokenizer(s_new[i], return_tensors="pt")
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                outputs = model1(input_ids)
            audio += outputs.waveform[0]
    else:
        audio = made_audio(data, params.lang, file_path)
    
    compressed_audio = compres(audio, sample_rate)
    compressed_audio.export("./server.wav", format='wav')

    byte_buffer = io.BytesIO()
    compressed_audio.export(byte_buffer, format="wav")

    audio_bytes = byte_buffer.getvalue()
    headers = { "Content-Disposition": "attachment; filename=audio.wav" }
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav", headers=headers)
    # compressed_data = snappy.compress(audio_bytes)
    # headers = { "Content-Disposition": "attachment; filename=compressed_audio.snappy" }
    # return StreamingResponse(io.BytesIO(compressed_data), media_type="application/octet-stream", headers=headers)

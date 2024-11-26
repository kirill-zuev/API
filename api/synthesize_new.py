#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import sys
from argparse import RawTextHelpFormatter

# pylint: disable=redefined-outer-name, unused-argument
from pathlib import Path

import os
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
import torch
from scipy.signal import butter, lfilter
# from clearml import Task, Logger

description = """
Synthesize speech on command line.

You can either use your trained model or choose a model from the provided list.

If you don't specify any models, then it uses LJSpeech based English model.

#### Single Speaker Models

- List provided models:

  ```
  $ tts --list_models
  ```

- Get model info (for both tts_models and vocoder_models):

  - Query by type/name:
    The model_info_by_name uses the name as it from the --list_models.
    ```
    $ tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
    ```
    For example:
    ```
    $ tts --model_info_by_name tts_models/tr/common-voice/glow-tts
    $ tts --model_info_by_name vocoder_models/en/ljspeech/hifigan_v2
    ```
  - Query by type/idx:
    The model_query_idx uses the corresponding idx from --list_models.

    ```
    $ tts --model_info_by_idx "<model_type>/<model_query_idx>"
    ```

    For example:

    ```
    $ tts --model_info_by_idx tts_models/3
    ```

  - Query info for model info by full name:
    ```
    $ tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
    ```

- Run TTS with default models:

  ```
  $ tts --text "Text for TTS" --out_path output/path/speech.wav
  ```

- Run TTS and pipe out the generated TTS wav file data:

  ```
  $ tts --text "Text for TTS" --pipe_out --out_path output/path/speech.wav | aplay
  ```

- Run a TTS model with its default vocoder model:

  ```
  $ tts --text "Text for TTS" --model_name "<model_type>/<language>/<dataset>/<model_name>" --out_path output/path/speech.wav
  ```

  For example:

  ```
  $ tts --text "Text for TTS" --model_name "tts_models/en/ljspeech/glow-tts" --out_path output/path/speech.wav
  ```

- Run with specific TTS and vocoder models from the list:

  ```
  $ tts --text "Text for TTS" --model_name "<model_type>/<language>/<dataset>/<model_name>" --vocoder_name "<model_type>/<language>/<dataset>/<model_name>" --out_path output/path/speech.wav
  ```

  For example:

  ```
  $ tts --text "Text for TTS" --model_name "tts_models/en/ljspeech/glow-tts" --vocoder_name "vocoder_models/en/ljspeech/univnet" --out_path output/path/speech.wav
  ```

- Run your own TTS model (Using Griffin-Lim Vocoder):

  ```
  $ tts --text "Text for TTS" --model_path path/to/model.pth --config_path path/to/config.json --out_path output/path/speech.wav
  ```

- Run your own TTS and Vocoder models:

  ```
  $ tts --text "Text for TTS" --model_path path/to/model.pth --config_path path/to/config.json --out_path output/path/speech.wav
      --vocoder_path path/to/vocoder.pth --vocoder_config_path path/to/vocoder_config.json
  ```

#### Multi-speaker Models

- List the available speakers and choose a <speaker_id> among them:

  ```
  $ tts --model_name "<language>/<dataset>/<model_name>"  --list_speaker_idxs
  ```

- Run the multi-speaker TTS model with the target speaker ID:

  ```
  $ tts --text "Text for TTS." --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>"  --speaker_idx <speaker_id>
  ```

- Run your own multi-speaker TTS model:

  ```
  $ tts --text "Text for TTS" --out_path output/path/speech.wav --model_path path/to/model.pth --config_path path/to/config.json --speakers_file_path path/to/speaker.json --speaker_idx <speaker_id>
  ```

### Voice Conversion Models

```
$ tts --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>" --source_wav <path/to/speaker/wav> --target_wav <path/to/reference/wav>
```
"""

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def main_tts(text, model_name, out_path, speaker_wav, language_idx, device0="cpu"):
    parser = argparse.ArgumentParser(
        description=description.replace("    ```\n", ""),
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--list_models",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained TTS and vocoder models.",
    )

    parser.add_argument(
        "--model_info_by_idx",
        type=str,
        default=None,
        help="model info using query format: <model_type>/<model_query_idx>",
    )

    parser.add_argument(
        "--model_info_by_name",
        type=str,
        default=None,
        help="model info using query format: <model_type>/<language>/<dataset>/<model_name>",
    )

    parser.add_argument("--text", type=str, default=None, help="Text to generate speech.")

    # Args for running pre-trained TTS models.
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained TTS models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument(
        "--vocoder_name",
        type=str,
        default=None,
        help="Name of one of the pre-trained  vocoder models in format <language>/<dataset>/<model_name>",
    )

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="tts_output.wav",
        help="Output wav file path.",
    )
    parser.add_argument("--use_cuda", type=bool, help="Run model on CUDA.", default=False)
    parser.add_argument("--device", type=str, help="Device to run model on.", default="cpu")
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument(
        "--encoder_path",
        type=str,
        help="Path to speaker encoder model file.",
        default=None,
    )
    parser.add_argument("--encoder_config_path", type=str, help="Path to speaker encoder config file.", default=None)
    parser.add_argument(
        "--pipe_out",
        help="stdout the generated TTS wav file for shell pipe.",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    
    # args for multi-speaker synthesis
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--language_ids_file_path", type=str, help="JSON file for multi-lingual model.", default=None)
    parser.add_argument(
        "--speaker_idx",
        type=str,
        help="Target speaker ID for a multi-speaker TTS model.",
        default=None,
    )
    parser.add_argument(
        "--language_idx",
        type=str,
        help="Target language ID for a multi-lingual TTS model.",
        default=None,
    )
    parser.add_argument(
        "--speaker_wav",
        nargs="+",
        help="wav file(s) to condition a multi-speaker TTS model with a Speaker Encoder. You can give multiple file paths. The d_vectors is computed as their average.",
        default=None,
    )
    parser.add_argument("--gst_style", help="Wav path file for GST style reference.", default=None)
    parser.add_argument(
        "--capacitron_style_wav", type=str, help="Wav path file for Capacitron prosody reference.", default=None
    )
    parser.add_argument("--capacitron_style_text", type=str, help="Transcription of the reference.", default=None)
    parser.add_argument(
        "--list_speaker_idxs",
        help="List available speaker ids for the defined multi-speaker model.",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--list_language_idxs",
        help="List available language ids for the defined multi-lingual model.",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    # aux args
    parser.add_argument(
        "--save_spectogram",
        type=bool,
        help="If true save raw spectogram for further (vocoder) processing in out_path.",
        default=False,
    )
    parser.add_argument(
        "--reference_wav",
        type=str,
        help="Reference wav file to convert in the voice of the speaker_idx or speaker_wav",
        default=None,
    )
    parser.add_argument(
        "--reference_speaker_idx",
        type=str,
        help="speaker ID of the reference_wav speaker (If not provided the embedding will be computed using the Speaker Encoder).",
        default=None,
    )
    parser.add_argument(
        "--progress_bar",
        type=str2bool,
        help="If true shows a progress bar for the model download. Defaults to True",
        default=True,
    )

    # voice conversion args
    parser.add_argument(
        "--source_wav",
        type=str,
        default=None,
        help="Original audio file to convert in the voice of the target_wav",
    )
    parser.add_argument(
        "--target_wav",
        type=str,
        default=None,
        help="Target audio file to convert in the voice of the source_wav",
    )

    parser.add_argument(
        "--voice_dir",
        type=str,
        default=None,
        help="Voice dir for tortoise model",
    )

    args = parser.parse_args()

    args.text = text
    args.model_name = model_name
    args.out_path = out_path
    args.speaker_wav = speaker_wav
    args.language_idx = language_idx

    # print the description if either text or list_models is not set
    check_args = [
        args.text,
        args.list_models,
        args.list_speaker_idxs,
        args.list_language_idxs,
        args.reference_wav,
        args.model_info_by_idx,
        args.model_info_by_name,
        args.source_wav,
        args.target_wav,
    ]
    if not any(check_args):
        parser.parse_args(["-h"])

    pipe_out = sys.stdout if args.pipe_out else None

    with contextlib.redirect_stdout(None if args.pipe_out else sys.stdout):
        # Late-import to make things load faster
        # from TTS.api import TTS
        from TTS.utils.manage import ModelManager
        from TTS.utils.synthesizer import Synthesizer

        # load model manager
        path = Path(__file__).parent / "../.models.json"
        manager = ModelManager(path, progress_bar=args.progress_bar)

        tts_path = None
        tts_config_path = None
        speakers_file_path = None
        language_ids_file_path = None
        vocoder_path = None
        vocoder_config_path = None
        encoder_path = None
        encoder_config_path = None
        vc_path = None
        vc_config_path = None
        model_dir = None

        device = args.device
        if args.use_cuda:
            device = "cuda"
        device = device0
        print(device)

        model_dir = "/home/ubuntu/projects/kp.zuev/voicegen/tts_models--multilingual--multi-dataset--xtts_v2/"
        # load models
        synthesizer = Synthesizer(
            tts_path,
            tts_config_path,
            speakers_file_path,
            language_ids_file_path,
            vocoder_path,
            vocoder_config_path,
            encoder_path,
            encoder_config_path,
            vc_path,
            vc_config_path,
            model_dir,
            args.voice_dir,
        ).to(device)

        # RUN THE SYNTHESIS
        wav = []
        dict_clearml = {}
        if model_dir is not None:
            wav, dict_clearml = synthesizer.tts(
                args.text, speaker_name=args.speaker_idx, language_name=args.language_idx, speaker_wav=args.speaker_wav
            )

        # save the results
        print(" > Saving output to {}".format(args.out_path))

        audio_np = np.array(wav)
        f_audio = butter_lowpass_filter(audio_np, cutoff=4500, fs=24000)
        wav = torch.tensor(f_audio)
        synthesizer.save_wav(wav, args.out_path, pipe_out=pipe_out)

        # task = Task.init(
        #     project_name='voicegen', 
        #     task_name='test'
        # )
        # log = Logger.current_logger()
        # log.report_text(" > Text: {}".format(args.text))
        # log.report_text(" > Model: {}".format(dict_clearml['model_dir']))
        # log.report_text(" > Device: {}".format(device))
        # log.report_single_value(name='process_time', value=dict_clearml['process_time'])
        # log.report_single_value(name='real_time_factor', value=dict_clearml['real_time_factor'])
        # task.upload_artifact(name='audio.raw', artifact_object=args.out_path)
        return wav

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    y = lfilter(b, a, data)
    return y

def prep0(s):
    new = s.replace('<','')
    new = new.replace('>','.')
    new = new.replace('{','')
    new = new.replace('}','.')
    new = new.replace('[','')
    new = new.replace(']','.')
    new = new.replace('(','')
    new = new.replace(')','.')
    new = new.replace('«','')
    new = new.replace('»','.')
    new = new.replace('"','.')
    new = new.replace('-',' ')
    new = new.replace('!','.')
    new = new.replace(',',' - ')
    new = new.split()
    return ' '.join(new)

def number_to_words(num, lang, prep_str=False):
    # 0 <= num <= 9999):
    if lang == 'ru' and not prep_str:
        ones = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
        teens = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
                "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
        tens = ["", "десять", "двадцать", "тридцать", "сорок", "пятьдесят",
                "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
        hundreds = ["", "сто", "двести", "триста", "четыреста", "пятьсот",
                    "шестьсот", "семьсот", "восемьсот", "девятьсот"]
        thousands = ["", "одна тысяча", "две тысячи", "три тысячи", "четыре тысячи",
                    "пять тысяч", "шесть тысяч", "семь тысяч", "восемь тысяч", "девять тысяч", "десять тысяч"]
    if lang == 'ru' and prep_str:
        ones = ["нуля", "одного", "двух", "трёх", "четырёх", "пяти", "шести", "семи", "восьми", "девяти"]
        teens = ["десяти", "одиннадцати", "двенадцати", "тринадцати", "четырнадцати", "пятнадцати",
          "шестнадцати", "семнадцати", "восемнадцати", "девятнадцати"]
        tens = ["", "десяти", "двадцати", "тридцати", "сорока", "пятидесяти",
          "шестидесяти", "семидесяти", "восьмидесяти", "девяносто"]
        hundreds = ["", "ста", "двухсот", "трёхсот", "четырёхсот", "пятисот",
              "шестисот", "семисот", "восьмисот", "девятьсот"]
        thousands = ["", "одной тысячи", "двух тысяч", "трёх тысяч", "четырёх тысяч",
              "пяти тысяч", "шести тысяч", "семи тысяч", "восьми тысяч", "девяти тысяч", "десяти тысяч"]

    if lang == 'en':
        ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
              "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "ten", "twenty", "thirty", "forty", "fifty",
              "sixty", "seventy", "eighty", "ninety"]
        hundreds = ["", "one hundred", "two hundred", "three hundred", "four hundred", "five hundred",
                  "six hundred", "seven hundred", "eight hundred", "nine hundred"]
        thousands = ["", "one thousand", "two thousand", "three thousand", "four thousand",
                  "five thousand", "six thousand", "seven thousand", "eight thousand", "nine thousand"]

    if lang == 'it':
        ones = ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove"]
        teens = ["dieci", "undici", "dodici", "tredici", "quattordici", "quindici",
              "sedici", "diciassette", "diciotto", "diciannove"]
        tens = ["", "dieci", "venti", "trenta", "quaranta", "cinquanta",
              "sessanta", "settanta", "ottanta", "novanta"]
        hundreds = ["", "cento", "duecento", "trecento", "quattrocento", "cinquecento",
                  "seicento", "settecento", "ottocento", "novecento"]
        thousands = ["", "mille", "duemila", "tremila", "quattromila",
                  "cinquemila", "seimila", "settemila", "ottomila", "novemila"]

    if lang == 'fr':
        ones = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"]
        teens = ["dix", "onze", "douze", "treize", "quatorze", "quinze",
              "seize", "dix-sept", "dix-huit", "dix-neuf"]
        tens = ["", "dix", "vingt", "trente", "quarante", "cinquante",
              "soixante", "soixante-dix", "quatre-vingts", "quatre-vingt-dix"]
        hundreds = ["", "cent", "deux cents", "trois cents", "quatre cents", "cinq cents",
                  "six cents", "sept cents", "huit cents", "neuf cents"]
        thousands = ["", "mille", "deux mille", "trois mille", "quatre mille",
                  "cinq mille", "six mille", "sept mille", "huit mille", "neuf mille"]

    if num == 0:
        return ones[0]
    words = []
    if num // 1000 > 0:
        words.append(thousands[num // 1000])
        num %= 1000
    if num // 100 > 0:
        words.append(hundreds[num // 100])
        num %= 100
    if num > 0:
        if 10 <= num < 20:
            words.append(teens[num - 10])
        else:
            if num // 10 > 0:
                words.append(tens[num // 10])
            if num % 10 > 0:
                words.append(ones[num % 10])
    return " ".join(words)

def prep(s, lang):
    new = s.split()
    numb = ['0','1','2','3','4','5','6','7','8','9']
    i = 0

    pred = ['после', 'до', 'течение', 'продолжение']
    pred_str = new[0]
    while i < len(new):
        for j in numb:
            if j in new[i]:
                nuber_str = ''
                new_str = ''
                for k in new[i]:
                    if k not in numb:
                        if nuber_str != '':
                            if pred_str in pred:
                                nuber_str = number_to_words(int(nuber_str), lang, True)
                            else:
                                nuber_str = number_to_words(int(nuber_str), lang, False)
                            new_str += ' ' + nuber_str + ' '
                        nuber_str = ''
                        new_str += k
                    else:
                        nuber_str += k
                if nuber_str != '':
                    if pred_str in pred:
                        nuber_str = number_to_words(int(nuber_str), lang, True)
                    else:
                        nuber_str = number_to_words(int(nuber_str), lang, False)
                    new_str += ' ' + nuber_str
                new[i] = new_str
        pred_str = new[i]
        if lang == 'ru':
            if 'бизнес' in new[i]:
                new[i] = new[i].replace('бизнес', 'бизнэс')
            if 'Sky' in new[i]:
                new[i] = new[i].replace('Sky', 'Скай')
            if 'Priority' in new[i]:
                new[i] = new[i].replace('Priority', 'Прайёрити')
            if 'стенд' in new[i]:
                new[i] = new[i].replace('стенд', 'стэнд')
            if 'A' in new[i]:
                new[i] = new[i].replace('A', 'А')
            if 'B' in new[i]:
                new[i] = new[i].replace('B', 'Бэ')
            if 'C' in new[i]:
                new[i] = new[i].replace('C', 'Цэ')
            if 'D' in new[i]:
                new[i] = new[i].replace('D', 'Дэ')
            if 'E' in new[i]:
                new[i] = new[i].replace('E', 'Йе')
            if 'F' in new[i]:
                new[i] = new[i].replace('F', 'Ф')
        i += 1
    for i in range(len(new)):
        if '.' in new[i]:
            if i == len(new)-1:
                new[i] = new[i].replace('.','')
    return (' '.join(new)).replace("  ", " ")

def phonem(text):
    from omogre import Transcriptor
    transcriptor_data_path = "omogre_data"
    transcriptor = Transcriptor(data_path=transcriptor_data_path)
    tts_text = ' '.join(transcriptor([text]))
    return tts_text

if __name__ == "__main__":
    # s = "Уважаемые пассажиры рейса 10 16 авиакомпании Аэрофлот - Российские авиалинии в Калининград. Посадка в самолёт начнётся через несколько минут, выход номер 120. При посадке в самолёт пассажиров бизнес-класса, участников программы Аэрофлот Бонус. платинового, золотого, серебряного уровней, а также пассажиров тарифной группы Максимум. просим воспользоваться коридором Sky Priority. Пассажиров с детьми до 7 лет просим обращаться к представителю авиакомпании для приоритетной посадки в самолёт."
    # s = "Уважаемые пассажиры, прибывшие рейсом 15 87 Аэрофлот-Российские авиалинии из Чебоксары. Приглашаем Вас к транспортёру номер три для получения багажа. Во избежание обменов багажа, убедительно просим Вас сверять номера багажных бирок на багаже с бирками в Ваших авиабилетах. Благодарим Вас за понимание. Негабаритный багаж и детские коляски Вы можете получить у транспортёра 1."
    s = "Уважаемые пассажиры! Информационное табло и телемониторы, находящиеся в терминале, по техническим причинам временно не работают. Убедительно просим вас внимательно слушать звуковые объявления. По всем интересующим вас вопросам просим обращаться на стойку информации в зале вылета Терминала Йе"
    # s = "Вниманию прибывших пассажиров! Выдача багажа задерживается в связи с активной грозовой деятельностью в районе аэропорта Шереметьево. Авиакомпания приносит извинения за доставленные неудобства."
    # s = 'Уважаемые пассажиры! На основании требований Федерального закона об ограничении курения табака, курение разрешено только в специально отведённых местах аэровокзала, обозначенных знаком .Место для курения. Спасибо за понимание.'
    # s = 'Уважаемые пассажиры! В соответствии с Правилами провоза жидкостей в салонах воздушных судов, пассажирам разрешается провоз в ручной клади только неопасных жидкостей, аэрозолей и гелей в ёмкостях, не превышающих сто миллилитров. Суммарный объем провозимых в ручной клади жидкостей не должен превышать одного литра на пассажира. Все жидкости, превышающие указанные нормы, должны сдаваться в багаж. Дополнительную информацию Вы можете получить на специальных информационных стендах, на стойках информации и на пунктах досмотра.'
    # s = 'Уважаемые пассажиры! В соответствии с Правилами провоза жидкостей в салонах воздушных судов, пассажирам разрешается провоз в ручной клади только неопасных жидкостей, аэрозолей и гелей в емкостях, не превышающих сто миллилитров. Суммарный объем провозимых в ручной клади жидкостей не должен превышать одного литра на пассажира. Все жидкости, превышающие указанные нормы, должны сдаваться в багаж. Дополнительную информацию Вы можете получить на специальных информационных стендах, на стойках информации и на пунктах досмотра.'
    with open("/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/audio.txt", 'r', encoding='utf-8') as file:
        results = []
        for line in file:
            line = line.strip()
            parts = line.split('|')
            code = parts[0]
            message = parts[1]
            message = prep0(message)
            message = prep(message, 'ru')
            print(message)
            print(code)
            main_tts(message, "tts_models/multilingual/multi-dataset/xtts_v2", f"./19_audio_5e-06_eval_10%_train_400_AdamW_batch=1_old_model_f/{code}.wav", "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/A7-RU.mp3", "ru", "cuda")
    # s = 'Уважаемые пассажиры! Регистрация на рейсы начинается не менее чем за 2 часа и заканчивается не позднее чем за 40 минут до времени вылета вашего рейса.'
    # s = prep0(s)
    # s = prep(s, 'ru')
    # print(s)
    # main_tts(s, "tts_models/multilingual/multi-dataset/xtts_v2", "./Output/output1.wav", "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/A7-RU.wav", "ru", "cuda")

    # s = "Уважаемые пассажиры рейса 10 16 авиакомпании Аэрофлот - Российские авиалинии в Калининград. Посадка в самолёт начнётся через несколько минут, выход номер 120. При посадке в самолёт пассажиров бизнес-класса, участников программы Аэрофлот Бонус. платинового, золотого, серебряного уровней, а также пассажиров тарифной группы Максимум. просим воспользоваться коридором Sky Priority. Пассажиров с детьми до 7 лет просим обращаться к представителю авиакомпании для приоритетной посадки в самолёт."
    # s = prep0(s)
    # s = prep(s, 'ru')
    # print(s)
    # main_tts(s, "tts_models/multilingual/multi-dataset/xtts_v2", "./Output/output1.wav", "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/A7-RU.mp3", "ru", "cuda")
    # s = "Уважаемые пассажиры, прибывшие рейсом 15 87 Аэрофлот-Российские авиалинии из Чебоксары. Приглашаем Вас к транспортёру номер три для получения багажа. Во избежание обменов багажа, убедительно просим Вас сверять номера багажных бирок на багаже с бирками в Ваших авиабилетах. Благодарим Вас за понимание. Негабаритный багаж и детские коляски Вы можете получить у транспортёра 1."
    # s = prep0(s)
    # s = prep(s, 'ru')
    # print(s)
    # main_tts(s, "tts_models/multilingual/multi-dataset/xtts_v2", "./Output/output2.wav", "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/A7-RU.mp3", "ru", "cuda")
    # s = "Уважаемые пассажиры! Информационное табло и телемониторы, находящиеся в терминале, по техническим причинам временно не работают. Убедительно просим вас внимательно слушать звуковые объявления. По всем интересующим вас вопросам просим обращаться на стойку информации в зале вылета Терминала Е."
    # s = prep0(s)
    # s = prep(s, 'ru')
    # print(s)
    # main_tts(s, "tts_models/multilingual/multi-dataset/xtts_v2", "./Output/output3.wav", "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/A7-RU.mp3", "ru", "cuda")
    # s = "Вниманию прибывших пассажиров! Выдача багажа задерживается в связи с активной грозовой деятельностью в районе аэропорта Шереметьево. Авиакомпания приносит извинения за доставленные неудобства."
    # s = prep0(s)
    # s = prep(s, 'ru')
    # print(s)
    # main_tts(s, "tts_models/multilingual/multi-dataset/xtts_v2", "./Output/output4.wav", "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/A7-RU.mp3", "ru", "cuda")

    # s = "Уважаемые пассажиры и гости аэропорта Шереметьево. С расписанием движения общественного транспорта до города Москвы можно ознакомиться на специальных информационных стэндах и на стойках информации - расположенных в терминалах аэропорта"
    s = prep0(s)
    s = prep(s, 'ru')
    print(s)
    # s = phonem(s)
    
    main_tts(s, "tts_models/multilingual/multi-dataset/xtts_v2", "./Output/result.wav", "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/A7-RU.mp3", "ru", "cuda")

    # def process_file(input_file, output_file):
    #         with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    #             for line in infile:
    #                 parts = line.strip().split('|')
    #                 if len(parts) == 3:
    #                     number, text1, text2 = parts
    #                     text1 = prep0(text1)
    #                     text2 = prep0(text2)
    #                     processed_text1 = phonem(text1)
    #                     processed_text2 = phonem(text2)
    #                     outfile.write(f"{number}|{processed_text1}|{processed_text2}\n")
    #                     print(processed_text1)

    # input_filename = "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/audio.txt"
    # output_filename = "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/audiop.txt"

    # process_file(input_filename, output_filename)

    # input_folder = "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/wavs/"
    # output_folder = "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/denoised_wavs/"

    # for filename in os.listdir(input_folder):
    #     if filename.endswith(".wav"):
    #         file_path = os.path.join(input_folder, filename)
    #         y, sr = librosa.load(file_path, sr=None)
    #         y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=1)
    #         output_path = os.path.join(output_folder, filename)
    #         sf.write(output_path, y_denoised, sr)

    #         print(f"{filename}")

import argparse
import sys
from argparse import RawTextHelpFormatter

# pylint: disable=redefined-outer-name, unused-argument
import numpy as np
from pathlib import Path
from gruut import sentences
from matplotlib.style import available

import xml
import xml.etree.ElementTree as ET
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

def main_tts_ssml(file):
    description = """Synthesize speech from an SSML file.
You can either use your trained model or choose a model from the provided list.
"""
    parser = argparse.ArgumentParser(
        description=description.replace("    ```\n", ""),
        formatter_class=RawTextHelpFormatter,
    )
    
    parser.add_argument("--file", type=str, default=None, help="Path to the SSML file.")
    parser.add_argument("--use_cuda", type=bool, help="Run model on CUDA.", default=False)
    parser.add_argument("--text", type=str, default=None, help="Text to generate speech.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/vctk/vits",
        help="Name of one of the pre-trained TTS models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="tts_output.wav",
        help="Output wav file path.",
    )
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
    parser.add_argument(
        "--voice_dir",
        type=str,
        default=None,
        help="Voice dir for tortoise model",
    )

    args = parser.parse_args()

    args.file = file
    # args.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    # args.speaker_wav = "Света-Иванова.wav"
    # args.language_idx = "en"

    path = Path(__file__).parent / "../.models.json"
    manager = ModelManager(path)

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

    if args.file is None:
        print("Please specify the SSML file.")
        sys.exit(1)

    if args.model_name is not None:
        model_path, config_path, model_item = manager.download_model(args.model_name)
        if model_item["model_type"] == "tts_models":
            tts_path = model_path
            tts_config_path = config_path
        if model_item["model_type"] == "voice_conversion_models":
            vc_path = model_path
            vc_config_path = config_path
        if model_item.get("author", None) == "fairseq" or isinstance(model_item["model_url"], list):
            model_dir = model_path
            tts_path = None
            tts_config_path = None
            args.vocoder_name = None

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
        args.use_cuda,
    )

    with open(args.file, "r") as f:
        ssml_text = f.read()
    
    available_speakers = []
    default_speaker = None
    if tts_path is not None:
        available_speakers = list(synthesizer.tts_model.speaker_manager.speaker_names)
        default_speaker = available_speakers[0]
    # elif model_dir is not None:
    #     wav = synthesizer.tts(
    #         args.text, speaker_name=args.speaker_idx, language_name=args.language_idx, speaker_wav=args.speaker_wav
    #     )

    full_sent = []
    orderedSpeakers = []
    for sent in sentences(ssml_text, ssml=True, espeak=True):
        sub_sent = []
        current_speaker = None
        index = -1
        for word in sent:
            if word.voice == '':
                word.voice = default_speaker
                if args.model_name != "tts_models/en/vctk/vits":
                    if index == -1:
                        sub_sent.append([''.join(word.phonemes)])
            if word.voice != current_speaker:
                current_speaker = word.voice
                sub_sent.append([''.join(word.phonemes)])
                orderedSpeakers.append(current_speaker)
                index += 1
            else:
                if index == -1:
                    index += 1
                else:
                    sub_sent[index].append(''.join(word.phonemes))
        full_sent.append(sub_sent)

    wavs = []
    for sub_sent in full_sent:
        for sent in sub_sent:
            sent = ' '.join(sent)
            if args.model_name == "tts_models/en/vctk/vits":
                speaker = orderedSpeakers[len(wavs)-1]
                if speaker not in available_speakers:
                    print(f" [!] Speaker {speaker} is not available. Using {default_speaker} instead.")
                    speaker = default_speaker
                wavs.append(synthesizer.tts(sent, speaker, None, None, ssml=True))
            else:
                wavs.append(synthesizer.tts(sent, speaker_name=args.speaker_idx, language_name=args.language_idx, speaker_wav=args.speaker_wav, ssml=True))

    final_wav = np.array([], dtype=np.float32)
    for wav in wavs:
        for sub_wav in wav:
            final_wav = np.append(final_wav, sub_wav)

    print(" > Saving output to {}".format(args.out_path))
    synthesizer.save_wav(final_wav, args.out_path)

    return final_wav

if __name__ == "__main__":
    main_tts_ssml("tests/data/ssml/mvp0.ssml")

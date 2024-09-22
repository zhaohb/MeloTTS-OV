from melo.api import TTS
from pathlib import Path
import time
import argparse
# Speed is adjustable
speed = 1.0


use_ov = True  ## Used to control whether to use torch or openvino
use_int8 = True
speech_enhance = True
lang = "EN" # or ZH


# Parse args for ov device
parser = argparse.ArgumentParser(description="Select inference devices for TTS and BERT")

parser.add_argument("--tts_device", type=str, choices=["CPU", "GPU"], default="CPU",
                    help="Select inference device for TTS: CPU or GPU")
parser.add_argument("--bert_device", type=str, choices=["CPU", "GPU", "NPU"], default="CPU",
                    help="Select inference device for BERT: CPU GPU or NPU")
parser.add_argument("--language", type=str, default="EN",
                    help="Specify the language for the models: ZH or EN")

# Parse command-line arguments
args = parser.parse_args()
# ov device
tts_device = args.tts_device
bert_device = args.bert_device
lang = args.language

if speech_enhance:
    from df.enhance import enhance, init_df, load_audio, save_audio
    import torchaudio
    def process_audio(input_file: str, output_file: str, new_sample_rate: int = 48000):
        """
        Load an audio file, enhance it using a DeepFilterNet, and save the result.

        Parameters:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the enhanced audio file.
        new_sample_rate (int): Desired sample rate for the output audio file (default is 48000 Hz).
        """

        model, df_state, _ = init_df()
        audio, sr = torchaudio.load(input_file)
        
        # Resample the WAV file to meet the requirements of DeepFilterNet
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=new_sample_rate)
        resampled_audio = resampler(audio)

        enhanced = enhance(model, df_state, resampled_audio)

        # Save the enhanced audio
        save_audio(output_file, enhanced, df_state.sr())

if lang == "ZH":
    text = "我们探讨如何在 Intel 平台上转换和优化artificial intelligence 模型"
elif lang == "EN":
    text = "For Intel platforms, we explore the methods for converting and optimizing models."

model = TTS(language=lang, tts_device=args.tts_device, bert_device=args.bert_device, use_int8=use_int8)
speaker_ids = model.hps.data.spk2id

speakers = list(speaker_ids.keys())


dur_time_list = []
loop_num = 1

if use_ov:
    ov_path = f"tts_ov_{lang}"
    if not Path(ov_path).exists():
        model.tts_convert_to_ov(ov_path, language= lang)
    model.ov_model_init(ov_path, language = lang)

for i in range(loop_num):
    if not use_ov:
         for speaker in speakers:
            output_path = 'en_pth_{}.wav'.format(str(speaker))
            start = time.perf_counter()
            model.tts_to_file(text, speaker_ids[speaker], output_path, speed=speed*0.75, use_ov = use_ov)
            end = time.perf_counter()
    else:
        for speaker in speakers:
            output_path = 'ov_en_int8_{}.wav'.format(speaker) if use_int8 else 'en_ov_{}.wav'.format(speaker)
            start = time.perf_counter()
            model.tts_to_file(text, speaker_ids[speaker], output_path, speed=speed, use_ov=use_ov)
            if speech_enhance:
                print("Use speech enhance")
                process_audio(output_path,output_path)
            end = time.perf_counter()         

    dur_time = (end - start) * 1000
    dur_time_list.append(dur_time)

if loop_num > 1:
    avg_lantecy = sum(dur_time_list[1:]) / (len(dur_time_list) - 1)
    print(f"MeloTTS model e2e avg latency: {avg_lantecy:.2f} ms")

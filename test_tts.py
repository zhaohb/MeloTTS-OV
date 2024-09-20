from melo.api import TTS
from pathlib import Path
import time

# Speed is adjustable
speed = 1.0
device = 'cpu' # or cuda:0

use_ov = True  ## Used to control whether to use torch or openvino
use_int8 = True
speech_enhance = True
lang = "EN" # or ZH

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

#text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
text = "I've been learning machine learning recently and hope to make contributions in the field of artificial intelligence in the future."

model = TTS(language=lang, device=device, use_int8=use_int8)
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
            model.tts_to_file(text, speaker_ids[speaker], output_path, speed=speed, use_ov = use_ov)
            end = time.perf_counter()
    else:
        for speaker in speakers:
            output_path = 'en_ov_{}.wav'.format(str(speaker))
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

import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch

from . import utils
from . import commons
from .models import SynthesizerTrn
from .split_utils import split_sentence
from .mel_processing import spectrogram_torch, spectrogram_torch_conv
from .download_utils import load_or_download_config, load_or_download_model
import openvino as ov
from pathlib import Path

class TTS(nn.Module):
    def __init__(self, 
                language,
                device='auto',
                use_hf=True,
                config_path=None,
                ckpt_path=None):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path = 
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts
    
    def tts_convert_to_ov(self, ov_path, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0,):
        ov_model_path = Path(f"{ov_path}/tts.xml")

        x_tst = torch.tensor([[  0,   0,   0,  97,   0,  65,   0, 100,   0,  89,   0,  55,   0,  49,
           0, 100,   0,  13,   0,  98,   0,  95,   0,  98,   0,  40,   0,  60,
           0,  12,   0,  77,   0,  54,   0,  62,   0,  59,   0,  32,   0,  62,
           0,  48,   0,  63,   0, 106,   0,   0,   0]])
        x_tst_lengths = torch.tensor([51])
        speakers = torch.tensor([1])
        tones = torch.tensor([[0, 0, 0, 3, 0, 3, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 2, 0, 2, 0, 2,
         0, 2, 0, 7, 0, 8, 0, 7, 0, 9, 0, 7, 0, 7, 0, 9, 0, 7, 0, 8, 0, 7, 0, 0,
         0, 0, 0]])
        lang_ids = torch.tensor([[0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
         0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
         0, 3, 0]])
        bert = torch.ones(1, 1024, 51) * 0
        ja_bert = torch.rand(( 1, 768, 51), dtype=torch.float32)
        noise_scale = torch.tensor([noise_scale])
        length_scale = torch.tensor([1. / speed])
        noise_scale_w = torch.tensor([noise_scale_w])
        sdp_ratio = torch.tensor([sdp_ratio])

        ov_model = ov.convert_model(
            self.model,
            example_input={
                "x": x_tst,
                "x_lengths": x_tst_lengths,
                "sid": speakers,
                "tone": tones,
                "language": lang_ids,
                "bert": bert,
                "ja_bert": ja_bert,
                "noise_scale": noise_scale,
                "length_scale": length_scale,
                "noise_scale_w": noise_scale_w,
                "sdp_ratio": sdp_ratio,
            },
        )
        outputs_name = ['audio']
        for output, output_name in zip(ov_model.outputs, outputs_name):
            output.get_tensor().set_names({output_name})
        ov.save_model(ov_model, Path(ov_model_path))

    def ov_model_init(self, ov_path=None):
        self.core = ov.Core()
        ov_model_path = Path(f"{ov_path}/tts.xml")

        self.tts_model = self.core.read_model(Path(ov_model_path))
        self.tts_compiled_model = self.core.compile_model(self.tts_model, 'CPU')
        self.tts_request = self.tts_compiled_model.create_infer_request()

    def ov_infer(self, x_tst=None, x_tst_lengths=None, speakers=None, tones=None, lang_ids=None, bert=None, ja_bert=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0):
            inputs_dict = {}
            inputs_dict['x'] = x_tst
            inputs_dict['x_lengths'] = x_tst_lengths
            inputs_dict['sid'] = speakers
            inputs_dict['tone'] = tones
            inputs_dict['language'] = lang_ids
            inputs_dict['bert'] = bert
            inputs_dict['ja_bert'] = ja_bert
            inputs_dict['noise_scale'] = torch.tensor([noise_scale])
            inputs_dict['length_scale'] = torch.tensor([1. / speed])
            inputs_dict['noise_scale_w'] = torch.tensor([noise_scale_w])
            inputs_dict['sdp_ratio'] = torch.tensor([sdp_ratio])
            
            self.tts_request.start_async(inputs_dict, share_inputs=True)
            self.tts_request.wait()
            audio = (self.tts_request.get_tensor("audio").data)[0][0]

            return audio

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False, use_ov=False):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)

                if use_ov:
                    audio = self.ov_infer(x_tst=x_tst, 
                                          x_tst_lengths=x_tst_lengths, 
                                          speakers=speakers, 
                                          tones=tones, 
                                          lang_ids=lang_ids, 
                                          bert=bert, 
                                          ja_bert=ja_bert, 
                                          sdp_ratio=sdp_ratio, 
                                          noise_scale=noise_scale, 
                                          noise_scale_w=noise_scale_w, 
                                          speed=1.0)
                else:
                    audio = self.model(
                            x_tst,
                            x_tst_lengths,
                            speakers,
                            tones,
                            lang_ids,
                            bert,
                            ja_bert,
                            sdp_ratio=sdp_ratio,
                            noise_scale=noise_scale,
                            noise_scale_w=noise_scale_w,
                            length_scale=1. / speed,
                        )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                # 
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)

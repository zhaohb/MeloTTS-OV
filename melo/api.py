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
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, PreTrainedModel
from transformers.onnx import FeaturesManager
import transformers
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import nncf

class ExportModel(PreTrainedModel):
    def __init__(self, base_model, config):
        super().__init__(config)
        self.model = base_model

    def forward(self, input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):

        out = self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        return {
            # "logits": out["logits"],
            # "hidden_states": torch.stack(list(out["hidden_states"]))
            "hidden_states": torch.cat(out["hidden_states"][-3:-2], -1)[0]
        }

class Bert():
    def __init__(self, use_int8=False):
        self.use_int8=use_int8
    
    def save_tokenizer(self, tokenizer, out_dir):
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception as e:
            log.error(f'tokenizer loading failed with {e}')

    def prepare_calibration_data(self, dataloader, init_steps):
        data = []
        for batch in dataloader:
            if len(data) == init_steps:
                break
            if batch is not None:
                with torch.no_grad():
                    inputs_dict = {}
                    inputs_dict['input_ids'] = batch['input_ids'].squeeze(0)
                    inputs_dict['token_type_ids'] = batch['token_type_ids'].squeeze(0)
                    inputs_dict['attention_mask'] = batch['attention_mask'].squeeze(0)
                    data.append(inputs_dict)
        return data

    def prepare_dataset(self, example_input=None, opt_init_steps=1, max_train_samples=1000):
        class CustomDataset(Dataset):
            def __init__(self, data_count=100, dummy_data=None):
                self.dataset = []
                for i in range(data_count):
                    self.dataset.append(dummy_data)
            def __len__(self):
                return len(self.dataset)

            def __getitem__(self,idx):
                data = self.dataset[idx]
                return data
        """
        Prepares a vision-text dataset for quantization.
        """
        dataset = CustomDataset(data_count=1, dummy_data=example_input)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)
        calibration_data = self.prepare_calibration_data(dataloader, opt_init_steps)
        return calibration_data

    def bert_convert_to_ov(self, ov_path, language = "ZH"):
        if "ZH" in language:
            model_id='bert-base-multilingual-uncased'
            text = "当需要把翻译对象表示, 可以使用这个方法。"
        elif "EN" in language:
            model_id='bert-base-uncased'
            text = "A buffer is a container for data that can be accessed from a device and the host."
        models = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizers = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        
        export_model = ExportModel(models, config)
        

        inputs = tokenizers(text, return_tensors="pt")

        example_input = {
                "input_ids": inputs['input_ids'],
                "token_type_ids": inputs['token_type_ids'],
                "attention_mask": inputs['attention_mask'],
            }
            
        ov_model = ov.convert_model(
            export_model,
            example_input = example_input,
        )
        
        get_input_names = lambda: ["input_ids", "token_type_ids", "attention_mask"]
        for input, input_name in zip(ov_model.inputs, get_input_names()):
            input.get_tensor().set_names({input_name})
        outputs_name = ['hidden_states']
        for output, output_name in zip(ov_model.outputs, outputs_name):
            output.get_tensor().set_names({output_name})
        
        """
        reshape model
        Set the batch size of all input tensors to 1 to facilitate the use of the C++ infer
        If you are only using the Python pipeline, this step can be omitted.
        """   
        shapes = {}     
        for input_layer  in ov_model.inputs:
            shapes[input_layer] = input_layer.partial_shape
            shapes[input_layer][0] = 1
        ov_model.reshape(shapes)

        self.save_tokenizer(tokenizers, Path(ov_path))
        models.config.save_pretrained(Path(ov_path))
        
        ov_model_path = Path(f"{ov_path}/bert_{language}.xml")
        ov.save_model(ov_model, Path(ov_model_path))
        
        if self.use_int8:
            calibration_data = self.prepare_dataset(example_input=example_input)
            calibration_dataset = nncf.Dataset(calibration_data)
            # quantized_model = nncf.quantize(
            #     model=ov_model,
            #     calibration_dataset=calibration_dataset,
            #     preset=nncf.QuantizationPreset.MIXED,
            #     # subset_size=len(calibration_data),
            #     )
            quantized_model = nncf.quantize(
                model=ov_model,
                calibration_dataset=calibration_dataset,
                model_type=nncf.ModelType.TRANSFORMER,
                subset_size=len(calibration_data),
                # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
                advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.6)
            )

            ov.save_model(quantized_model, Path(f"{ov_path}/bert_{language}.xml"))
        
    def ov_bert_model_init(self, ov_path=None, language = "ZH"):
        core = ov.Core()
        #if self.use_int8:
        #    ov_model_path = Path(f"{ov_path}/bert_{language}_int8.xml")
        #else:
        ov_model_path = Path(f"{ov_path}/bert_{language}.xml")
        self.bert_model = core.read_model(Path(ov_model_path))
        self.bert_compiled_model = core.compile_model(self.bert_model, 'CPU')
        self.bert_request = self.bert_compiled_model.create_infer_request()
                
        self.bert_tokenizer = AutoTokenizer.from_pretrained(ov_path, trust_remote_code=True)
        self.bert_config = AutoConfig.from_pretrained(ov_path, trust_remote_code=True)
        
    def ov_bert_infer(self, input_ids=None, token_type_ids=None, attention_mask=None):
        inputs_dict = {}
        inputs_dict['input_ids'] = input_ids
        inputs_dict['token_type_ids'] = token_type_ids
        inputs_dict['attention_mask'] = attention_mask
        
        self.bert_request.start_async(inputs_dict, share_inputs=True)
        self.bert_request.wait()
        bert_output = (self.bert_request.get_tensor("hidden_states").data.copy())

        return bert_output
class TTS(nn.Module):
    def __init__(self, 
                language,
                device='auto',
                use_hf=True,
                use_int8=False,
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
        
        self.bert_model = Bert(use_int8=use_int8)
        self.use_int8 =use_int8

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
    
    def prepare_calibration_data(self, dataloader, init_steps):
        data = []
        for batch in dataloader:
            if len(data) == init_steps:
                break
            if batch is not None:
                with torch.no_grad():
                    inputs_dict = {}
                    inputs_dict['phones'] = batch['x'].squeeze(0)
                    inputs_dict['phones_length'] = batch['x_lengths'].squeeze(0)
                    
                    inputs_dict['tones'] = batch['tone'].squeeze(0)
                    inputs_dict['lang_ids'] = batch['language'].squeeze(0)
                    inputs_dict['bert'] = batch['bert'].squeeze(0)
                    inputs_dict['ja_bert'] = batch['ja_bert'].squeeze(0)
                    speakers =1
                    sdp_ratio=0.2
                    noise_scale=0.6
                    noise_scale_w=0.8
                    speed=1.0
                    inputs_dict['speakers'] = torch.tensor([speakers])
                    inputs_dict['noise_scale'] = torch.tensor([noise_scale])
                    inputs_dict['length_scale'] = torch.tensor([1. / speed])
                    inputs_dict['noise_scale_w'] = torch.tensor([noise_scale_w])
                    inputs_dict['sdp_ratio'] = torch.tensor([sdp_ratio])
                    data.append(inputs_dict)
        return data

    def prepare_dataset(self, example_input=None, opt_init_steps=1, max_train_samples=1000):
        class CustomDataset(Dataset):
            def __init__(self, data_count=100, dummy_data=None):
                self.dataset = []
                for i in range(data_count):
                    self.dataset.append(dummy_data)
            def __len__(self):
                return len(self.dataset)

            def __getitem__(self,idx):
                data = self.dataset[idx]
                return data
        """
        Prepares a vision-text dataset for quantization.
        """
        dataset = CustomDataset(data_count=1, dummy_data=example_input)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)
        calibration_data = self.prepare_calibration_data(dataloader, opt_init_steps)
        return calibration_data
    
    def tts_convert_to_ov(self, ov_path, language = "ZH", sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0,):
        self.bert_model.bert_convert_to_ov(ov_path, language)
        
        ov_model_path = Path(f"{ov_path}/tts_{language}.xml")

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
            }
            
        ov_model = ov.convert_model(
            self.model,
            example_input=example_input,
        )
        get_input_names = lambda: ["phones", "phones_length", "speakers",
                                  "tones", "lang_ids", "bert", "ja_bert",
                                  "noise_scale", "length_scale", "noise_scale_w", "sdp_ratio"]
        for input, input_name in zip(ov_model.inputs, get_input_names()):
            input.get_tensor().set_names({input_name})
        outputs_name = ['audio']
        for output, output_name in zip(ov_model.outputs, outputs_name):
            output.get_tensor().set_names({output_name})
        
        """
        reshape model
        Set the batch size of all input tensors to 1 to facilitate the use of the C++ infer
        If you are only using the Python pipeline, this step can be omitted.
        """   
        shapes = {}     
        for input_layer  in ov_model.inputs:
            shapes[input_layer] = input_layer.partial_shape
            shapes[input_layer][0] = 1
        ov_model.reshape(shapes)

        ov.save_model(ov_model, Path(ov_model_path))
        
        if self.use_int8:
            calibration_data = self.prepare_dataset(example_input=example_input)
            calibration_dataset = nncf.Dataset(calibration_data)
            # quantized_model = nncf.quantize(
            #     model=ov_model,
            #     calibration_dataset=calibration_dataset,
            #     preset=nncf.QuantizationPreset.MIXED,
            #     # subset_size=len(calibration_data),
            #     )
            quantized_model = nncf.quantize(
                model=ov_model,
                calibration_dataset=calibration_dataset,
                model_type=nncf.ModelType.TRANSFORMER,
                subset_size=len(calibration_data),
                # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
                advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.6)
            )

            ov.save_model(quantized_model, Path(f"{ov_path}/tts_{language}_int8.xml"))

    def ov_model_init(self, ov_path=None, language = "ZH"):
        self.bert_model.ov_bert_model_init(ov_path, language=language)
        
        self.core = ov.Core()
        if self.use_int8:
            ov_model_path = Path(f"{ov_path}/tts_int8_{language}.xml")
        else:
            ov_model_path = Path(f"{ov_path}/tts_{language}.xml")
        print(f"ov_path : {ov_model_path}")
        self.tts_model = self.core.read_model(Path(ov_model_path))
        self.tts_compiled_model = self.core.compile_model(self.tts_model, 'CPU')
        self.tts_request = self.tts_compiled_model.create_infer_request()

    def ov_infer(self, x_tst=None, x_tst_lengths=None, speakers=None, tones=None, lang_ids=None, bert=None, ja_bert=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0):
            inputs_dict = {}
            inputs_dict['phones'] = x_tst
            inputs_dict['phones_length'] = x_tst_lengths
            inputs_dict['speakers'] = speakers
            inputs_dict['tones'] = tones
            inputs_dict['lang_ids'] = lang_ids
            inputs_dict['bert'] = bert
            inputs_dict['ja_bert'] = ja_bert
            inputs_dict['noise_scale'] = torch.tensor([noise_scale])
            inputs_dict['length_scale'] = torch.tensor([1. / speed])
            inputs_dict['noise_scale_w'] = torch.tensor([noise_scale_w])
            inputs_dict['sdp_ratio'] = torch.tensor([sdp_ratio])
            
            self.tts_request.start_async(inputs_dict, share_inputs=True)
            self.tts_request.wait()
            audio = (self.tts_request.get_tensor("audio").data.copy())[0][0]

            return audio

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False, use_ov=True):
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
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, symbol_to_id=self.symbol_to_id, bert_model=self.bert_model, use_ov=use_ov)
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

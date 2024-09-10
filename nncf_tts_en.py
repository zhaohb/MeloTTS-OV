from melo.api import TTS
from pathlib import Path
import time
import openvino as ov
import nncf
import string
from torch.utils.data import Dataset, DataLoader
from melo import utils
import torch
import re
from transformers import AutoTokenizer
from datasets import load_dataset
import random
# Speed is adjustable
speed = 1.0
device = 'cpu' 

lang = "EN"# ZH for Chinese
ov_path = f"/home/qiu/TTS_OV/MeloTTS-OV/tts_ov_{lang}"
model = TTS(language=lang, device=device, use_hf= False ,use_int8= False)
model.ov_model_init(ov_path,language= lang)



data_count = 512

#file_path = "/home/qiu/TTS_OV/MeloTTS-OV/test/basetts_test_resources/zh_mix_en_egs_text.txt"
#file_path = "/home/qiu/TTS_OV/MeloTTS-OV/cleaned_train.txt"
ov_model_path = Path(f"{ov_path}/tts_{lang}.xml")
global speaker_ids 
speaker_ids = model.hps.data.spk2id  # dict

def clean_lines_from_dataset(data_count=300):
    """
    Loads the Yelp dataset, removes punctuation from each line, and returns a list of cleaned lines.
    
    :param data_count: The number of lines to return (default is 300).
    :return: List of strings with punctuation removed.
    """
    # Load the dataset and select the first 'data_count' examples
    dataset = load_dataset('yelp_polarity', split='train')
    small_sample = dataset.select(range(data_count))
    
    # Define a translation table that maps punctuation to None
    translator = str.maketrans('', '', string.punctuation)
    
    cleaned_lines = []
    for item in small_sample:
        line = item['text']  # Access the text field from each sample
        # Remove punctuation and strip leading/trailing whitespace
        cleaned_line = line.translate(translator).strip()
        # Separate camel-cased words if any (like "goodService" -> "good Service")
        cleaned_line = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_line)
        cleaned_lines.append(cleaned_line)
    
    return cleaned_lines
cleaned_data = clean_lines_from_dataset(10)
#for line in cleaned_data[:5]:  # Print the first 5 cleaned lines
#    print(line)


def get_text(filename) -> list:
    """
    Reads a file, removes punctuation from each line, and returns a list of cleaned lines.
    
    :param filename: Path to the file to be read.
    :return: List of strings with punctuation removed.
    """
    # Define a translation table that maps punctuation to None
    translator = str.maketrans('', '', string.punctuation)
    
    lines = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Remove punctuation and strip leading/trailing whitespace
            cleaned_line = line.translate(translator).strip()
            cleaned_line = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_line)
            lines.append(cleaned_line[:30])
    
    return lines[:data_count]


def transform_fn(data_item):
    print(data_item[0])
    data_item = data_item[0]
    bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(data_item, "EN", model.hps, device, symbol_to_id=model.symbol_to_id, bert_model=model.bert_model, use_ov=True)
    
    x_tst = phones.to(device).unsqueeze(0)
    tones = tones.to(device).unsqueeze(0)
    lang_ids = lang_ids.to(device).unsqueeze(0)
    bert = bert.to(device).unsqueeze(0)
    ja_bert = ja_bert.to(device).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)

    #speaker_id = speaker_ids['ZH'] # ZH_MIX_EN
    #print(random.choice(speaker_ids))
    #breakpoint()
    speakers = torch.LongTensor([0]).to(device)

    sdp_ratio=0.2
    noise_scale=0.6
    noise_scale_w=0.8
    speed=1.0
    
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
    return inputs_dict

encoder_calibration_data = clean_lines_from_dataset(10)#get_text(file_path)
val_data_loader = DataLoader(encoder_calibration_data, batch_size=1, shuffle=False)


#calibration_data = prepare_dataset(example_input=example_input)
#calibration_dataset = nncf.Dataset(calibration_data)

calibration_dataset = nncf.Dataset(val_data_loader, transform_fn)
#model.bert_model.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
ov_model = ov.Core().read_model(ov_model_path)
quantized_model = nncf.quantize(
                model=ov_model,
                calibration_dataset=calibration_dataset,
                model_type=nncf.ModelType.TRANSFORMER,
                #subset_size=len(calibration_data),
                # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
                preset=nncf.QuantizationPreset.MIXED,
                #advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.6)
                advanced_parameters=nncf.AdvancedQuantizationParameters(
                    #disable_channel_alignment=False,
                    smooth_quant_alpha=0.8
                )
)

ov.save_model(quantized_model, Path(f"{ov_path}/tts_int8_{lang}.xml"))

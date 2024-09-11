from melo.api import TTS
from pathlib import Path
from openvino.runtime import Core
from openvino import Type
import openvino as ov
import time
from transformers import AutoTokenizer
import torch

device = 'CPU' # or cuda:0
language = 'ZH'
ov_path = f"/tts_ov_{language}"
ov_model_path = Path(f"{ov_path}/bert_int8.xml")
ov_model_save_path = Path(f"{ov_path}/bert_static_int8.xml")
bert_static_shape = [1,64]
def reshape_for_npu(model, bert_static_shape):
        # change dynamic shape to static shape
        shapes = dict()
        for input_layer  in model.inputs:
            shapes[input_layer] = bert_static_shape
        model.reshape(shapes)
        ov.save_model(model, Path(ov_model_save_path))
        print(f"save static model in {Path(ov_model_save_path)}")


def pad_input(input_dict, pad_length=64):
    def pad_tensor(input_tensor, pad_length):
        pad_size = pad_length - input_tensor.shape[1]
        if pad_size > 0:
            return torch.nn.functional.pad(input_tensor, (0, pad_size), 'constant', 0)
        else:
            return input_tensor
    
    padded_inputs = {}
    for key, value in input_dict.items():
        padded_inputs[key] = pad_tensor(value, pad_length)
    return padded_inputs

def test_static_shape(compiled_model,device="NPU"):
    if "ZH" in language:
            model_id='bert-base-multilingual-uncased'
            text = "buffer是一个数据容器，可以从device和host访问。"
    elif "EN" in language:
            model_id='bert-base-uncased'
            text = "A buffer is a container for data that can be accessed from a device and the host."
    tokenizers = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizers(text, return_tensors="pt")
    padded_inputs = pad_input(inputs,pad_length=64)
   
    infer_request = compiled_model.create_infer_request()

    infer_request.infer(padded_inputs)
    res =  infer_request.get_tensor("hidden_states").data.copy()
    #print(res)
    pass


def main():
    core = Core()
    model = core.read_model(ov_model_path)
    reshape_for_npu(model, bert_static_shape=bert_static_shape)
    compiled_model = core.compile_model(ov_model_save_path,device)
    test_static_shape(compiled_model, "NPU")
    
    
if __name__ == "__main__":
    main()




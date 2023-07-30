import numpy as np

from DataModels.DataType import DataType
from DataModels.ModelMetadata import ModelMetadata
from DataModels.TextModel import TextModel
from datasets import load_dataset
import copy
import torch
from tqdm import tqdm
import torch.nn.functional as F

def play_with_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')

    bert_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                  align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
    roberta_metadata = ModelMetadata(model_name="roberta-base", data_type=DataType.Text,
                                     align_tokens_to_bert_tokens=False, use_cls_and_sep=True)

    bert_model = TextModel(bert_metadata, device=device)
    roberta_model = TextModel(roberta_metadata, device=device)

    text = dataset[1]["text"]

    input_ids = bert_model.tokenizer.encode(text, add_special_tokens=bert_model.model_metadata.use_cls_and_sep,
                                            return_tensors='pt')

    # inputs = tokenizer.encode_plus(text,  return_tensors="pt", add_special_tokens = True, truncation=True, pad_to_max_length = True,
    #                                          return_attention_mask = True,  max_length=64)
    if bert_model.model_metadata.use_cls_and_sep:
        rand_ind = torch.randint(1, input_ids.shape[1] - 1, (1,))
    else:
        rand_ind = torch.randint(0, input_ids.shape[1], (1,))

    print(input_ids)
    labels = copy.deepcopy(input_ids)
    input_ids[0][rand_ind] = bert_model.tokenizer.mask_token_id
    labels[input_ids != bert_model.tokenizer.mask_token_id] = -100  # This is ignored index to Bert and Roberta

    res = bert_model.model(
        input_ids=input_ids,
        labels=labels,
        # attention_mask=inputs["attention_mask"],
        # token_type_ids=inputs["token_type_ids"]
    )
    print('loss', res['loss'])

    pred = torch.argmax(res['logits'][0][rand_ind]).item()
    print("predicted token:", pred, bert_model.tokenizer.convert_ids_to_tokens([pred]))
    print("right token:", labels[0][rand_ind], bert_model.tokenizer.convert_ids_to_tokens([labels[0][rand_ind]]))

    # calculate loss manually
    logSoftmax = torch.nn.LogSoftmax(dim=1)
    NLLLos = torch.nn.NLLLoss()
    print(NLLLos(logSoftmax(res['logits'][0][rand_ind]), torch.tensor([pred])))


@torch.inference_mode()
def run_stats_loss(model_metadata, dataset):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model = TextModel(model_metadata, device=device)

    all_losses = []
    all_accuracy = []
    for sample in tqdm(dataset):
        text = sample['text']
        input_ids = text_model.tokenizer.encode(text, add_special_tokens=text_model.model_metadata.use_cls_and_sep,
                                                return_tensors='pt')
        if text_model.model_metadata.use_cls_and_sep:
            rand_ind = torch.randint(1, input_ids.shape[1] - 1, (1,))
        else:
            rand_ind = torch.randint(0, input_ids.shape[1], (1,))

        labels = copy.deepcopy(input_ids)
        input_ids[0][rand_ind] = text_model.tokenizer.mask_token_id
        labels[input_ids != text_model.tokenizer.mask_token_id] = -100  # This is ignored index to Bert and Roberta

        res = text_model.model(
            input_ids=input_ids,
            labels=labels,
        )
        all_losses.append(res['loss'].item())
        pred = torch.argmax(res['logits'][0][rand_ind]).item()
        all_accuracy.append(pred == labels[0][rand_ind].item())


    all_losses = np.array(all_losses)
    all_accuracy = np.array(all_accuracy)
    print("Data info:")
    print(f"    Number of Sampels: {len(dataset)}")
    print("Model info:")
    print(f"    Name: {model_metadata.model_name}")
    print(f"    Add special tokens: {model_metadata.use_cls_and_sep}")
    print("Results:")
    print(f"    Loss (mean): {np.mean(all_losses)}")
    print(f"    Loss (std): {np.std(all_losses)}")
    print(f"    Accuracy: {np.sum(all_accuracy) / len(all_accuracy)}")


if __name__ == '__main__':
    bert_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                  align_tokens_to_bert_tokens=False, use_cls_and_sep=False)
    roberta_metadata = ModelMetadata(model_name="roberta-base", data_type=DataType.Text,
                                     align_tokens_to_bert_tokens=False, use_cls_and_sep=True)

    # dataset_name = 'albertvillanova/universal_dependencies'
    # dataset_split = 'en_ewt'
    # dataset = load_dataset(dataset_name, dataset_split, split='validation')
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')
    model_names = ["bert-base-uncased", "roberta-base"]
    uses_cls_and_sep = [True, False]

    for model_name in model_names:
        for use_cls_and_sep in uses_cls_and_sep:
            model_metadata = ModelMetadata(model_name=model_name, data_type=DataType.Text, align_tokens_to_bert_tokens=False, use_cls_and_sep=use_cls_and_sep)
            run_stats_loss(model_metadata, dataset)
import os
import torchaudio
from tqdm import tqdm
import pandas as pd
from DataModels.Sample import Sample
from AttentionExtractors.ExtractorFactory import ExtractorFactory, ModelMetadata, DataType
from universal_dependencies import get_dataset, clean_dataset


dataset = clean_dataset(get_dataset())


if __name__ == '__main__':

  model_metadata = ModelMetadata(model_name="facebook/wav2vec2-base-960h", data_type=DataType.Audio,
                                 align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
  extractor = ExtractorFactory.create_attention_extractor(model_metadata, )
  gt_path = '/home/vpnuser/cs_huji/anlp/SoundOfAttention/DependencySyntax/data_for_asr_v2.csv'
  base_path = '/home/vpnuser/cs_huji/anlp/tts_data/ud_gum'
  attn_path = "/home/vpnuser/cs_huji/anlp/ud_w2v2_attn"
  agg_attn_path = '/home/vpnuser/cs_huji/anlp/ud_w2v2_attn_agg_tokens'
  with open(gt_path, 'r') as f:
    for i, line in enumerate(tqdm(f)):
      try:
        idx, text = line.split("\t")
        assert text.strip() == dataset[i]["text"]
        tokens = " ".join([t.upper() for t in dataset[i]["tokens"]])
        fname = f"{i}_{idx}.wav"
        attn_fname = os.path.join(attn_path, f"{fname.split('.')[0]}.pkl")
        agg_attn_fname = os.path.join(agg_attn_path, f"{fname.split('.')[0]}.pkl")
        audio_fname = os.path.join(base_path, fname)
        assert os.path.exists(attn_fname)
        assert os.path.exists(audio_fname)

        audio, fs = torchaudio.load(audio_fname)
        assert fs == 16000
        sample = Sample(id=fname.split(".")[0], audio=dict(array=audio[0], fs=fs,
                                                           sampling_rate=fs),
                        text=tokens)
        attentions = pd.read_pickle(attn_fname)
        print("Aligning attentions")
        agg_attentions = extractor.align_attentions(sample, attentions, True)
        pd.to_pickle(agg_attentions, agg_attn_fname)
      except Exception as e:
        print(f"Exception {i}, {e}")
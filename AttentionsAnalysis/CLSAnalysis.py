import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from Common.Constants import Constants
from DataModels.DataType import DataType
from DataModels.ModelMetadata import ModelMetadata
from DataModels.Sample import Sample
from AttentionsAnalysis.AnalysisGenerator import AnalysisGenerator
from scipy.spatial.distance import cosine


DEFAULT_AUDIO_KEY = Constants.AudioModelProcessorConstants.LIBRISPEECH_AUDIO_KEY
DISPLAY = True


if __name__ == '__main__':

    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')

    model2_metadata = ModelMetadata(model_name="facebook/wav2vec2-base-960h", data_type=DataType.Audio,
                                    align_to_text_tokens=True, use_cls_and_sep=True)
    model1_metadata = ModelMetadata(model_name="facebook/wav2vec2-base-960h", data_type=DataType.Audio,
                                    align_to_text_tokens=True, use_cls_and_sep=False)

    analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric='Cosine')
    means_ones_twos = []
    medians_ones_twos = []
    means_threes_fours = []
    medians_threes_fours = []
    means_lasts = []
    medians_lasts = []
    num_sampels = 0
    for data in tqdm(dataset):
        sample = Sample(id=data["id"], text=data["text"], audio=data["audio"])
        # attention_model1 = analysis_generator.extractor1.extract_attention(sample=sample)
        attention_model2 = analysis_generator.extractor2.extract_attention(sample=sample)
        if attention_model2.shape[-1] < 6:
            continue
        num_sampels += 1
        ones = attention_model2.attentions[:,:,:,0].reshape(-1, attention_model2.shape[-1])
        twos = attention_model2.attentions[:,:,:,1].reshape(-1, attention_model2.shape[-1])
        threes = attention_model2.attentions[:,:,:,2].reshape(-1, attention_model2.shape[-1])
        fours = attention_model2.attentions[:,:,:,3].reshape(-1, attention_model2.shape[-1])
        last = attention_model2.attentions[:,:,:,-1].reshape(-1, attention_model2.shape[-1])
        one_to_last = attention_model2.attentions[:,:,:,-2].reshape(-1, attention_model2.shape[-1])

        ones_twos = np.array([1 - cosine(one, two) for one, two in zip(ones, twos)])
        threes_fours = np.array([1 - cosine(three, four) for three, four in zip(threes, fours)])
        lasts = np.array([1 - cosine(three, four) for three, four in zip(one_to_last, last)])

        means_ones_twos.append(np.mean(ones_twos))
        medians_ones_twos.append(np.median(ones_twos))
        means_threes_fours.append(np.mean(threes_fours))
        medians_threes_fours.append(np.median(threes_fours))
        means_lasts.append(np.mean(lasts))
        medians_lasts.append(np.median(lasts))

    t=1
    # for data in dataset:
    #     sample = Sample(id=data["id"], text=data["text"], audio=data["audio"])
    #     attention_model1 = analysis_generator.extractor1.extract_attention(sample=sample)
    #     attention_model2 = analysis_generator.extractor2.extract_attention(sample=sample)

    # correlations_attentions_comparisons = analysis_generator.get_correlations_of_attentions(attention_model1,
    #                                                                                         attention_model2)
    # VisualizerAttentionsResults.plot_correlation_of_attentions(sample=sample1,
    #                                                            model_name1=model1_metadata.model_name,
    #                                                            model_name2=model2_metadata.model_name,
    #                                                            correlations_attentions_comparisons=correlations_attentions_comparisons)
    # VisualizerAttentionsResults.plot_histogram_of_layers_and_heads(correlations_attentions_comparisons)
    # End

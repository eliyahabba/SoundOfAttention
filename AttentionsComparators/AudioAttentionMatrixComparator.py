from AttentionExtractor.AudioAttentionExtractor import AudioAttentionExtractor
from AttentionsComparators.AttentionsComparator import AttentionsComparator


class AudioTextAttentionMatrixComparator(AttentionsComparator):
    def __init__(self, audio_model_name1: str, audio_model_name2: str):
        # Load the BERT model and tokenizer
        self.audio_model_name1 = audio_model_name1
        self.audio_model_name2 = audio_model_name2
        self.audio_attention_extractor_model1 = AudioAttentionExtractor(audio_model_name1)
        self.audio_attention_extractor_model2 = AudioAttentionExtractor(audio_model_name2)

    def create_attention_matrices(self, audio, attention_layer, head):
        model1_attention = self.audio_attention_extractor_model1.extract_attention(audio, attention_layer, head)
        model2_attention = self.audio_attention_extractor_model2.extract_attention(audio, attention_layer, head)
        return model1_attention, model2_attention


if __name__ == '__main__':
    pass

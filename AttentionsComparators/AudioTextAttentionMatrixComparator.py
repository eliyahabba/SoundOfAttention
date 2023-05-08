from AttentionExtractor.AudioAttentionExtractor import AudioAttentionExtractor
from AttentionExtractor.TextAttentionExtractor import TextAttentionExtractor
from AttentionsComparators.AttentionsComparator import AttentionsComparator


class AudioTextAttentionMatrixComparator(AttentionsComparator):
    def __init__(self, text_model_name: str, audio_model_name: str):
        self.text_model_name = text_model_name
        self.audio_model_name = audio_model_name
        self.text_attention_extractor_model = TextAttentionExtractor(self.text_model_name)
        self.audio_attention_extractor_model = AudioAttentionExtractor(self.audio_model_name)

    def create_attention_matrices(self, audio, text, attention_layer, head):
        model1_attention = self.text_attention_extractor_model.extract_attention(text, attention_layer, head)
        model2_attention = self.audio_attention_extractor_model.extract_attention(audio, attention_layer, head)
        return model1_attention, model2_attention


if __name__ == '__main__':
    pass

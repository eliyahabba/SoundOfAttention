from AttentionComparators.AttentionComparator import AttentionComparator
from AttentionExtractor.AudioAttentionExtractor import AudioAttentionExtractor


class AudioTextAttentionMatrixComparator(AttentionComparator):
    def __init__(self, model_name1: str, model_name2: str):
        # Load the BERT model and tokenizer
        self.model_name1 = model_name1
        self.model_name2 = model_name2
        self.audio_attention_extractor_model1 = AudioAttentionExtractor(model_name1)
        self.audio_attention_extractor_model2 = AudioAttentionExtractor(model_name2)

    def create_attention_matrices(self, text, head):
        model1_attention = self.audio_attention_extractor_model1.extract_attention(text, head)
        model2_attention = self.audio_attention_extractor_model2.extract_attention(text, head)
        return model1_attention, model2_attention


if __name__ == '__main__':
    pass

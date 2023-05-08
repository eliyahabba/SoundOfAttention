from AttentionComparators.AttentionComparator import AttentionComparator
from AttentionExtractor.TextAttentionExtractor import TextAttentionExtractor


class TextAttentionMatrixComparator(AttentionComparator):
    def __init__(self, model_name1: str, model_name2: str):
        # Load the BERT model and tokenizer
        self.model_name1 = model_name1
        self.model_name2 = model_name2
        self.text_attention_extractor_model1 = TextAttentionExtractor(model_name1)
        self.text_attention_extractor_model2 = TextAttentionExtractor(model_name2)

    def create_attention_matrices(self, text, head):
        model1_attention = self.text_attention_extractor_model1.extract_attention(text, head)
        model2_attention = self.text_attention_extractor_model2.extract_attention(text, head)
        return model1_attention, model2_attention


if __name__ == '__main__':
    text_attention_matrix_comparator = TextAttentionMatrixComparator('bert-base-uncased', 'roberta-base')
    model1_attention_head, model2_attention_head = text_attention_matrix_comparator.create_attention_matrices(
        'Hello, my dog is cute', 0)
    assert model1_attention_head.shape == model2_attention_head.shape
    resuls = text_attention_matrix_comparator.compare_attention_matrices(model1_attention_head, model2_attention_head)

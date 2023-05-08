from AttentionExtractor.TextAttentionExtractor import TextAttentionExtractor
from AttentionsComparators.AttentionsComparator import AttentionsComparator
from Common.Constants import Constants

TextConstants = Constants.TextConstants
TextAttentionExtractorConstants = Constants.TextAttentionExtractorConstants


class TextAttentionMatrixComparator(AttentionsComparator):
    def __init__(self, text_model_name1: str, text_model_name2: str):
        # Load the BERT model and tokenizer
        self.text_model_name1 = text_model_name1
        self.text_model_name2 = text_model_name2
        self.text_attention_extractor_model1 = TextAttentionExtractor(text_model_name1)
        self.text_attention_extractor_model2 = TextAttentionExtractor(text_model_name2)

    def create_attention_matrices(self, text, attention_layer, head):
        model1_attention = self.text_attention_extractor_model1.extract_attention(text, attention_layer, head)
        model2_attention = self.text_attention_extractor_model2.extract_attention(text, attention_layer, head)
        return model1_attention, model2_attention


if __name__ == '__main__':
    text_attention_matrix_comparator = TextAttentionMatrixComparator(TextConstants.BERT_BASE,
                                                                     TextConstants.ROBERTA_BASE)
    model1_attention_head, model2_attention_head = text_attention_matrix_comparator.create_attention_matrices(
        'Hello, my dog is cute', attention_layer=TextAttentionExtractorConstants.DEFAULT_ATTENTION_LAYER,
        head=TextAttentionExtractorConstants.DEFAULT_ATTENTION_HEAD)
    assert model1_attention_head.shape == model2_attention_head.shape
    results = text_attention_matrix_comparator.compare_attention_matrices(model1_attention_head, model2_attention_head)

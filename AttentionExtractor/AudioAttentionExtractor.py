from AttentionExtractor.AttentionExtractor import AttentionExtractor


class AudioAttentionExtractor(AttentionExtractor):
    def __init__(self, model_name: str):
        # use super() to call the parent class constructor
        super().__init__(model_name)
        pass

    def run_model(self, text):
        pass

    def extract_attention(self, text, head):
        outputs = self.run_model(text)
        attentions = self.get_attention_matrix(outputs, head)
        return attentions

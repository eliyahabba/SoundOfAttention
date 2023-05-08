from abc import abstractmethod


class AttentionExtractor:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_attention_weights(self, model_outputs, attention_layer: int = 0):
        attentions = model_outputs.attentions[attention_layer]
        return attentions

    def get_first_batch_of_attentions_layers(self, attentions):
        # Extract the first batch (which is the only one) from the attention matrices for both models
        attentions_first_batch = attentions.squeeze(0)
        return attentions_first_batch

    def get_attention_head(self, attentions_of_one_example, head: int):
        return attentions_of_one_example[head]

    def get_attention_matrix(self, model_outputs, attention_layer, head):
        attentions = self.get_attention_weights(model_outputs, attention_layer=attention_layer)
        attentions_first_batch = self.get_first_batch_of_attentions_layers(attentions)
        attention_head = self.get_attention_head(attentions_first_batch, head)
        return attention_head

    @abstractmethod
    def extract_attention(self, sample, attention_layer, head):
        pass

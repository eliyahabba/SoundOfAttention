import torch
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel


class AttentionMatrixComparator:
    def __init__(self):
        # Load the BERT model and tokenizer
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load the RoBERTa model and tokenizer
        self.roberta_model = RobertaModel.from_pretrained('roberta-base', output_attentions=True)
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def run_models(self, text):
        # Tokenize the input text with both models
        bert_tokens = self.bert_tokenizer.tokenize(text)
        roberta_tokens = self.roberta_tokenizer.tokenize(text)

        # TODO: check if we need to add special tokens for BERT
        # Add special tokens for BERT
        # bert_tokens = ['[CLS]'] + bert_tokens + ['[SEP]']

        # Convert the tokens to input IDs for both models
        bert_input_ids = self.bert_tokenizer.convert_tokens_to_ids(bert_tokens)
        roberta_input_ids = self.roberta_tokenizer.convert_tokens_to_ids(roberta_tokens)

        # Convert the input IDs to PyTorch tensors
        bert_input_ids = torch.tensor(bert_input_ids).unsqueeze(0)
        roberta_input_ids = torch.tensor(roberta_input_ids).unsqueeze(0)

        # Get the outputs and attention matrices for both models
        bert_outputs = self.bert_model(bert_input_ids)
        roberta_outputs = self.roberta_model(roberta_input_ids)
        return bert_outputs, roberta_outputs

    def get_attention_weights(self, bert_outputs, roberta_outputs):
        bert_attentions = bert_outputs.attentions[0]
        roberta_attentions = roberta_outputs.attentions[0]

        # Return the attention matrices for both models
        return bert_attentions, roberta_attentions

    def get_first_batch(self, bert_attentions, roberta_attentions):
        # Extract the first batch (which is the only one) from the attention matrices for both models
        bert_first_batch = bert_attentions.squeeze(0)
        roberta_first_batch = roberta_attentions.squeeze(0)
        return bert_first_batch, roberta_first_batch

    def get_attention_head(self, bert_one_batch, roberta_one_batch, head: int):
        bert_head = bert_one_batch[head]
        roberta_head = roberta_one_batch[head]
        return bert_head, roberta_head

    def compare_attention_matrices(self, text, head):
        bert_outputs, roberta_outputs = self.run_models(text)
        bert_attentions, roberta_attentions = self.get_attention_weights(bert_outputs, roberta_outputs)
        bert_first_batch, roberta_first_batch = self.get_first_batch(bert_attentions, roberta_attentions)
        bert_head, roberta_head = self.get_attention_head(bert_first_batch, roberta_first_batch, head)
        return bert_head, roberta_head


if __name__ == '__main__':
    comparator = AttentionMatrixComparator()
    bert_head, roberta_head = comparator.compare_attention_matrices('Hello, my dog is cute', 0)
    assert bert_head.shape == roberta_head.shape

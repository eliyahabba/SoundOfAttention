class Constants:
    class AttentionsConstants:
        LAYER_AXIS = 0
        HEAD_AXIS = 1
        LAYER = 'layer'
        HEAD = 'head'
        CORRELATION = 'correlation'

    class AudioConstants:
        W2V2 = 'facebook/wav2vec2-base-960h'

    class TextConstants:
        BERT_BASE = 'bert-base-uncased'
        ROBERTA_BASE = 'roberta-base'

    class AlignmentConstants:
        FS = 16_000

    class AudioModelProcessorConstants:
        SAMPLING_RATE = 16_000
        LIBRISPEECH_AUDIO_KEY = 'array'

    class TextAttentionExtractorConstants:
        DEFAULT_ATTENTION_LAYER = 0
        DEFAULT_ATTENTION_HEAD = 0
        NUM_OF_ATTENTION_LAYERS = 12
        NUM_OF_ATTENTION_HEADS = 12

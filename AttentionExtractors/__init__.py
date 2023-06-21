import torch
from AttentionExtractors.AttentionExtractor import AttentionExtractor
from AttentionExtractors.AudioAttentionExtractor import AudioAttentionExtractor
from AttentionExtractors.TextAttentionExtractor import TextAttentionExtractor


def ExtractorFactory(type: str, model_name: str, device: torch.device = torch.device('cpu')) -> AttentionExtractor:
    if type == 'text':
        return TextAttentionExtractor(model_name, device)
    elif type == 'audio':
        return AudioAttentionExtractor(model_name, device)
    else:
        ValueError("Please provide 'type' that in ['audio', 'text']")

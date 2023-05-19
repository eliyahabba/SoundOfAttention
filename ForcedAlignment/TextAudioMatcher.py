from Common.Constants import Constants
from Common.Resources import Resources

AlignmentConstants = Constants.AlignmentConstants


class TextAudioMatcher:
    def __init__(self, resources: Resources):
        self.resources = resources
        self.tokenizer, self.dataset, self.aligner = self.resources.load()

    def match(self, text, audio):
        """
        Matches the given text to the given audio.

        :param text: The text to match.
        :param audio: The audio to match.
        :return: A list of matches, where each match is a dictionary with the following keys: 'text' (str), 'audio' (np.ndarray).
        """
        sample_text = text
        assert audio['sampling_rate'] == AlignmentConstants.FS
        assert "' " not in sample_text, f"skipping examples with '<space>' due to tokenizer issues"

        sample_tokens = self.tokenizer.tokenize(sample_text)
        assert self.tokenizer.convert_tokens_to_string(sample_tokens).replace(" ' ",
                                                                              "'") == sample_text.lower()

        segments = self.aligner(audio['array'], sample_text)
        tokens_mapping = self._create_tokens_mapping(sample_tokens, segments)

        matches = []
        for mapping in tokens_mapping:
            match = self._create_match(mapping, audio)
            matches.append(match)
        return matches

    def _create_tokens_mapping(self, sample_tokens, segments):
        """
        Creates a mapping between the tokens and the segments.
        :param sample_tokens: The tokens of the sample.
        :param segments: The segments of the sample.
        :return: A mapping between the tokens and the segments.
        """
        tokens_mapping = list()
        seg_idx = -1
        for token in sample_tokens:
            if not token.startswith("#"):
                seg_idx += 1
            stripped_token = token.replace("#", "")
            segments_for_token = segments[seg_idx: seg_idx + len(stripped_token)]
            assert stripped_token == "".join([s.label for s in segments_for_token]).lower()
            seg_idx += len(stripped_token)
            tokens_mapping.append(dict(token=token,
                                       stripped_token=stripped_token,
                                       start=segments_for_token[0].start,
                                       end=segments_for_token[-1].end,
                                       score=sum([s.score for s in segments_for_token]) / len(segments_for_token)))
        return tokens_mapping

    def _create_match(self, mapping, audio):
        match_audio = audio['array'][int(mapping['start'] * 16000 / 50): int(mapping['end'] * 16000 / 50)]
        return {
            'text': mapping['token'],
            'audio': match_audio,
            'sample_rate': AlignmentConstants.FS,
            'audio_start': mapping['start'],
            'audio_end':  mapping['end']
        }

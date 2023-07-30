import torch
from dataclasses import dataclass
from transformers import AutoConfig, AutoModelForCTC, Wav2Vec2Processor


class Wav2Vec2Aligner:
    def __init__(self, model_name, cuda: bool):
        self.cuda = cuda
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.eval()
        if self.cuda:
            self.model.to(device="cuda")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.blank_id = self.processor.tokenizer.pad_token_id

    def __call__(self, audio, gt_text):
        gt_text = gt_text.replace(" ", "|")
        gt_tokens = self.processor.tokenizer.tokenize(gt_text)
        gt_ids = self.processor.tokenizer.convert_tokens_to_ids(gt_tokens)
        assert "".join(self.processor.tokenizer.convert_ids_to_tokens(gt_ids)) == gt_text
        inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
        if self.cuda:
            inputs = inputs.to(device="cuda")
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_sentences = self.processor.batch_decode(predicted_ids)
            print(predicted_sentences)
        emissions = torch.log_softmax(logits, dim=-1)
        emission = emissions[0].cpu().detach()
        trellis = self.get_trellis(emission, gt_ids)

        path = self.backtrack(trellis, emission, gt_ids)
        print(path)
        segments = self.merge_repeats(path)
        return segments

    def get_trellis(self, emission, tokens, ):
        num_frame = emission.size(0)
        num_tokens = len(tokens)
        num_tokens_with_blanks = 2 * len(tokens) + 1
        # add blanks - from "apple" to "^a^p^p^l^e^"
        tokens_with_blank = [self.blank_id] * num_tokens_with_blanks
        tokens_with_blank[1::2] = tokens
        # mark indices that can be skipped - every blank index that is not between multiple char.
        # In "apple" for example, the blank between the p's can't be skipped in a path
        skip_optional = [float("inf")] * num_tokens_with_blanks
        skip_optional[0::2] = [1] + [1 if c_i != c_i_plus_1 else float("inf")
                                     for c_i, c_i_plus_1 in zip(tokens[:-1], tokens[1:])] + [1]
        skip_optional = torch.Tensor(skip_optional)
        min_path_len = len(skip_optional) - torch.sum(skip_optional == 1)

        # The extra dim for time axis is for simplification of the code.
        trellis = torch.full((num_frame + 1, num_tokens_with_blanks), -float("inf"))
        trellis[0, [0, 1]] = 0
        # The path can start from blank or from first non-blank character
        trellis[1:-min_path_len, 0] = torch.cumsum(emission[:-min_path_len, self.blank_id], 0)
        trellis[1, 1] = emission[0, tokens[0]]
        # trellis[0, -num_tokens_with_blanks:] = -float("inf")
        # trellis[-min_path_len:, 0] = -float("inf")

        for t in range(1, num_frame):
            max_prob_source = torch.max(torch.stack(
                                     [
                                         # Score for staying at the same token
                                         trellis[t, 1:],
                                         # Score for changing to the next token from previous token
                                         trellis[t, : -1],
                                         # Score for changing to the next token while skipping unnecessary blank token
                                         torch.cat([torch.Tensor([-float("inf")]), skip_optional[1:-1] * trellis[t, :-2], ])
                                     ]), 0
                                 )
            trellis[t + 1, 1:] = emission[t, tokens_with_blank[1:]] + max_prob_source.values
        return trellis

    def backtrack(self, trellis, emission, tokens, ):
        # Note:
        # j and t are indices for trellis, which has extra dimensions
        # for time and tokens at the beginning.
        # When referring to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when referring to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 2 + torch.argmax(trellis[-1, -2:])
        # t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(trellis.size(0) - 1, 0, -1):

            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t, j]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t, j - 1] if j > 0 else float("-inf")

            skipped_and_changed = trellis[t, j - 2] if j % 2 == 1 and j > 1 and tokens[j // 2] != tokens[j // 2 - 1] \
                else float("-inf")

            max_prob_source = max(stayed, changed, skipped_and_changed)

            # 3. Update the token
            if max_prob_source == changed and max_prob_source != stayed:
                j -= 1
            elif max_prob_source == skipped_and_changed:
                j -= 2

            # 2. Store the path with frame-wise probability.
            token = tokens[j // 2] if j % 2 else self.blank_id
            prob = emission[t - 1, token].exp().item()
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(token, t - 1, prob))

        return path[::-1]

    # Merge the labels
    def merge_repeats(self, path):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token == path[i2].token:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    path[i1].token,
                    self.processor.tokenizer.decode(path[i1].token),
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments


@dataclass
class Point:
    token: int
    time_index: int
    score: float


@dataclass
class Segment:
    token: int
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start




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
        emissions = torch.log_softmax(logits, dim=-1)
        emission = emissions[0].cpu().detach()
        trellis = self.get_trellis(emission, gt_ids)

        path = self.backtrack(trellis, emission, gt_ids)

        segments = self.merge_repeats(path, gt_text)
        return segments

    def get_trellis(self, emission, tokens, ):
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        # Trellis has extra diemsions for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code.
        trellis = torch.empty((num_frame + 1, num_tokens + 1))
        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(emission[:, self.blank_id], 0)
        trellis[0, -num_tokens:] = -float("inf")
        trellis[-num_tokens:, 0] = float("inf")

        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, self.blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis

    def backtrack(self, trellis, emission, tokens, blank_id=0):
        # Note:
        # j and t are indices for trellis, which has extra dimensions
        # for time and tokens at the beginning.
        # When referring to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when referring to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            # 2. Store the path with frame-wise probability.
            prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(j - 1, t - 1, prob))

            # 3. Update the token
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError("Failed to align")
        return path[::-1]

    # Merge the labels
    def merge_repeats(self, path, gt_text):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    gt_text[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start




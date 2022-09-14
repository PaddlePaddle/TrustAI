import paddle
from paddlenlp.transformers import AutoModelForQuestionAnswering


class Predictor(paddle.nn.Layer):

    def __init__(self, args):
        super(Predictor, self).__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    def forward(self, x, mode="train"):
        data = {
            "input_ids": x[0],
            "token_type_ids": x[1],
        }
        if mode == "train":
            data["start_positions"] = x[2]
            data["end_positions"] = x[3]

        logits = self.model(input_ids=data["input_ids"], token_type_ids=data["token_type_ids"])

        if mode == "dev" or mode == "test":
            return logits

        # Compute loss
        start_logits, end_logits = logits
        start_position = paddle.unsqueeze(data["start_positions"], axis=-1)
        end_position = paddle.unsqueeze(data["end_positions"], axis=-1)
        start_loss = paddle.nn.functional.cross_entropy(input=start_logits, label=start_position)
        end_loss = paddle.nn.functional.cross_entropy(input=end_logits, label=end_position)
        loss = (start_loss + end_loss) / 2
        return loss, logits

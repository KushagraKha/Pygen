from transformers import AutoTokenizer

MODEL_NAME = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class APPSCollator:
    def __init__(self, tokenizer, max_source_length=512, max_target_length=512):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, batch):
        # batch is a list of (question_text, solution_text) pairs
        questions, solutions = zip(*batch)

        encodings = self.tokenizer(
            list(questions),
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt"
        )

        decodings = self.tokenizer(
            list(solutions),
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt"
        )

        # We typically set the label IDs to -100 where there's padding, so the loss ignores them
        labels = decodings["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,  # The decoder target
        }





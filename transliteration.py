import torch

from peft import PeftModel
from transformers import MarianMTModel, MarianTokenizer, PreTrainedModel


class Transliterator(PreTrainedModel):
    def __init__(self, config, base_model=None, tokenizer=None, lora_model=None):
        super().__init__(config)

        self.model = base_model
        self.tokenizer = tokenizer
        self.lora_model = lora_model

    @classmethod
    def from_pretrained(cls, model_path):
        base_model = MarianMTModel.from_pretrained(model_path)
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        lora_model = PeftModel.from_pretrained(base_model, model_path)
        config = base_model.config

        return cls(config, base_model, tokenizer, lora_model)

    @torch.inference_mode()
    def transliterate(self, text, max_length=64):
        inputs = self.tokenizer(text, return_tensors='pt')

        outputs = self.lora_model.generate(
            **inputs,
            max_length=max_length,
            num_beams=3,
            temperature=0.7,
            do_sample=True,
        )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return result

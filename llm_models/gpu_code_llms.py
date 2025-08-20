from llm_models.code_llms import CodeLLM
from typing import Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class TransformersCodeLLM(CodeLLM):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def invoke(self, input_variables: Dict[str, str], prompt_template: str):
        # format the prompt
        prompt = prompt_template.format(**input_variables)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # generate with log probabilities
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            return_dict_in_generate=True,
            output_scores=True
        )

        # decode text
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # compute token-level logprobs
        scores = outputs.scores  # list[tensor], each tensor is [batch, vocab_size]
        gen_tokens = outputs.sequences[0][inputs["input_ids"].shape[-1]:]  # only new tokens
        logprobs = []
        for i, token_id in enumerate(gen_tokens):
            logits = scores[i][0]  # (vocab_size,)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            logprobs.append(log_probs[token_id].item())

        return {
            "text": generated_text,
            "tokens": [self.tokenizer.decode([t]) for t in gen_tokens],
            "logprobs": logprobs
        }
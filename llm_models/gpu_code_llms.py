from llm_models.code_llms import CodeLLM
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

class TransformersCodeLLM(CodeLLM):
    def __init__(self, model_name: str, answers: List[Any] = None) -> None:
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only = False)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
        if answers is not None:
            self.obtain_max_new_tokens(answers = answers)
        else:
            self.max_new_token = 512

        print("LLM model successfully deployed.")

    def obtain_max_new_tokens(self, answers: List[Any]):
        longest_token = max(len(self.tokenizer.encode(ans)) for ans in answers)
        x =2
        while x < longest_token:
            x *= 2
        self.max_new_token = x

    def invoke(self, input_variables: Dict[str, str], prompt_template: str) -> Dict[str, str | float]:
        # format the prompt
        prompt = prompt_template.format(**input_variables)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # generate with log probabilities
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_token,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # decode text
        gen_tokens = outputs.sequences[0][inputs["input_ids"].shape[-1]:]
        answer_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # compute token-level logprobs
        scores = outputs.scores  # list[tensor], each tensor is [batch, vocab_size]
        gen_tokens = outputs.sequences[0][inputs["input_ids"].shape[-1]:]  # only new tokens
        logprobs = []
        for i, token_id in enumerate(gen_tokens):
            logits = scores[i][0]  # (vocab_size,)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            logprobs.append(log_probs[token_id].item())

        geom_mean_prob = math.exp(sum(logprobs) / len(logprobs)) if logprobs else 0.0

        return {
            "ans": answer_text,
            "geom_mean_prob": geom_mean_prob
        }
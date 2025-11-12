# whipa deployment
import os
import json

from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor
from peft import PeftModel

from scripts.metrics import retokenize_ipa


class WHIPA:
    def __init__(self, model_path: str, base_model_name: str = False,
                 lora: bool = False):
        self.model_path = model_path
        self.lora = lora
        self.base_model_name = (base_model_name if base_model_name else self.infer_base())
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
        self.ft_config = self.load_config()
        self.processor = WhisperProcessor.from_pretrained(self.base_model_name, task="transcribe")

    def infer_base(self):
        # infer base model name from model_path
        if "base" in self.model_path:
            return "openai/whisper-base"
        elif "large" in self.model_path:
            return "openai/whisper-large-v2"
        else:
            raise ValueError("Could not infer base model name from model_path; specify base_model_name argument.")

    def load_config(self):
        config_path = os.path.join(self.model_path, "ft_config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                ft_config = json.load(f)
        else:
            ft_config = {"gen_args": {"num_beams": 1}, "fallback": [3,5,7]}
            print(f"Warning: No ft_config.json found in {self.model_path}, using default n_beams=1 and fallback_beams=[3,5,7]")
        return ft_config

    def get_tokenizer(self):
        if self.lora:
            tokenizer = WhisperTokenizer.from_pretrained(self.base_model_name, task="transcribe")
            # add special IPA token to tokenizer 
            tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<|ip|>"] + tokenizer.all_special_tokens})
        else:
            tokenizer = WhisperTokenizer.from_pretrained(self.model_path, task="transcribe")
        return tokenizer

    def get_model(self):
        if self.lora:
            # load vanilla Whisper model as base
            base_model = WhisperForConditionalGeneration.from_pretrained(self.base_model_name)
            # add custom IPA language ID to model
            base_model.generation_config.lang_to_id["<|ip|>"] = self.tokenizer.convert_tokens_to_ids(["<|ip|>"])[0]
            # resize token embeddings to account for new special token
            base_model.resize_token_embeddings(len(self.tokenizer))
            # load LoRA adapter model
            whipa_model = PeftModel.from_pretrained(base_model, self.model_path)

        else:
            whipa_model = WhisperForConditionalGeneration.from_pretrained(self.model_path)

        whipa_model.generation_config.language = "<|ip|>"
        whipa_model.generation_config.task = "transcribe"

        return whipa_model.eval()

    def transcribe_ipa(self, sample, n_beams: int = 0, fallback: bool = True,
                       max_phones_per_sec_rate=20.0, fallback_beams: list = False,
                       verbose: bool = False):
        if not n_beams:
            n_beams = self.ft_config["gen_args"]["num_beams"]
        input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to(whipa.model.device)
        out = self.tokenizer.decode(self.model.generate(input_features, num_beams=n_beams)[0])

        if fallback:
            if not fallback_beams:
                fallback_beams = [x for x in (1,3,5,7) if x!=n_beams]

            rate_limit = max(round(len(sample["audio"]["array"])/16_000) * max_phones_per_sec_rate, 1)  # safety net for 0sec length samples
            if verbose: print(f"Sample length: {round(len(sample['audio']['array'])/16_000)} seconds; Rate limit: max {rate_limit} phones estimated")
            # check validity of prediciton
            if len(retokenize_ipa(out)) > rate_limit:  # max articulatory phone/secs rate
                if verbose: print(f"Predcition: '{out}'\nWarning: {len(retokenize_ipa(out))} phones predicted (max: {rate_limit}); starting fallback")
                # 1: beam backoff
                backoff_bank = (fallback_beams if fallback_beams
                                    else ([x for x in (1,3,5,7) if x>n_beams] 
                                          + [x for x in (1,3,5,7) if x<n_beams]))
                backoff = backoff_bank.copy()

                # try to alter beam size and yield a non-overshot transcription
                while (len(retokenize_ipa(out)) > rate_limit) and backoff:
                    beams = list(backoff)[0]
                    backoff.remove(beams)
                    out = self.tokenizer.decode(self.model.generate(input_features, num_beams=beams)[0])

            # 2: Beam backoff failed; try backoff with repetition_penalty
            if (len(retokenize_ipa(out))) > rate_limit:
                # reset decoding backoff
                backoff = backoff_bank.copy()
                # first try og beam size again with penatly, then repeat backoff
                beams = n_beams

                while (len(retokenize_ipa(out)) > rate_limit) and backoff:
                    out = self.tokenizer.decode(self.model.generate(
                            input_features, num_beams=beams,repetition_penalty=1.15)[0])
                    # update beam size if loop is not excited
                    beams = list(backoff)[0]
                    backoff.remove(beams)

            # 3: Beam backoff with repetition penalty also failed;
            # truncate prediciton with exponential weight decay penalty
            if (len(retokenize_ipa(out))) > rate_limit:
                penalty = 2.0
                while (len(retokenize_ipa(out)) > rate_limit) and penalty <= 5:
                    out = self.tokenizer.decode(self.model.generate(input_features, num_beams=n_beams,
                                exponential_decay_length_penalty=(int(rate_limit*0.8), penalty))[0])
                    penalty += 1.5
            # if this still fails somehow (positive logs for example); force truncate
            if (len(retokenize_ipa(out))) > rate_limit:
                out = self.tokenizer.decode(self.model.generate(input_features, num_beams=n_beams)[0])[:rate_limit]

        return out


# if __name__=="__main__":
#     import torch
#     from datasets import Audio, Dataset
#     from scripts.whipa_utils import prep_dataset
#     # example usage
#     data = Dataset.from_json("../data/multipa/test/test_set.json").cast_column("audio", Audio(sampling_rate=16_000))
#     whipa = WHIPA(model_path="../models/CV/lowhipa-large-cv/checkpoint-880", lora=True)
#     data = prep_dataset(data, whipa.processor, whipa.tokenizer, seed=False,
#                         max_len=whipa.model.generation_config.max_length)
#     print(whipa.transcribe_ipa(data[0], verbose=True))

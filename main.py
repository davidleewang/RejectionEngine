from unsloth import FastLanguageModel
from transformers import TextStreamer

max_seq_length = 2048
dtype = None # None for auto detection
load_in_4bit = True # using 4bit quantization

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # loading fine-tuned model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # enabling faster inference

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

inputs = tokenizer(
[
    alpaca_prompt.format(
        "Write a rejection email.", # instruction
        "The applicant's name is Amanda. The company she is applying to is West Monroe.", # input
        "", # output - left blank for generation
    )
], return_tensors = "pt").to("cuda")

# using a text streamer to get the 'ChatGPT' effect
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
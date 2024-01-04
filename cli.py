# -*- coding: utf-8 -*-

"""
Chat with a model with command line interface.

Usage:
python3 -m cli --model ronniewy/vicuna_api_parameters  # huggingface
python3 -m cli --model ./checkpoints/checkpoint-500 # 本地
python3 -m cli --model lmsys/fastchat-t5-3b-v1.0

Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
- Type "!!remove" to remove the last prompt.
- Type "!!regen" to regenerate the last message.
- Type "!!save <filename>" to save the conversation history to a json file.
- Type "!!load <filename>" to load a conversation history from a json file.
"""

import argparse

from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template, get_conv_template
from fastchat.model.model_adapter import VicunaAdapter
from fastchat.serve.cli import main


USE_FINE_TUNE_MODEL = True
# 运行时动态修改vicuna默认模板

if USE_FINE_TUNE_MODEL:
    register_conv_template(
        Conversation(
            name="vicuna_v1.2",
            system_message="[Responsibility]\nYou are a test engineer, especially good at interface testing, "
                           "responsible for performing generative tasks based on APIs which defined in JSON Schema format. "
                           "Task description: generate valid/invalid values for API request parameters.\n\n"
                           "[Output Specification]\nThe output should strictly follow the JSON structure shown below:\n"
                           "{\"type\": \"object\", \"properties\": {\"street_address\": {\"type\": \"string\"}, \"country\": {\"default\": \"United States of America\", \"enum\": [\"United States of America\", \"Canada\"]}, \"postal_code\": {\"type\": \"string\", \"pattern\": \"\"}, \"phone_number\": {\"type\": \"string\", \"pattern\": \"\"}}, \"required\": [\"street_address\", \"country\", \"postal_code\"], \"if\": {\"properties\": {\"country\": {\"const\": \"United States of America\"}}}, \"then\": {\"properties\": {\"postal_code\": {\"pattern\": \"[0-9]{5}(-[0-9]{4})?\"}, \"phone_number\": {\"pattern\": \"\\\\(\\\\d{3}\\\\) \\\\d{3}-\\\\d{4}\"}}}, \"else\": {\"properties\": {\"postal_code\": {\"pattern\": \"[A-Z][0-9][A-Z] \\\\d[A-Z]\\\\d\"}, \"phone_number\": {\"pattern\": \"\\\\+\\\\d{1,3}-\\\\d{3}-\\\\d{4}\"}}}}\n\n",
            roles=("human", "gpt"),
            sep_style=SeparatorStyle.ADD_COLON_TWO,
            sep=" ",
            sep2="</s>",
        )
    )

    # 运行时动态修改vicuna默认模板获取方法
    def custom_get_default_conv_template(cls, model_path: str) -> Conversation:
        return get_conv_template("vicuna_v1.2")

    VicunaAdapter.get_default_conv_template = classmethod(custom_get_default_conv_template)


def add_model_args(parser):
    parser.add_argument(
        "--model-path",
        type=str,
        default="ronniewy/vicuna_api_parameters",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face Hub model revision identifier",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu", "npu"],
        default="cuda",
        help="The device type",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per GPU for storing model weights. Use a string like '13Gib'",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--cpu-offloading",
        action="store_true",
        help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU",
    )
    parser.add_argument(
        "--gptq-ckpt",
        type=str,
        default=None,
        help="Used for GPTQ. The path to the local GPTQ checkpoint.",
    )
    parser.add_argument(
        "--gptq-wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="Used for GPTQ. #bits to use for quantization",
    )
    parser.add_argument(
        "--gptq-groupsize",
        type=int,
        default=-1,
        help="Used for GPTQ. Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--gptq-act-order",
        action="store_true",
        help="Used for GPTQ. Whether to apply the activation order GPTQ heuristic",
    )
    parser.add_argument(
        "--awq-ckpt",
        type=str,
        default=None,
        help="Used for AWQ. Load quantized model. The path to the local AWQ checkpoint.",
    )
    parser.add_argument(
        "--awq-wbits",
        type=int,
        default=16,
        choices=[4, 16],
        help="Used for AWQ. #bits to use for AWQ quantization",
    )
    parser.add_argument(
        "--awq-groupsize",
        type=int,
        default=-1,
        help="Used for AWQ. Groupsize to use for AWQ quantization; default uses full row.",
    )
    parser.add_argument(
        "--enable-exllama",
        action="store_true",
        help="Used for exllamabv2. Enable exllamaV2 inference framework.",
    )
    parser.add_argument(
        "--exllama-max-seq-len",
        type=int,
        default=4096,
        help="Used for exllamabv2. Max sequence length to use for exllamav2 framework; default 4096 sequence length.",
    )
    parser.add_argument(
        "--exllama-gpu-split",
        type=str,
        default=None,
        help="Used for exllamabv2. Comma-separated list of VRAM (in GB) to use per GPU. Example: 20,7,7",
    )
    parser.add_argument(
        "--enable-xft",
        action="store_true",
        help="Used for xFasterTransformer Enable xFasterTransformer inference framework.",
    )
    parser.add_argument(
        "--xft-max-seq-len",
        type=int,
        default=4096,
        help="Used for xFasterTransformer. Max sequence length to use for xFasterTransformer framework; default 4096 sequence length.",
    )
    parser.add_argument(
        "--xft-dtype",
        type=str,
        choices=["fp16", "bf16", "int8", "bf16_fp16", "bf16_int8"],
        help="Override the default dtype. If not set, it will use bfloat16 for first token and float16 next tokens on CPU.",
        default=None,
    )


# 获取用户命令行参数，然后启动模型
parser = argparse.ArgumentParser()

# 如果没有通过`--model`传参，这里会设置默认的模型"ronniewy/vicuna_api_parameters"
add_model_args(parser)

parser.add_argument(
    "--conv-template", type=str, default=None, help="Conversation prompt template."
)
parser.add_argument(
    "--conv-system-msg", type=str, default=None, help="Conversation system message."
)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--no-history", action="store_true")
parser.add_argument(
    "--style",
    type=str,
    default="simple",
    choices=["simple", "rich", "programmatic"],
    help="Display style.",
)
parser.add_argument(
    "--multiline",
    action="store_true",
    help="Enable multiline input. Use ESC+Enter for newline.",
)
parser.add_argument(
    "--mouse",
    action="store_true",
    help="[Rich Style]: Enable mouse support for cursor positioning.",
)
parser.add_argument(
    "--judge-sent-end",
    action="store_true",
    help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Print useful debug information (e.g., prompts)",
)
args = parser.parse_args()
main(args)

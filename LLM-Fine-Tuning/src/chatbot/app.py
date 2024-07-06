import os
from pyprojroot import here
import traceback
import json
import chainlit as cl
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.functions_prep import PrepareFunctions
from utils.llm_function_caller import LLMFuntionCaller
from utils.memory import Memory
from utils.load_config import LoadConfig
from utils.inference import InferenceGPT

from typing import Dict
APP_CFG = LoadConfig()


model_path = 'openlm-research/open_llama_3b'
finetuned_model_dir = here(
    f"models/fine_tuned_models/CubeTriangle_open_llama_3b_2e_qa_qa")
max_input_tokens = 1000
max_length = 100

llm = AutoModelForCausalLM.from_pretrained(
    finetuned_model_dir, local_files_only=True, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)


def ask_cubetriangle_llm(query: str):
    """
    Generates a response from a Cubetriangle Company's private large language model based on the given query.

    Parameters:
    - query (str): The input query for the language model.

    Returns:
    - str: The generated response from the language model.
    """
    inputs = tokenizer(query, return_tensors="pt",
                       truncation=True, max_length=max_input_tokens).to("cuda")
    tokens = llm.generate(**inputs, max_length=max_length)
    response = tokenizer.decode(
        tokens[0], skip_special_tokens=True)[len(query):]
    return response


@cl.on_chat_start
async def on_chat_start():
    try:
        cl.user_session.set("session_time", str(int(time.time())))
        await cl.Avatar(
            name="Enterprise LLM",
            path="src/chatbot/public/logo.png"
        ).send()
        await cl.Avatar(
            name="Error",
            url="src/chatbot/public/logo.png"
        ).send()
        await cl.Avatar(
            name="User",
            path="src/chatbot/public/logo_light.png"
        ).send()
        if not os.path.exists("memory"):
            os.makedirs("memory")
        await cl.Message(f"Hello! I am the CubeTriangle ChatBot. How can I help you?").send()
    except BaseException as e:
        print(f"Caught error on on_chat_start in app.py: {e}")
        traceback.print_exc()


@cl.on_message
async def on_message(message: cl.Message):
    try:
        # display loader spinner while performing actions
        msg = cl.Message(content="")
        await msg.send()
        chat_history_lst = Memory.read_recent_chat_history(
            file_path=APP_CFG.memory_directry.format(
                cl.user_session.get("session_time")), num_entries=APP_CFG.num_entries)
        # Prepare input for the first model (function caller)
        input_chat_history = str(chat_history_lst)
        messages = LLMFuntionCaller.prepare_messages(
            APP_CFG.llm_function_caller_system_role, input_chat_history, message.content)
        print("First LLM messages:", messages, "\n")
        # Pass the input to the first model (function caller)
        llm_function_caller_full_response = LLMFuntionCaller.ask(
            APP_CFG.llm_function_caller_gpt_model, APP_CFG.llm_function_caller_temperature, messages, [PrepareFunctions.jsonschema(
                ask_cubetriangle_llm)])

        # If function called indeed called out a function
        if "function_call" in llm_function_caller_full_response.choices[0].message.keys():
            print(
                llm_function_caller_full_response.choices[0].message, "\n")
            # Get the pythonic response of that function
            func_name: str = llm_function_caller_full_response.choices[
                0].message.function_call.name
            print("\nCalled function:", func_name)
            func_args: Dict = json.loads(
                llm_function_caller_full_response.choices[0].message.function_call.arguments)
            # Call the function with the given arguments
            if func_name == 'ask_cubetriangle_llm':
                llm_response = ask_cubetriangle_llm(
                    **func_args)
            else:
                raise ValueError(f"Function '{func_name}' not found.")
            messages = InferenceGPT.prepare_messages(
                llm_response=llm_response, user_query=message.content, llm_system_role=APP_CFG.llm_inference_system_role, input_chat_history=input_chat_history)
            print("Second LLM messages:", messages, "\n")
            llm_inference_full_response = InferenceGPT.ask(
                APP_CFG.llm_inference_gpt_model, APP_CFG.llm_inference_temperature, messages)
            # print the response for the user
            llm_inference_response = llm_inference_full_response[
                "choices"][0]["message"]["content"]
            await msg.stream_token(llm_inference_response)
            chat_history_lst = [
                (message.content, llm_inference_response)]

        else:  # No function was called. LLM function caller is using its own knowledge.
            llm_function_caller_response = llm_function_caller_full_response[
                "choices"][0]["message"]["content"]
            await msg.stream_token(llm_function_caller_response)
            chat_history_lst = [
                (message.content, llm_function_caller_response)]

        # Update memory
        Memory.write_chat_history_to_file(chat_history_lst=chat_history_lst, file_path=APP_CFG.memory_directry.format(
            cl.user_session.get("session_time")))

    except BaseException as e:
        print(f"Caught error on on_message in app.py: {e}")
        traceback.print_exc()
        await cl.Message("An error occured while processing your query. Please try again later.").send()


# Generate question embedding
# In this module, we generate embeddings for Human QA'ed dataset containing (question, inquiryUsed, categoryNm, serviceNm). The embeddings are generated for frequently asked Benefit related questions. Each question has inquiryUsed, categoryNm, serviceNm fields from NLS API which were validated through NLS schema 

# After generating embeddings we cache question, inquiryUsed, categoryNm, serviceNm along with generated question embeddings into OpenSearch. 
# This is an offline activity and not updated frequently.
# BeCA Evaluation
# Created by Gong, Yiru, last modified on May 17, 2024
# Evaluation Framework
# Offline Evaluation

# Query Parsing Module Evaluation
# Ranking Module Evaluation
# LLM Answer Evaluation
# LLMQA
# Online Evaluation

# End-to-end Regression Test


# BeCA Evaluation
# Created by Gong, Yiru, last modified on May 17, 2024
# Evaluation Framework
# Offline Evaluation

# Query Parsing Module Evaluation
# Ranking Module Evaluation
# LLM Answer Evaluation
# LLMQA
# Online Evaluation

# End-to-end Regression Test


# Fine tuning
# FDSP and DeepSpeed can be used to improve fine tuning performance. FDSP and DeepSpeed can both be used via huggingface accelerate, so training scripts that are using HuggingFace normally can directly used these techniques.

# How To
# Create a python script that loads the data and does the training. This works with normal fine tuning, or with KTO/DPO and any other techniques. Example script: kto.py
# If the model supports it, you can also use flash attention to further improve performance
# Run the script with 5 data samples to make sure it works as expected. Make sure the script is saving the model at the end as expected.
# Use
# accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml train_mistral_7b.py
# with the final dataset to speed up using FSDP or DeepSpeed. Example configs can be found here: fsdp.yaml . This should launch the model using all GPUs as per the config file, and training should be happening in parallel. Note: this only works right now if the model is small enough to fit in a single GPU during fine tuning.

# Inferencing
# For inferencing, smaller models can be optimised by running the inferencing in parallel on multiple GPUs, by setting device_map=auto when loading the model. For larger models which don't fit on a single GPU, there is currently no known way to improve performance over the huggingface transformers baseline.

# Techniques tested:
# Flash attention - almost 2x speedup on mistral 7b, no impact on llama2-70B. - https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2
# HuggingFace Optimum Nvidia - All optimisations here have already been merged into transformers 4.3, so no point in using this module
# DeepSpeed MII - does not currently support LORA modules, which is how we are fine tuning models at present.
# exllamav2 - Requires converting the model to their own format (GPTQ quantisation). Quantization level can be selected. 2.6bpw was ~2-3x faster at inference, even with a LoRA module loaded. However we need to train the LoRA module with the same quantisation using exllamav2.
# vllm - Does not work on Carelon Jupyter Hub due to a small /dev/shm (it needs at least 1GB, while all instances are provisioned with 64MB). On KubeFlow, it does not support tensor parallelism or pipeline parallelism (splitting a large model onto multiple GPUs) when using bitsandbytes quantisation, which means it has the same limitation as exllamav2- we can load the LoRA, but the model's quantisation has changed.
# Unsloth - Does not work on Carelon Jupyter Hub - core dump during initial import, might be due to small /dev/shm size. On Kubeflow it successfully loads the model, but inference fails due to device map issues.
# Changing generation config: The default huggingface generation method uses greedy search, which is much faster, however it generates slightly worse answers. Generally beam search is preferred for longer answers, but a beam search with n=3 takes almost 3x longer than greedy generation.
# This notebook demonstrates loading a model and merging the QLORA adapter, and demonstrates the greedy generation (base configuration) vs beam search Llama2 Inference (1).ipynb



# Conclusions
# The model quantisation scheme should be picked before training, based on tradeoffs during both training and inferencing. Importantly, we need to check with engineering if they can deploy the model with the same quantisation, since converting between quantisations is lossy.
# Unsloth is currently not usable on Kubeflow or Carelon Jupyter Hub. Vllm and Exllamav2 are both usable, but have their own set of quantisations which need to be evaluated.

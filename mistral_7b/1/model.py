import random

import ray
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from instill.helpers.const import DataType, TextGenerationChatInput
from instill.helpers.ray_io import StandardTaskIO
from instill.helpers.ray_config import (
    instill_deployment,
    InstillDeployable,
)

from ray_pb2 import (
    ModelReadyRequest,
    ModelReadyResponse,
    ModelMetadataRequest,
    ModelMetadataResponse,
    ModelInferRequest,
    ModelInferResponse,
    InferTensor,
)

import json


@instill_deployment
class Mistral8x7b:
    def __init__(self, model_path: str):
        # model_id = "mistralai/Mixtral-8x7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_safetensors=True,
            torch_dtype=torch.float16
        )

    def ModelMetadata(self, req: ModelMetadataRequest) -> ModelMetadataResponse:
        resp = ModelMetadataResponse(
            name=req.name,
            versions=req.version,
            framework="python",
            inputs=[
                ModelMetadataResponse.TensorMetadata(
                    name="prompt",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="prompt_images",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="chat_history",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="system_message",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="max_new_tokens",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="temperature",
                    datatype=str(DataType.TYPE_FP32.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="top_k",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="random_seed",
                    datatype=str(DataType.TYPE_UINT64.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="extra_params",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                ModelMetadataResponse.TensorMetadata(
                    name="text",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[-1, -1],
                ),
            ],
        )
        return resp

    def ModelReady(self, req: ModelReadyRequest) -> ModelReadyResponse:
        resp = ModelReadyResponse(ready=True)
        return resp

    async def ModelInfer(self, request: ModelInferRequest) -> ModelInferResponse:
        resp = ModelInferResponse(
            model_name=request.model_name,
            model_version=request.model_version,
            outputs=[],
            raw_output_contents=[],
        )

        task_text_generation_chat_input: TextGenerationChatInput = (
            StandardTaskIO.parse_task_text_generation_chat_input(request=request)
        )
        print("----------------________")
        print(task_text_generation_chat_input)
        print("----------------________")

        print("print(task_text_generation_chat.prompt")
        print(task_text_generation_chat_input.prompt)
        print("-------\n")

        print("print(task_text_generation_chat.prompt_images")
        print(task_text_generation_chat_input.prompt_images)
        print("-------\n")

        print("print(task_text_generation_chat.chat_history")
        print(task_text_generation_chat_input.chat_history)
        print("-------\n")

        print("print(task_text_generation_chat.system_message")
        print(task_text_generation_chat_input.system_message)
        print("-------\n")

        print("print(task_text_generation_chat.max_new_tokens")
        print(task_text_generation_chat_input.max_new_tokens)
        print("-------\n")

        print("print(task_text_generation_chat.temperature")
        print(task_text_generation_chat_input.temperature)
        print("-------\n")

        print("print(task_text_generation_chat.top_k")
        print(task_text_generation_chat_input.top_k)
        print("-------\n")

        print("print(task_text_generation_chat.random_seed")
        print(task_text_generation_chat_input.random_seed)
        print("-------\n")

        print("print(task_text_generation_chat.stop_words")
        print(task_text_generation_chat_input.stop_words)
        print("-------\n")

        print("print(task_text_generation_chat.extra_params")
        print(task_text_generation_chat_input.extra_params)
        print("-------\n")

        # if task_text_generation_chat_input.temperature <= 0.0:
        #     task_text_generation_chat_input.temperature = 0.8

        # if task_text_generation_chat_input.random_seed > 0:
        #     random.seed(task_text_generation_chat_input.random_seed)
        #     np.random.seed(task_text_generation_chat_input.random_seed)
        

        
        inputs = self.tokenizer(task_text_generation_chat_input.prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=20)
        print("output: ")
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))


        sequences = []
        for output in outputs:
            # concated_complete_output = prompt + "".join([ # Chat model no needs to repeated the prompt
            sequences.append(
                {"generated_text": self.tokenizer.decode(output, skip_special_tokens=True)}
            )

        task_text_generation_chat_output = (
            StandardTaskIO.parse_task_text_generation_chat_output(sequences=sequences)
        )

        resp.outputs.append(
            InferTensor(
                name="text",
                shape=[1, len(sequences)],
                datatype=str(DataType.TYPE_STRING),
            )
        )

        resp.raw_output_contents.append(task_text_generation_chat_output)

        return resp


deployable = InstillDeployable(
    Mistral8x7b, model_weight_or_folder_name="Mixtral-8x7B-v0.1/", use_gpu=True
)

deployable.update_max_replicas(2)
deployable.update_min_replicas(0)

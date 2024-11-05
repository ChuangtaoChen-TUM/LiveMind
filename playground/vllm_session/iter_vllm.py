""" vllm model supporting streaming. 
This implementation is based on the vllm 0.5.3.post1, the API may change in the future. """
from typing import List
from vllm import LLM
import torch
from transformers import TextIteratorStreamer
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm import LLMEngine

__all__ = ["ItervLLM"]

class ItervLLM(LLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iter_generate(
        self,
        input_ids: torch.Tensor|list[list[int]],
        max_new_tokens: int,
        eos_token_id: list,
        do_sample: bool=True,
        temperature: float|None=None,
        top_p: float|None=None,
        top_k: int|None=None,
        streamer: TextIteratorStreamer|None=None,
        seed: int=0,
        **kwargs,
    ):
        if not do_sample:
            temperature = 0.0
            top_p = 1.0
            top_k = -1
        if temperature is None:
            temperature = 0.0
        if top_p is None:
            top_p = 1.0
        if top_k is None:
            top_k = -1
        param = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            stop_token_ids=eos_token_id,
            seed=seed,
        )
        if isinstance(input_ids, torch.Tensor):
            assert input_ids.shape[0] == 1, "Batch size must be 1"
            input_ids = input_ids.tolist()
        inputs = self._convert_v1_inputs(prompts=None, prompt_token_ids=input_ids)
        self._validate_and_add_requests(
            inputs,
            param,
            lora_request=None,
            prompt_adapter_request=None
        )
        engine_out = self._iter_run_engine(streamer=streamer)
        results = LLMEngine.validate_outputs(engine_out, RequestOutput)
        outputs = []
        for output in results:
            outputs.append(output.prompt_token_ids+list(output.outputs[0].token_ids))
        return outputs


    def _iter_run_engine(self, streamer:TextIteratorStreamer=None) -> List[RequestOutput]:
        # Run the engine.
        outputs: List[RequestOutput] = []
        gen_text = ""
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            assert isinstance(step_outputs, list)
            if len(step_outputs) == 1 and streamer is not None:
                assert isinstance(step_outputs[0].outputs, list)
                new_text = step_outputs[0].outputs[0].text[len(gen_text):]
                gen_text = step_outputs[0].outputs[0].text
                streamer.on_finalized_text(new_text, step_outputs[0].finished)
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)

        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))

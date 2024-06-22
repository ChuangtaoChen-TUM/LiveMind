""" modified vllm module to support streaming """
from vllm import LLM
from typing import List, Optional, Union
import torch
from tqdm import tqdm
from transformers import TextIteratorStreamer
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import MultiModalData

__all__ = ["ItervLLM"]

class ItervLLM(LLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(
        self,
        input_ids: torch.Tensor|list[list[int]],
        max_new_tokens: int,
        eos_token_id: list,
        temperature: float,
        top_p: float,
        streamer: TextIteratorStreamer,
        **kwargs,
    ):
        if temperature is None:
            temperature = 0.0
        if top_p is None:
            top_p = 1.0
        param = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            stop_token_ids=eos_token_id,
        )
        if isinstance(input_ids, torch.Tensor):
            assert input_ids.shape[0] == 1, "Batch size must be 1"
            input_ids = input_ids.tolist()
        results = self._generate(prompt_token_ids=input_ids, sampling_params=param, streamer=streamer)
        outputs = []
        for output in results:
            outputs.append(output.prompt_token_ids+output.outputs[0].token_ids)
        return outputs

    def _generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = False,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
        streamer: Optional[TextIteratorStreamer] = None,
    ) -> List[RequestOutput]:
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if self.llm_engine.model_config.skip_tokenizer_init \
            and prompts is not None:
            raise ValueError("prompts must be None if skip_tokenizer_init "
                             "is True")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")

        if prompts is not None:
            num_requests = len(prompts)
        else:
            assert prompt_token_ids is not None
            num_requests = len(prompt_token_ids)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        elif isinstance(sampling_params,
                        list) and len(sampling_params) != num_requests:
            raise ValueError("The lengths of prompts and sampling_params "
                             "must be the same.")
        if multi_modal_data:
            multi_modal_data.data = multi_modal_data.data.to(torch.float16)

        # Add requests to the engine.
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[
                i]
            self._add_request(
                prompt,
                sampling_params[i]
                if isinstance(sampling_params, list) else sampling_params,
                token_ids,
                lora_request=lora_request,
                # Get ith image while maintaining the batch dim.
                multi_modal_data=MultiModalData(
                    type=multi_modal_data.type,
                    data=multi_modal_data.data[i].unsqueeze(0))
                if multi_modal_data else None,
            )
        return self._run_engine(use_tqdm, streamer=streamer)

    def _run_engine(self, use_tqdm: bool, streamer:TextIteratorStreamer=None) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests,
                        desc="Processed prompts",
                        dynamic_ncols=True)
        # Run the engine.
        outputs: List[RequestOutput] = []
        gen_text = ""
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            if len(step_outputs) == 1 and streamer is not None:
                new_text = step_outputs[0].outputs[0].text[len(gen_text):]
                gen_text = step_outputs[0].outputs[0].text
                # print(new_text)
                streamer.on_finalized_text(new_text, step_outputs[0].finished)
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs

<p align="center">
  <h3 align="center"><strong>LiveMind: Low-latency Large Language Models with Simultaneous Inference</strong></h3>

<p align="center">
    Chuangtao Chen<sup>1</sup>,
    Grace Li Zhang<sup>2</sup>,
    XunZhao Yin<sup>3</sup>,
    Cheng Zhuo<sup>3</sup>,
    Ulf Schlichtmann<sup>1</sup>,
    Bing Li<sup>1</sup><br>
    <sup>1</sup>Technical University of Munich<br>
    <sup>2</sup>Technical University of Darmstadt<br>
    <sup>3</sup>Zhejiang Univerity
</p>


<div align="center">

<a href='https://arxiv.org/abs/2406.14319'><img src='https://img.shields.io/badge/arXiv-2406.14319-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href=''><img src='https://img.shields.io/badge/License-MIT-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</div>

<p align="center">
    <img src="./res/01_overview.png" alt>
    <em>(a) LiveMind inference with Llama-3-70B model; (b) LiveMind collaborative inference with Llama-3-70B and Llama-3-8B models; (c) Conventional CoT inference.</em>
</p>

# Usage

## Configurations
Install required packages:
```
pip install datasets alive_progress nltk
```
Before running the script, you need to change the following configurations in `live_mind/config.py` to set the LLMs and datasets:
1. `MMLU_PRO_PATH`: path to the MMLU-Pro dataset, the path should contains `.parquet` dataset files;
2. Implement the `get_model` method: you can use your own model here as long it has the required method (see `live_mind/config.py`);
3. You can also use the `get_model_vllm_example` implementation;
4. To use the `get_model_vllm_example` function, you need to specify the paths `LLAMA_3_8B_PATH` and `LLAMA_3_70B_PATH`. A `config.json` file and `tokenizer.json` file should be found in these paths. Besides, make sure the packages `vllm` and `transformers` are installed.
```
pip install vllm transformers
```
## Run real-time estimation
Run the following commands to reproduce the results of real-time estimation:
```
python run_solver.py --model llama-3-70b --use_lm --output_file ./output/mmlu_pro/time_info/llama_3_70b_lm/all.json
python run_solver.py --model llama-3-70b --output_file ./output/mmlu_pro/time_info/llama_3_70b_base/all.json
python run_solver.py --model llama-3-70b --assist_model llama-3-8b --use_lm --action_set SAS --output_file ./output/mmlu_pro/time_info/llama_3_70b_w_8b_lm/all.json
python run_solver.py --model llama-3-8b --output_file ./output/mmlu_pro/time_info/llama_3_8b_base/all.json
```
## Run batched inference
Run the following commands to reproduce the results of batched inference:
```
python run_batch_solver.py --model llama-3-70b --use_lm --output_file ./output/mmlu_pro/batched/llama_3_70b_lm/all.json
python run_batch_solver.py --model llama-3-70b --output_file ./output/mmlu_pro/batched/llama_3_70b_base/all.json
python run_batch_solver.py --model llama-3-70b --use_lm --assist_model llama-3-8b --action_set SAS --output_file ./output/mmlu_pro/batched/llama_3_70b_w_8b_lm/all.json
python run_batch_solver.py --model llama-3-8b --output_file ./output/mmlu_pro/batched/llama_3_8b_base/all.json
```

## Result analysis
Run the following commands to analyze the output files and reproduce the experiment results:
### Real-time latency measure
```
python analyze_time_info.py ./output/mmlu_pro/time_info/llama_3_70b_lm/all.json
python analyze_time_info.py ./output/mmlu_pro/time_info/llama_3_70b_base/all.json
python analyze_time_info.py ./output/mmlu_pro/time_info/llama_3_70b_w_8b_lm/all.json
python analyze_time_info.py ./output/mmlu_pro/time_info/llama_3_8b_base/all.json
```

This step will create two csv files: `timeinfo_by_category.csv` and `timeinfo_by_len.csv` at each folder with the `all.json` file.
### Batched inference
```
python analyze_batched.py ./output/mmlu_pro/batched/llama_3_70b_lm/all.json
python analyze_batched.py ./output/mmlu_pro/batched/llama_3_70b_base/all.json
python analyze_batched.py ./output/mmlu_pro/batched/llama_3_70b_w_8b_lm/all.json
python analyze_batched.py ./output/mmlu_pro/batched/llama_3_8b_base/all.json
```
This step will create two csv files: `timeinfo_by_category.csv` and `timeinfo_by_len.csv` at each folder with the `all.json` file.

## Action analysis
Run the following commands to reproduce the results present in Sec. 4.4 in the paper:

### Action percentage
```
python analyze_actions.py ./output/mmlu_pro/batched/llama_3_70b_lm/all.json
python analyze_actions.py ./output/mmlu_pro/batched/llama_3_70b_w_8b_lm/all.json
```
This step will create two csv files: `actions_per_step` and `actions_per_len` in these two folders, corresponding to the data presented in Fig. 8.

### Action set
To reproduce the results in Table 2, first run the batched inference with the following configurations:
```
python run_batch_solver.py --model llama-3-8b --use_lm --action_set CAS --output_file ./output/mmlu_pro/ablation/llama_3_8b_lm_comp/all.json
python run_batch_solver.py --model llama-3-8b --use_lm --action_set SAS --output_file ./output/mmlu_pro/ablation/llama_3_8b_lm_simp/all.json
python run_batch_solver.py --model llama-3-8b --use_lm --assist_model llama-3-70b --action_set CAS --output_file ./output/mmlu_pro/ablation/llama_3_8b_w_70b_lm_comp/all.json
python run_batch_solver.py --model llama-3-8b --use_lm --assist_model llama-3-70b --action_set SAS --output_file ./output/mmlu_pro/ablation/llama_3_8b_w_70b_lm_simp/all.json
python run_batch_solver.py --model llama-3-70b --use_lm --action_set SAS --output_file ./output/mmlu_pro/ablation/llama_3_70b_lm_simp/all.json
python run_batch_solver.py --model llama-3-70b --use_lm --assist_model llama-3-8b --action_set CAS --output_file ./output/mmlu_pro/ablation/llama_3_70b_w_8b_lm_simp/all.json
```

Then run `python analyze_batched.py **/all.json` to report the results.

#### Citation

To cite this paper, use:

```
@article{chen2024livemind,
      title={{LiveMind}: Low-latency Large Language Models with Simultaneous Inference},
      author={Chuangtao Chen and Grace Li Zhang and Xunzhao Yin and Cheng Zhuo and Ulf Schlichtmann and Bing Li},
      journal={arXiv preprint arXiv:2406.14319},
      year={2024},
}
```
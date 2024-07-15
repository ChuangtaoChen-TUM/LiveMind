#./output/bi./output/bash

# python run_solver.py -i llama-3-70b -g clause -d mmlu_pro -f --log -pf u-pi
# python run_solver.py -i llama-3-70b -g clause -d mmlu_pro -f --log -pf u-pi --no-hypo
# python run_solver.py -i llama-3-70b -g clause -d mmlu_pro -f --log -pf u-pi --no-sum
# python run_solver.py -i llama-3-70b -g clause -d mmlu_pro -f --log -pf u-pli
# python run_solver.py -i llama-3-70b -g clause -d mmlu_pro -f --log -pf u-pli --no-hypo
python run_solver.py -i llama-3-70b -g clause -d mmlu_pro -f --log -pf u-pli --no-sum
python run_solver.py -i llama-3-70b -g clause -d mmlu_pro -f --log -pf u-pil
python run_solver.py -i llama-3-70b -g clause -d mmlu_pro -f --log -pf u-ip
python run_solver.py -i llama-3-70b -g clause -d mmlu_pro -f --log -pf u-ipl
python run_solver.py -i llama-3-70b -g clause -d mmlu_pro -f --log -pf ua-pil
python run_solver.py -i llama-3-70b -g clause -d mmlu_pro -f --log -pf u-spi
python run_solver.py -i llama-3-70b -g clause -d mmlu_pro -f --log -pf ua-spi

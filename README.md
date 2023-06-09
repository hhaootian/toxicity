# A Benchmark Study of Graph Models for Molecular Acute Toxicity Prediction

## Getting Started

```bash
git clone https://github.com/hhaootian/toxicity
cd toxicity
# install dependencies
pip install -r requirement.txt
# install dgl cuda version
# replace cuda_version, e.g., cu116
# see https://www.dgl.ai/pages/start.html
pip install dgl -f https://data.dgl.ai/wheels/<cuda_version>/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

## Training

```bash
cd src
python setup.py
python clean.py
# replace task_name and model_name
# e.g., python train.py --task-name VF --model-name PagtnModel
python train.py --task-name <task_name> --model-name <model_name>
python test.py --task-name <task_name> --model-name <model_name>
```

Available task and model names (case sensitive):
- task name: VF, DM, TP, Fish
- model name: PagtnModel, GATModel, GCNModel, AttentiveFPModel, MPNNModel

## License

GPL-3.0

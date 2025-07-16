# DISCO: learning to DISCover an evolution Operator for multi-physics-agnostic prediction

This repository contains the code for DISCO.

![DISCO Overview](assets/model_figure.png)
*DISCO uses a large hypernetwork (transformer) to process a short trajectory and generate the parameters of a much smaller operator network, which then predicts the next state through time integration.*

# Installation 

With python 3.11.2.

```python
pip install -r requirements.txt ...
```

# Training the model

```python
python train.py --yaml-config disco_pdebench.yaml
```

You'll need to download PDEBench or The Well datasets and add the paths in `data_utils.py`.
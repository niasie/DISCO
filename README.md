# DISCO: learning to DISCover an evolution Operator for multi-physics-agnostic prediction

This repository will contain the code for implementing DISCO. 

![DISCO Overview](assets/model_figure.png)
*DISCO uses a large hypernetwork (transformer) to process a short trajectory and generate the parameters of a much smaller operator network, which then predicts the next state through time integration.*
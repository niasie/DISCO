from .attention import *
from .disco import *


def build_model(params):
    if params.model == "disco_house":
        return DISCO(
            n_states=params.n_states,
            hidden_dim=params.embed_dim,
            patch_size=params.patch_size,
            ndims=params.ndims,
            dataset_names=params.train_datasets,
            groups=12,
            processor_blocks=params.processor_blocks,
            drop_path=params.drop_path,
            num_heads=params.num_heads,
            bias_type=params.bias_type,
            rtol=params.rtol,
            hpnn_head_hidden_dim=params.hpnn_head_hidden_dim,
            max_steps=params.max_steps,
            integration_library=params.integration_library,
        )
    elif params.model == "operator_network_solver":
        return OperatorNetworkSolver(
            dataset_names=params.train_datasets,
            max_steps=params.max_steps,
            rtol=params.rtol,
            integration_library=params.integration_library,
        )
    else:
        raise ValueError(f"Unknown model type: {params.model}")

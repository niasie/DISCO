# %%
from pathlib import Path
from typing_extensions import OrderedDict
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import YParams

from train import Trainer, model_rollout, nrmse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
class Validator(Trainer):
    def restore_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)

        state_dict = {}

        for k, v in ckpt["model_state"].items():
            if k.startswith("module."):
                k = k.replace("module.", "")

            if "hpnn" in k and "conv." in k:
                k = k.replace("conv.", "")

            if "decoder_common" in k:
                k = k.replace("decoder_common", "param_gen_common")

            if "decoder_head_adim" in k:
                k = k.replace("decoder_head_adim", "param_gen_adim")

            if "decoder_head_dim" in k:
                k = k.replace("decoder_head_dim", "param_gen_dim")

            if "decoder_head_dset" in k:
                k = k.replace("decoder_head_dset", "param_gen_channels")

            if "module." in k:
                k = k.replace("module.", "")

            if k.startswith("pnns."):
                k = k.replace("pnns.", "opnns.")

            if k == "theta_norm_sw":
                k = k.replace("theta_norm_sw", "theta_norm_swe")

            if k == "theta_norm_diffre2":
                k = k.replace("theta_norm_diffre2", "theta_norm_diffre2d")

            state_dict[k] = v

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def validate(self, cutoff: int | None = None) -> OrderedDict:
        """
        Validates - for each batch just use a small subset to make it easier.

        Note: need to split datasets for meaningful metrics, but TBD.
        """
        # Don't bother with full validation set between epochs
        self.model.eval()
        self.single_print("STARTING VALIDATION")
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=False, dtype=self.mp_type):
                sub_dsets = self.valid_dataset.sub_dsets
                logs = {}
                distinct_dsets = [dset.dataset_name for dset in sub_dsets]
                counts = {dset: 0 for dset in distinct_dsets}
                # iterate over the validation datasets
                for subset in sub_dsets:
                    dset_name = subset.dataset_name
                    val_loader = torch.utils.data.DataLoader(
                        subset,
                        batch_size=self.params.batch_size,
                        num_workers=self.params.num_data_workers,
                        shuffle=True,
                        generator=torch.Generator().manual_seed(0),
                        drop_last=True,
                    )
                    count = 0
                    for batch_idx, batch in tqdm(enumerate(val_loader)):
                        # Only do a few batches of each dataset if not doing full validation
                        if cutoff is not None and count >= cutoff:  # validating on burgers equations is extremely long
                            del val_loader
                            break
                        count += 1
                        counts[dset_name] += 1

                        inp, tar = batch["input_fields"].to(self.device), batch["output_fields"].to(self.device)
                        n_past, n_future_val = self.params.n_past, self.params.val_rollout
                        if inp.shape[1] > n_past:
                            # indicates we need to split the input
                            tar = torch.cat([inp[:, n_past:, ...], tar], dim=1)
                            inp = inp[:, :n_past, ...]

                        state_labels = torch.tensor(
                            self.train_dataset.subset_dict.get(
                                subset.get_name(), [-1] * len(self.valid_dataset.subset_dict[subset.get_name()])
                            ),
                            device=self.device,
                        ).unsqueeze(0)

                        # forward
                        output, _ = model_rollout(
                            self.model,
                            inp,
                            predict_normed=True,
                            n_future_steps=min(self.params.val_rollout, tar.shape[1]),
                            state_labels=state_labels[0],
                            dset_name=dset_name,
                        )  # Shape: (B, T, C, H, W) or (B, T, C, W)

                        if batch_idx == 0:  # Only plot for first batch
                            b = 0  # batch index
                            num_channels = output.shape[2]
                            num_timesteps = output.shape[1]  # Plot all timesteps
                            is_1d = len(output.shape) == 4  # (B, T, C, W) is 1D

                            # Create separate plot for each channel
                            for c in range(num_channels):
                                if is_1d:
                                    # 1D data: use line plots
                                    fig, axes = plt.subplots(
                                        2, num_timesteps, figsize=(4 * num_timesteps, 12), squeeze=False
                                    )

                                    for t in range(num_timesteps):
                                        x = torch.arange(output.shape[3])  # spatial dimension

                                        # Plot target and output (row 0)
                                        ax_tar = axes[0, t]
                                        ax_tar.plot(x, tar[b, t, c].cpu().numpy(), label="GT", color="blue")
                                        ax_tar.plot(
                                            x,
                                            output[b, t, c].cpu().numpy(),
                                            label="Prediction",
                                            color="red",
                                            linestyle="--",
                                        )
                                        ax_tar.set_title(f"GT vs Prediction T{t}")
                                        ax_tar.legend()
                                        ax_tar.grid(True)

                                        # Plot absolute difference (row 1)
                                        ax_diff = axes[1, t]
                                        diff = torch.abs(output[b, t, c] - tar[b, t, c]).cpu().numpy()
                                        ax_diff.plot(x, diff, color="red")
                                        ax_diff.set_title(f"Abs Diff T{t}")
                                        ax_diff.grid(True)
                                else:
                                    # 2D data: use imshow
                                    fig, axes = plt.subplots(
                                        3, num_timesteps, figsize=(4 * num_timesteps, 12), squeeze=False
                                    )

                                    # Find common vmin/vmax for target and output
                                    tar_data = tar[b, :, c].cpu().numpy()
                                    out_data = output[b, :, c].cpu().numpy()
                                    vmin = min(tar_data.min(), out_data.min())
                                    vmax = max(tar_data.max(), out_data.max())

                                    for t in range(num_timesteps):
                                        # Plot target (row 0)
                                        ax_tar = axes[0, t]
                                        im_tar = ax_tar.imshow(
                                            tar[b, t, c].cpu().numpy(), cmap="gist_ncar", vmin=vmin, vmax=vmax
                                        )
                                        ax_tar.set_title(f"GT T{t}")
                                        ax_tar.axis("off")
                                        plt.colorbar(im_tar, ax=ax_tar)

                                        # Plot output (row 1)
                                        ax_out = axes[1, t]
                                        im_out = ax_out.imshow(
                                            output[b, t, c].cpu().numpy(), cmap="gist_ncar", vmin=vmin, vmax=vmax
                                        )
                                        ax_out.set_title(f"Prediction T{t}")
                                        ax_out.axis("off")
                                        plt.colorbar(im_out, ax=ax_out)

                                        # Plot absolute difference (row 2)
                                        ax_diff = axes[2, t]
                                        diff = torch.abs(output[b, t, c] - tar[b, t, c]).cpu().numpy()
                                        im_diff = ax_diff.imshow(diff, cmap="viridis")
                                        ax_diff.set_title(f"Abs Diff T{t}")
                                        ax_diff.axis("off")
                                        plt.colorbar(im_diff, ax=ax_diff)

                                plt.suptitle(f"Channel {c} - {dset_name}")
                                plt.tight_layout()
                                plt.show()

                        # loss
                        residuals = output - tar
                        spatial_dims = tuple(range(residuals.ndim))[3:]  # Assume 0, 1 are B, C
                        val_nrmse = nrmse(tar, output, dims=spatial_dims)
                        for step in [1, 2, 4, 8]:
                            if step <= tar.shape[1]:
                                logs[f"{dset_name}/valid_nrmse_t{step}"] = (
                                    logs.get(f"{dset_name}/valid_nrmse_t{step}", 0) + val_nrmse[:, step - 1, ...].mean()
                                )

            self.single_print("DONE VALIDATING - NOW SYNCING")

            # divide by number of batches
            for k, v in logs.items():
                dset_name = k.split("/")[0]
                logs[k] = v / counts[dset_name]

            # # replace keys <>
            # average nrmse across datasets
            logs["valid_nrmse"] = 0
            for dset_name in distinct_dsets:
                logs["valid_nrmse"] += logs[f"{dset_name}/valid_nrmse_t1"] / len(distinct_dsets)

            self.single_print("DONE SYNCING - NOW LOGGING")
        return logs


# %%
YAML_CONFIG = Path("/mnt/home/nsiegenheim/DISCO/config/validation.yaml")
CKPT_PATH = Path("/mnt/home/rmorel/ceph/logs/genphy/share_torchdiffeq_v4/training_checkpoints/ckpt.tar_epoch110")

# %%
params = YParams(YAML_CONFIG)

params["debug"] = False
params["use_ddp"] = False
params["start_epoch"] = 0
params["checkpoint_path"] = CKPT_PATH
params["best_checkpoint_path"] = CKPT_PATH
params["resuming"] = True
params["log_to_wandb"] = False
params["log_to_screen"] = True
params["num_workers"] = 0  # No need for multiple workers in validation

validator = Validator(params, global_rank=0, local_rank=0, device=device)

# %%
validator.validate(cutoff=1)  # Adjust cutoff as needed for validation

# %%

import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from models.FlexCast import Model
from data_provider.data_factory import data_provider

CONTEXT_LEN = 4096
PATCH_LEN = 16
SEQ_LEN = 3072
SEQ_LEN_MAP = {"globaltemp": 2048}
PRED_LENS = [96, 192, 336, 720]

DATASETS = {
    "ETTh1": {"data": "ETTh1", "root_path": "./dataset/ETT-small/", "data_path": "ETTh1.csv", "features": "M"},
    "ETTh2": {"data": "ETTh2", "root_path": "./dataset/ETT-small/", "data_path": "ETTh2.csv", "features": "M"},
    "ETTm1": {"data": "ETTm1", "root_path": "./dataset/ETT-small/", "data_path": "ETTm1.csv", "features": "M"},
    "ETTm2": {"data": "ETTm2", "root_path": "./dataset/ETT-small/", "data_path": "ETTm2.csv", "features": "M"},
    "weather": {"data": "custom", "root_path": "./dataset/weather/", "data_path": "weather.csv", "features": "M"},
    "globaltemp": {"data": "globaltemp", "root_path": "./dataset/globaltemp/", "data_path": "global_temp_test.jsonl", "features": "S"},
}


class Exp_Inference:
    def __init__(self, checkpoint_path, device="cuda"):
        self.logger = logging.getLogger("FlexCast_Inference")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self.logger.addHandler(sh)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model, self.num_quantile, self.quantile_levels = self._load_model()
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model params: {n_params/1e6:.1f}M")

    def _load_model(self):
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        config = ckpt["config"]
        state_dict = ckpt.get("state_dict", ckpt)

        configs = SimpleNamespace(
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            e_layers=config["e_layers"],
            d_ff=config.get("d_ff", config["d_model"] * 4),
            patch_len=config.get("patch_len", 16),
            stride=config.get("stride", config.get("patch_len", 16)),
            dropout=config.get("dropout", 0.1),
            factor=config.get("factor", 1),
            activation=config.get("activation", "gelu"),
            num_quantile=config.get("num_quantile", 99),
        )
        num_quantile = configs.num_quantile
        quantile_levels = [i / (num_quantile + 1) for i in range(1, num_quantile + 1)]

        model = Model(configs).float()
        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model, num_quantile, quantile_levels

    def _point_forecast(self, q_pred):
        ql = torch.tensor(self.quantile_levels, device=q_pred.device)
        w = 0.5 - (0.5 - ql).abs()
        w = w / w.sum()
        return (q_pred * w).sum(dim=-1)

    def _effective_lengths(self, x):
        T = x.shape[1]
        nan_mask = torch.isnan(x[:, :, 0])
        has_value = ~nan_mask
        idx = torch.arange(T, device=x.device).unsqueeze(0).expand_as(has_value)
        idx_valid = idx.clone()
        idx_valid[~has_value] = 0
        return idx_valid.max(dim=1).values + 1

    def _split_nan_masks(self, x):
        B, T, N = x.shape
        nan_mask = torch.isnan(x)
        eff_lens = self._effective_lengths(x)
        arange = torch.arange(T, device=x.device).unsqueeze(0).unsqueeze(2)
        pad_mask = (arange >= eff_lens.unsqueeze(1).unsqueeze(2)).expand_as(nan_mask)
        miss_mask = nan_mask & ~pad_mask
        return pad_mask, miss_mask

    def evaluate(self, dataset_names=None, pred_lens=None, batch_size=32):
        if dataset_names is None:
            dataset_names = list(DATASETS.keys())
        if pred_lens is None:
            pred_lens = PRED_LENS

        results = {}
        for ds_name in dataset_names:
            if ds_name not in DATASETS:
                self.logger.warning(f"Unknown dataset: {ds_name}, skipping.")
                continue
            ds_cfg = DATASETS[ds_name]
            self.logger.info(f"=== Evaluating on {ds_name} ===")

            effective_seq_len = SEQ_LEN_MAP.get(ds_name, SEQ_LEN)
            for pred_len in pred_lens:
                T = effective_seq_len + pred_len
                args = SimpleNamespace(
                    data=ds_cfg["data"],
                    root_path=ds_cfg["root_path"],
                    data_path=ds_cfg["data_path"],
                    features=ds_cfg["features"],
                    target="OT",
                    seq_len=T,
                    label_len=0,
                    pred_len=0,
                    batch_size=batch_size,
                    num_workers=4,
                )
                _, test_loader = data_provider(args, flag="test")

                mse_list, mae_list = [], []
                with torch.no_grad():
                    for batch_x in test_loader:
                        if isinstance(batch_x, (list, tuple)):
                            batch_x = batch_x[0]
                        x_raw = batch_x[:, -T:, :].float().to(self.device)
                        B, _, N = x_raw.shape

                        x = x_raw
                        mask = torch.ones(B, T, N, device=self.device)
                        mask[:, effective_seq_len:effective_seq_len + pred_len, :] = 0
                        pad_mask, miss_mask = self._split_nan_masks(x)

                        with torch.autocast('cuda', dtype=torch.bfloat16,
                                            enabled=(self.device.type == 'cuda')):
                            out = self.model(x, mask, pad_mask=pad_mask, miss_mask=miss_mask)

                        pred = self._point_forecast(out)
                        loss_mask = (mask == 0) & (~pad_mask) & (~miss_mask)
                        mse_val = F.mse_loss(pred[loss_mask], x[loss_mask])
                        mae_val = F.l1_loss(pred[loss_mask], x[loss_mask])
                        mse_list.append(mse_val.item())
                        mae_list.append(mae_val.item())

                avg_mse = float(np.mean(mse_list)) if mse_list else 0.0
                avg_mae = float(np.mean(mae_list)) if mae_list else 0.0
                key = f"{ds_name}_seq{effective_seq_len}_pred{pred_len}"
                results[key] = {"mse": avg_mse, "mae": avg_mae}
                self.logger.info(f"  {ds_name} seq={effective_seq_len} pred={pred_len}: MSE={avg_mse:.6f} MAE={avg_mae:.6f}")

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FlexCast Inference Evaluation")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/FlexCast.pth")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Datasets to evaluate. Default: all")
    parser.add_argument("--pred_lens", type=int, nargs="+", default=None,
                        help="Prediction lengths. Default: 96 192 336 720")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    exp = Exp_Inference(checkpoint_path=args.checkpoint, device=args.device)
    results = exp.evaluate(
        dataset_names=args.datasets,
        pred_lens=args.pred_lens,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for key, metrics in results.items():
        print(f"  {key}: MSE={metrics['mse']:.6f}  MAE={metrics['mae']:.6f}")


if __name__ == "__main__":
    main()

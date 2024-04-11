from typing import Any, List, Optional
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter

class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, alpha, beta,k, write_interval: str):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.k = k
        self.beta_tensor = torch.tensor(beta).unsqueeze(1).unsqueeze(1).T
        self.alpha_tensor = torch.tensor(alpha).unsqueeze(1).unsqueeze(1).T

    def write_on_batch_end(
            self, trainer, pl_module: pl.LightningModule, prediction: Any, batch_indices: List[int], batch: Any,
            batch_idx: int, dataloader_idx: int):
        sampled_rankings, click_scores, doc_feats = prediction
        batch_size = click_scores.shape[0]
        num_rankings = click_scores.shape[1]
        clicks = torch.bernoulli(self.alpha_tensor * click_scores[:, :, :self.k] + self.beta_tensor).detach()
        feat = torch.stack([torch.index_select(doc_feats[i, :, :], 0, sampled_rankings[i, :, :self.k].flatten()) for i in range(batch_size)]).detach()
        batch_ranking = torch.arange(self.k).unsqueeze(1).T.expand(num_rankings * batch_size, -1).flatten().detach()
        print("LOGGING DATA")
        torch.save(clicks, os.path.join(self.output_dir, f"click_{dataloader_idx}_{batch_idx}.pt"))
        torch.save(feat, os.path.join(self.output_dir, f"click_{dataloader_idx}_{batch_idx}.pt"))
        torch.save(batch_ranking, os.path.join(self.output_dir, f"click_{dataloader_idx}_{batch_idx}.pt"))

    def write_on_epoch_end(
            self, trainer, pl_module: pl.LightningModule, predictions: List[Any], batch_indices: List[Any]):
        print("LOGGING DATA")
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))      
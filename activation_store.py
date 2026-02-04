from abc import ABC, abstractmethod

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens.hook_points import HookedRootModule

from config import EncoderConfig, SAEConfig, TranscoderConfig


class BaseActivationsStore(ABC):
    """Base class for activation collection with shared token handling."""

    def __init__(self, model: HookedRootModule, cfg: EncoderConfig):
        self.model = model
        self.cfg = cfg
        self.dataset = iter(load_dataset(cfg.dataset_path, split="train", streaming=True))
        self.context_size = min(cfg.seq_len, model.cfg.n_ctx)
        self.model_batch_size = cfg.model_batch_size
        self.device = cfg.device
        self.num_batches_in_buffer = cfg.num_batches_in_buffer
        self.tokens_column = self._get_tokens_column()
        self.tokenizer = model.tokenizer

    def _get_tokens_column(self) -> str:
        sample = next(self.dataset)
        if "tokens" in sample:
            return "tokens"
        elif "input_ids" in sample:
            return "input_ids"
        elif "text" in sample:
            return "text"
        else:
            raise ValueError("Dataset must have a 'tokens', 'input_ids', or 'text' column.")

    def get_batch_tokens(self) -> torch.Tensor:
        all_tokens = []
        while len(all_tokens) < self.model_batch_size * self.context_size:
            batch = next(self.dataset)
            if self.tokens_column == "text":
                tokens = self.model.to_tokens(
                    batch["text"], truncate=True, move_to_device=True, prepend_bos=True
                ).squeeze(0)
            else:
                tokens = batch[self.tokens_column]
            if isinstance(tokens, torch.Tensor):
                all_tokens.extend(tokens.tolist())
            else:
                all_tokens.extend(tokens)
        token_tensor = torch.tensor(
            all_tokens, dtype=torch.long, device=self.device
        )[: self.model_batch_size * self.context_size]
        return token_tensor.view(self.model_batch_size, self.context_size)

    @abstractmethod
    def get_activations(self, batch_tokens: torch.Tensor):
        """Collect activations from model. Override in subclass."""
        pass

    @abstractmethod
    def _fill_buffer(self):
        """Fill activation buffer(s). Override in subclass."""
        pass

    @abstractmethod
    def _get_dataloader(self) -> DataLoader:
        """Create dataloader from buffer(s). Override in subclass."""
        pass

    @abstractmethod
    def next_batch(self):
        """Get next batch of activations. Override in subclass."""
        pass


class ActivationsStore(BaseActivationsStore):
    """Collects activations from a single hook point (for SAE training)."""

    def __init__(self, model: HookedRootModule, cfg: SAEConfig):
        super().__init__(model, cfg)
        self.cfg: SAEConfig = cfg
        self.hook_point = cfg.hook_point

    def get_activations(
        self, batch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get activations. Returns (acts, acts) tuple for API consistency with transcoder."""
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.cfg.layer + 1,
            )
        acts = cache[self.hook_point]
        return acts, acts

    def _fill_buffer(self) -> torch.Tensor:
        all_activations = []
        for _ in range(self.num_batches_in_buffer):
            batch_tokens = self.get_batch_tokens()
            acts, _ = self.get_activations(batch_tokens)
            all_activations.append(acts.reshape(-1, self.cfg.act_size))
        return torch.cat(all_activations, dim=0)

    def _get_dataloader(self) -> DataLoader:
        return DataLoader(
            TensorDataset(self.activation_buffer),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (batch, batch) tuple for API consistency with transcoder."""
        try:
            batch = next(self.dataloader_iter)[0]
        except (StopIteration, AttributeError):
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)[0]
        return batch, batch


class TranscoderActivationsStore(BaseActivationsStore):
    """Collects paired (input, target) activations for transcoder training."""

    def __init__(self, model: HookedRootModule, cfg: TranscoderConfig):
        super().__init__(model, cfg)
        self.cfg: TranscoderConfig = cfg
        self.input_hook_point = cfg.input_hook_point
        self.output_hook_point = cfg.output_hook_point

    def get_activations(
        self, batch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get both input and output activations."""
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.input_hook_point, self.output_hook_point],
                stop_at_layer=self.cfg.output_layer + 1,
            )
        return cache[self.input_hook_point], cache[self.output_hook_point]

    def _fill_buffer(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_input_activations = []
        all_output_activations = []
        for _ in range(self.num_batches_in_buffer):
            batch_tokens = self.get_batch_tokens()
            input_acts, output_acts = self.get_activations(batch_tokens)
            all_input_activations.append(input_acts.reshape(-1, self.cfg.input_size))
            all_output_activations.append(output_acts.reshape(-1, self.cfg.output_size))
        return (
            torch.cat(all_input_activations, dim=0),
            torch.cat(all_output_activations, dim=0),
        )

    def _get_dataloader(self) -> DataLoader:
        return DataLoader(
            TensorDataset(self.input_buffer, self.output_buffer),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (input_batch, target_batch)."""
        try:
            batch = next(self.dataloader_iter)
            return batch[0], batch[1]
        except (StopIteration, AttributeError):
            self.input_buffer, self.output_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)
            return batch[0], batch[1]

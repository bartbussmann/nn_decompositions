"""Multi-layer activation store for Cross-Layer Transcoders.

Captures activations from multiple input/output modules simultaneously.
Each forward pass through the model collects (input, output) pairs at every
layer, yielding aligned batches for CLT training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from activation_store import DataConfig


class MultiLayerActivationsStore:
    """Activation store that hooks into multiple (input, output) module pairs.

    Args:
        model: The transformer model to hook into.
        input_modules: Modules whose outputs are the CLT encoder inputs (one per layer).
        output_modules: Modules whose outputs are the CLT reconstruction targets (one per layer).
        data_config: Configuration for data loading.
        input_size: Per-layer input activation dimension.
        output_size: Per-layer output activation dimension.
    """

    def __init__(
        self,
        model: nn.Module,
        input_modules: list[nn.Module],
        output_modules: list[nn.Module],
        data_config: DataConfig,
        input_size: int,
        output_size: int,
    ):
        assert len(input_modules) == len(output_modules)
        self.model = model
        self.output_modules = output_modules
        self.data_config = data_config
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = len(input_modules)

        # Activation storage (written by hooks during forward pass)
        self._input_acts: list[torch.Tensor | None] = [None] * self.n_layers
        self._output_acts: list[torch.Tensor | None] = [None] * self.n_layers

        # Register hooks
        for i, mod in enumerate(input_modules):
            mod.register_forward_hook(self._make_hook(self._input_acts, i))
        for i, mod in enumerate(output_modules):
            mod.register_forward_hook(self._make_hook(self._output_acts, i))

        # Setup dataset
        self._setup_dataset()

        # Buffer state (initialized on first next_batch call)
        self.input_buffers: list[torch.Tensor] | None = None
        self.output_buffers: list[torch.Tensor] | None = None
        self.dataloader: DataLoader | None = None
        self.dataloader_iter: iter | None = None

    @staticmethod
    def _make_hook(storage: list, idx: int):
        def hook(module, input, output):
            storage[idx] = output.detach()
        return hook

    def _setup_dataset(self):
        from datasets import load_dataset

        cfg = self.data_config
        self.tokenizer = cfg.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset = load_dataset(cfg.dataset_name, split="train", streaming=cfg.streaming)
        self.dataset = self.dataset.shuffle(seed=cfg.seed, buffer_size=10000)
        self.data_iter = iter(self.dataset)

    def get_batch_tokens(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get a batch of tokens. Returns (input_ids, attention_mask)."""
        if self.data_config.is_tokenized:
            return self._get_batch_tokens_pretokenized()
        return self._get_batch_tokens_text()

    def _get_batch_tokens_text(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.data_config
        texts = []
        for _ in range(cfg.model_batch_size):
            try:
                sample = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataset)
                sample = next(self.data_iter)
            text = sample[cfg.text_column]
            if cfg.lowercase:
                text = text.lower()
            texts.append(text)

        tokens = self.tokenizer(
            texts,
            truncation=True,
            max_length=cfg.seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        return (
            tokens["input_ids"].to(cfg.device),
            tokens["attention_mask"].to(cfg.device),
        )

    def _get_batch_tokens_pretokenized(self) -> tuple[torch.Tensor, None]:
        cfg = self.data_config
        batch_ids = []
        for _ in range(cfg.model_batch_size):
            try:
                sample = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataset)
                sample = next(self.data_iter)
            ids = sample[cfg.token_column]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            batch_ids.append(ids[: cfg.seq_len])
        return torch.stack(batch_ids).to(cfg.device), None

    @torch.no_grad()
    def get_activations(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run forward pass and return per-layer (inputs, outputs) activations."""
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        self.model(input_ids, **kwargs)
        assert all(a is not None for a in self._input_acts)
        assert all(a is not None for a in self._output_acts)
        return list(self._input_acts), list(self._output_acts)

    def _fill_buffer(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Fill buffers with activations from multiple forward passes."""
        to_cpu = self.data_config.buffer_on_cpu
        all_inputs = [[] for _ in range(self.n_layers)]
        all_outputs = [[] for _ in range(self.n_layers)]

        for _ in range(self.data_config.num_batches_in_buffer):
            input_ids, attention_mask = self.get_batch_tokens()
            input_acts, output_acts = self.get_activations(input_ids, attention_mask)

            for i in range(self.n_layers):
                if attention_mask is not None:
                    mask = attention_mask.bool().unsqueeze(-1)
                    inp = input_acts[i][mask.expand_as(input_acts[i])].reshape(-1, self.input_size)
                    out = output_acts[i][mask.expand_as(output_acts[i])].reshape(-1, self.output_size)
                else:
                    inp = input_acts[i].reshape(-1, self.input_size)
                    out = output_acts[i].reshape(-1, self.output_size)
                all_inputs[i].append(inp.cpu() if to_cpu else inp)
                all_outputs[i].append(out.cpu() if to_cpu else out)

        return (
            [torch.cat(buf) for buf in all_inputs],
            [torch.cat(buf) for buf in all_outputs],
        )

    def _get_dataloader(self) -> DataLoader:
        """Create dataloader from buffers. All layers are aligned by token position."""
        all_tensors = self.input_buffers + self.output_buffers
        return DataLoader(
            TensorDataset(*all_tensors),
            batch_size=self.data_config.train_batch_size,
            shuffle=True,
        )

    def next_batch(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Get next aligned batch of per-layer (inputs, targets) activations."""
        try:
            batch = next(self.dataloader_iter)
        except (StopIteration, TypeError):
            self.input_buffers, self.output_buffers = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)

        n = self.n_layers
        inputs = list(batch[:n])
        targets = list(batch[n:])
        if self.data_config.buffer_on_cpu:
            device = self.data_config.device
            inputs = [x.to(device) for x in inputs]
            targets = [x.to(device) for x in targets]
        return inputs, targets

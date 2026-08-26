import torch
from torch.utils.data import Dataset

class ConditionPairDataset(Dataset):
    def __init__(
        self,
        X_emul: torch.Tensor,       # (N, C, H, W) : Training data from emulator
        X_real: torch.Tensor = None,# (N, 1, H, W) or (N, H, W) : Training data from computer models (if needed)
        train: bool = True,
        c_train_ratio: float = 0.6,  
        base_seed: int = 12345,
        deterministic_val: bool = True,
    ):
        super().__init__()
        self.X_emul = X_emul
        self.X_real = X_real
        self.train = train
        self.base_seed = int(base_seed)
        self.deterministic_val = bool(deterministic_val)

        # Dimensions: (N, C, H, W)
        assert self.X_emul.ndim == 4, f"Expected X_emul to be 4D (N,C,H,W). Got {self.X_emul.shape}."
        self.N = self.X_emul.shape[0]
        self.C = self.X_emul.shape[1]
        
        if self.X_real is not None:
            assert self.X_real.shape[0] == self.N, f"X_real N({self.X_real.shape[0]}) must match X_emul N({self.N})"

        split_point = int(self.C * c_train_ratio)
        if train:
            self.c_indices = list(range(0, split_point))
        else:
            self.c_indices = list(range(split_point, self.C))

        self.num_c = len(self.c_indices)
        if self.num_c < 2:
            mode_str = "Train" if train else "Validation"
            raise ValueError(
                f"[{mode_str}] Not enough conditions to form pairs! Need at least 2, but got {self.num_c}."
            )

        self.use_real = (self.X_real is not None)
        self.num_items = self.N * 4 if self.use_real else self.N * 2

        if (not train) and deterministic_val:
            self._gen = torch.Generator()

    def __len__(self):
        return self.num_items

    def _randint(self, low: int, high: int, gen: torch.Generator) -> int:
        return int(torch.randint(low, high, (1,), generator=gen).item())
    
    def _pick_c_index(self, gen=None):
        if gen is None:
            idx = int(torch.randint(0, self.num_c, (1,)).item())
        else:
            idx = self._randint(0, self.num_c, gen)
        return self.c_indices[idx]

    def __getitem__(self, idx: int):
        if (not self.train) and self.deterministic_val:
            self._gen.manual_seed(self.base_seed + int(idx))
            gen = self._gen
        else:
            gen = None

        # case = 0: Computer model-Emul Same, 1: Computer model-Emul Diff, 2: Emul-Emul Same, 3: Emul-Emul Diff
        if self.use_real:
            case = idx // self.N
            i = idx % self.N
        else:
            case = (idx // self.N) + 2 
            i = idx % self.N

        # ==========================================
        # CASE 0: Computer model vs Emulator (Same)
        # ==========================================
        if case == 0:
            c1 = self._pick_c_index(gen)
            img1 = self.X_real[i]
            img2 = self.X_emul[i, c1]
            label = 1.0

        # ==========================================
        # CASE 1: Computer model vs Emulator (Diff)
        # ==========================================
        elif case == 1:
            offset = self._randint(1, self.N, gen) if gen else int(torch.randint(1, self.N, (1,)).item())
            j = (i + offset) % self.N
            c2 = self._pick_c_index(gen)
            
            img1 = self.X_real[i]
            img2 = self.X_emul[j, c2]
            label = 0.0

        # ==========================================
        # CASE 2: Emulator vs Emulator (Same)
        # ==========================================
        elif case == 2:
            if gen is None:
                perm = torch.randperm(self.num_c)[:2]
            else:
                perm = torch.randperm(self.num_c, generator=gen)[:2]
            
            c1, c2 = self.c_indices[perm[0]], self.c_indices[perm[1]]
            img1 = self.X_emul[i, c1]
            img2 = self.X_emul[i, c2]
            label = 1.0

        # ==========================================
        # CASE 3: Emulator vs Emulator (Diff)
        # ==========================================
        else: # case == 3
            offset = self._randint(1, self.N, gen) if gen else int(torch.randint(1, self.N, (1,)).item())
            j = (i + offset) % self.N
            
            c1 = self._pick_c_index(gen)
            c2 = self._pick_c_index(gen)
            
            img1 = self.X_emul[i, c1]
            img2 = self.X_emul[j, c2]
            label = 0.0

        if img1.ndim == 2:
            img1 = img1.unsqueeze(0)
        if img2.ndim == 2:
            img2 = img2.unsqueeze(0)

        img1 = img1.to(dtype=torch.float32)
        img2 = img2.to(dtype=torch.float32)


        return img1, img2, torch.tensor(label, dtype=torch.float32)
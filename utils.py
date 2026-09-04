import torch
import shutil
import os
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class AverageMeter:
    def __init__(self, name: str = 'Metric', fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[float]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size)
            res.append(acc.item())
        
        return res

def save_checkpoint(state: Dict[str, Any], is_best: bool, filename: str = 'checkpoint.pth', folder: str = 'saved_models') -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    filepath = os.path.join(folder, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(folder, 'model_best.pth')
        shutil.copyfile(filepath, best_path)
        logger.info(f"Nova melhor performance salva em: {best_path}")

def load_checkpoint(model: torch.nn.Module, 
                    optimizer: Optional[torch.optim.Optimizer] = None, 
                    filepath: str = '') -> int:
    
    if not os.path.isfile(filepath):
        logger.error(f"Checkpoint não encontrado em: {filepath}")
        return 0

    logger.info(f"Carregando checkpoint: {filepath}")
    checkpoint = torch.load(filepath)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    start_epoch = checkpoint.get('epoch', 0)
    
    logger.info(f"Carregado com sucesso (Epoch {start_epoch})")
    return start_epoch

def save_features(X, y, checkpoint_dir, epoch_num, loader, train = False):
    features_dir = "features/train" if train else "features/test"
    os.makedirs(os.path.join(checkpoint_dir, features_dir), exist_ok=True)
    np.save(os.path.join(checkpoint_dir, features_dir, f"features+epoch{epoch_num}.npy"), X)
    np.save(os.path.join(checkpoint_dir, features_dir, f"labels+epoch{epoch_num}.npy"), y)

    data = loader.dataset.dataset.class_to_idx
    with open(os.path.join(checkpoint_dir, features_dir, f"labels+epoch{epoch_num}.json"), "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


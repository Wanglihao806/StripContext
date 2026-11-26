import torch
import torch.nn.functional as F

def batch_indices(count, batch_size, generator=None, shuffle=False):
    if batch_size < 2:
        raise ValueError("Training microbatch must be >=2 for FDSNet BatchNorm")
    order = torch.randperm(count, generator=generator).tolist() if shuffle else list(range(count))
    batches = [order[i:i + batch_size] for i in range(0, count, batch_size)]
    if len(batches[-1]) == 1:
        # 4+1 -> 3+2; preserve every sample without increasing the memory cap.
        if batch_size == 2:
            raise ValueError("Batch size 2 cannot redistribute a singleton without exceeding memory cap")
        batches[-1].insert(0, batches[-2].pop())
    return batches

def output_list(outputs):
    return list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]

def ce_dice(logits, labels):
    # Float32 reductions also under BF16 autocast. Exactly the author's two-class
    # macro Dice: sums over B,H,W; epsilon only in denominator (empty class=0).
    logits = logits.float()
    probabilities = logits.softmax(1)
    target = F.one_hot(labels, num_classes=2).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probabilities * target).sum(dims)
    denominator = (probabilities + target).sum(dims)
    dice = 1 - (2 * intersection / (denominator + 1e-7)).mean()
    # Equivalent unweighted CE, avoiding nondeterministic CUDA NLLLoss2d
    # reduction. Validate against PyTorch CE values AND gradients in preflight.
    ce = -logits.log_softmax(1).gather(1, labels[:, None]).mean()
    return 0.5 * (ce + dice)

def model_loss(name, outputs, labels):
    heads = output_list(outputs)
    weights = [1.0, 0.2, 0.2, 0.2] if name == "fastsdnet" else [1.0]
    if len(heads) != len(weights):
        raise RuntimeError("Unexpected supervision heads for " + name)
    return sum(w * ce_dice(head, labels) for w, head in zip(weights, heads))

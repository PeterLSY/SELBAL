"""Stage 4 -- mutation, implemented as a short finetune of each offspring.

Carried over from evolutionary_wavg_training.py:705-831. There is no separate
weight-perturbation step in SESiL: "mutation" IS this finetune.

Note the schedule here is the legacy one (LinearLR stepped per batch with
total_iters = one epoch's worth of batches, wavg:769-771 / :802). It is kept
as-is on purpose -- changing it would change evolution results, and this module
must stay behaviourally identical to the reference scripts. The pretraining
stage, which is new code with no baseline to match, uses a proper schedule.

`updates_per_epoch` is the one deviation, and it defaults to 0 = off here so
the legacy path is bit-identical unless a caller opts in. When set, an epoch is
that many updates instead of a full pass, and the LinearLR decay window follows
it, so the schedule keeps its legacy SHAPE (decay to the floor within epoch 1).
"""

from collections import defaultdict

import torch
from tqdm.auto import tqdm

from utils import CrossEntropyLoss, EarlyStopper, GradScaler, autocast, get_device

from .data import cycle_batches, updates_for


def evaluate_logits(
    model,
    test_loader,
    return_confusion=False,
    use_flip_aug=False,
    remap_class_idxs=None,
    class_idxs=None,
    eval_mask=None,
):
    model.eval()
    correct = 0
    total = 0
    totals = defaultdict(lambda: 0)
    corrects = defaultdict(lambda: 0)
    loss_fn = CrossEntropyLoss()
    device = next(iter(model.parameters())).device
    total_loss = 0
    total_iter = len(test_loader)

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, "Evaluating classification model"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if isinstance(outputs, list):
                outputs = outputs[0]

            if use_flip_aug:
                flip_outputs = model(torch.flip(inputs, (3,)))
                if isinstance(flip_outputs, list):
                    flip_outputs = flip_outputs[0]
                outputs += flip_outputs

            if eval_mask is not None:
                outputs[:, eval_mask == 0] = -torch.inf

            pred = outputs.argmax(dim=-1)
            total += pred.shape[0]

            if remap_class_idxs is not None:
                remapped_labels = remap_class_idxs[labels]
            else:
                remapped_labels = labels

            loss = loss_fn(outputs, remapped_labels)
            total_loss += loss

            for gt, p in zip(remapped_labels, pred):
                gt, p = gt.item(), p.item()
                totals[gt] += 1
                if gt == p:
                    correct += 1
                    corrects[gt] += 1

    num_classes = max(totals) + 1 if totals else 0
    totals = [totals[i] for i in range(num_classes)]
    corrects = [corrects[i] for i in range(num_classes)]

    if return_confusion:
        acc_per_class = [(c / t if t > 0 else 0.0) for c, t in zip(corrects, totals)]
        return correct / sum(totals), acc_per_class
    return correct / total if total > 0 else 0.0


def train_logits(model, train_loader, test_loader, epochs=2, lr=0.001,
                 remap_class_idxs=None, updates_per_epoch=0):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    ne_iters = updates_for(train_loader, updates_per_epoch)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=1e-7, total_iters=ne_iters
    )
    early_stopper = EarlyStopper(patience=epochs, min_delta=0.0001)

    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(reduction="mean")
    device = get_device(model)
    acc = 0.0
    best_acc = 0.0
    best_epoch = 0
    best_sd = None

    batches = cycle_batches(train_loader)
    pbar = tqdm(range(epochs), desc=f"Training, prev acc: {acc}: ")
    for epoch in pbar:
        model.train()
        for _ in range(ne_iters):
            inputs, labels = next(batches)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(inputs.to(device))
                if isinstance(logits, list):
                    logits = logits[0]
                if remap_class_idxs is not None:
                    remapped_labels = remap_class_idxs[labels].to(device)
                else:
                    remapped_labels = labels.to(device)

                loss = loss_fn(logits, remapped_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        acc = evaluate_logits(model, test_loader, remap_class_idxs=remap_class_idxs)
        if acc > best_acc:
            best_sd = model.state_dict()
            best_acc = acc
            best_epoch = epoch

        if early_stopper.early_stop(acc):
            print(
                f"Stopping at Epoch: {epoch}. Best Accuracy {best_acc}, "
                f"Achieved at Epoch {best_epoch}"
            )
            break
        pbar.set_description(f"Training, prev acc: {acc}: ")

    if best_sd is not None:
        model.load_state_dict(best_sd)
    acc = evaluate_logits(model, test_loader, remap_class_idxs=remap_class_idxs)
    print("Acc at Best Model: {}".format(acc))
    return model, best_acc


def finetune_merged_model(model, trainloader, testloader, epochs=2, lr=0.001,
                          updates_per_epoch=0):
    model.train()
    model, final_acc = train_logits(
        model=model,
        train_loader=trainloader,
        test_loader=testloader,
        epochs=epochs,
        lr=lr,
        updates_per_epoch=updates_per_epoch,
    )
    return model, final_acc

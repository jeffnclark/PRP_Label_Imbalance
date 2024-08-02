import time
import torch
import random
from sklearn.metrics import f1_score, precision_score, recall_score

from helpers import list_of_distances, make_one_hot


from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import wandb

# Function to ensure deterministic behavior


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    # Seed all workers with the same seed
    seed = 42
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    all_targets = []
    all_predictions = []

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(
                    model.module.prototype_class_identity[:, label]).cuda()

                inverted_distances, _ = torch.max(
                    (max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) *
                              prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(
                    max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class,
                              dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

                if use_l1_mask:
                    l1_mask = 1 - \
                        torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end - start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(
            total_avg_separation_cost / n_batches))

    # Calculate F1 score, precision, and recall for each class
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    class_f1_scores = f1_score(all_targets, all_predictions, average=None)
    class_precisions = precision_score(
        all_targets, all_predictions, average=None)
    class_recalls = recall_score(all_targets, all_predictions, average=None)

    for class_idx, (f1, precision, recall) in enumerate(zip(class_f1_scores, class_precisions, class_recalls)):
        log('\tClass {0} - F1: {1:.4f}, Precision: {2:.4f}, Recall: {3:.4f}'.format(
            class_idx, f1, precision, recall))
        if optimizer is not None:
            wandb.log({f'f1_train_class_{class_idx}': f1})
        else:
            wandb.log({f'f1_val_class_{class_idx}': f1})

    overall_f1 = f1_score(all_targets, all_predictions, average='weighted')
    overall_precision = precision_score(
        all_targets, all_predictions, average='weighted')
    overall_recall = recall_score(
        all_targets, all_predictions, average='weighted')

    log('\tOverall - F1: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}'.format(
        overall_f1, overall_precision, overall_recall))

    log('\tl1: \t\t{0}'.format(
        model.module.last_layer.weight.norm(p=1).item()))

    p = model.module.prototype_vectors.view(
        model.module.num_prototypes, -1).cpu()

    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    # log with W&B
    if optimizer is not None:
        wandb.log({'p dist pair train': p_avg_pair_dist.item()})
        wandb.log({'l1 train': model.module.last_layer.weight.norm(p=1).item()})
        wandb.log({'cross ent train': total_cross_entropy / n_batches})
        if class_specific:
            wandb.log({'separation train': (total_separation_cost / n_batches)})
            wandb.log({'avg separation train': (
                total_avg_separation_cost / n_batches)})

    else:
        # wandb.log({f'p dist pair val': p_avg_pair_dist.item()})
        # wandb.log({f'l1 val': model.module.last_layer.weight.norm(p=1).item()})
        wandb.log({'cross ent val': total_cross_entropy / n_batches})
        wandb.log({'separation val': (total_separation_cost / n_batches)})
        wandb.log({'avg separation val': (
            total_avg_separation_cost / n_batches)})

    return overall_f1


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert (optimizer is not None)

    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\tjoint')

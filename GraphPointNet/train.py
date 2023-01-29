# parts of code adapted from https://github.dev/yanx27/Pointnet_Pointnet2_pytorch/blob/master/train_partseg.py

from genericpath import exists
import os
import torch
import wandb
import yaml
from munch import Munch
from ShapeNetDataLoader import PartNormalDataset
from tqdm import tqdm
import numpy as np
import provider
import logging
from datetime import datetime

#CONFIG
with open('GraphPointNet/config.yaml', 'rt', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = Munch.fromDict(config)

if config.MODEL.NAME == "PN":
    from models.PN import get_model, get_loss
elif config.MODEL.NAME == "GPN":
    from models.GPN import get_model, get_loss
elif config.MODEL.NAME == "PN2":
    from models.PN2 import get_model, get_loss
elif config.MODEL.NAME == "GPN2":
    from models.GPN2 import get_model, get_loss
else:
    raise ValueError("Model not found")

device = config.DEVICE

batch_size = config.DATALOADER.BATCH_SIZE
num_workers = config.DATALOADER.NUM_WORKERS
cache = config.DATALOADER.CACHE

# criterion = config.SOLVER.CRITERION
lr = config.SOLVER.LR
epochs = config.SOLVER.EPOCHS

save_checkpoint = config.TRAIN.SAVE_CHECKPOINT
save_checkpoint_path = config.TRAIN.SAVE_CHECKPOINT_PATH
load_checkpoint = config.TRAIN.LOAD_CHECKPOINT
load_checkpoint_path = config.TRAIN.LOAD_CHECKPOINT_PATH
wandb_log = config.TRAIN.WANDB_LOG
project_name = config.TRAIN.WANDB_PROJECT
normal_channel = config.MODEL.NORMAL_CHANNEL

npoints = config.DATASET.NUMBER_OF_POINTS

early_stopping = config.TRAIN.EARLY_STOPPING
patience = config.TRAIN.PATIENCE
restore_best_weights = config.TRAIN.RESTORE_BEST_WEIGHTS

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

num_classes = 16
part_num = 50

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.to(device)
    return new_y

def main():
    logger.info("Initiate training...")
    #============LOAD DATA===============
    #------------------------------------
    logger.info("Load data...")
    data_train = PartNormalDataset(split="train", npoints=npoints,normal_channel=normal_channel, config=config, cache=cache)

    # TRAIN DATA
    dl_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=data_train.my_collate)

    # VAL DATA
    data_val = PartNormalDataset(split="val", npoints=npoints, normal_channel=normal_channel, config=config, cache=cache)
    dl_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=data_val.my_collate)

    #============MODEL===============
    #--------------------------------
    model = get_model(config = config, part_num=part_num)
    model.to(device)

    #============CRITERION===============
    #------------------------------------
    criterion = get_loss().to(device)

    #============OPTIMIZER===============
    #------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #============WRITER==============
    if wandb_log:
        os_start_method = 'spawn' if os.name == 'nt' else 'fork'
        run_datetime = datetime.now().isoformat().split('.')[0]
        wandb.init(
        project=project_name,
        name=run_datetime, 
        settings=wandb.Settings(start_method=os_start_method),
        config = config
        )

    #============LOAD================
    #--------------------------------
    start_epoch = 0
    if (load_checkpoint):
        if exists(load_checkpoint_path):
            checkpoint = torch.load(load_checkpoint_path)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("Checkpoint not found")


    #============TRAIN===============
    #--------------------------------
    logger.info("Training...")
    training_loop(
        model,
        dl_train,
        dl_val,
        criterion,
        optimizer,
        epochs,
        start_epoch,
        save_checkpoint,
        save_checkpoint_path,
        wandb_log,
        device)
    logger.info("Training complete")


def training_loop(model, dl_train, dl_val, criterion, optimizer, epochs, start_epoch, save_checkpoint, save_checkpoint_path, wandb_log,  device):


    if early_stopping:
        max_loss_val = 0
        patience_counter = 0

    for epoch in range(start_epoch, epochs):
        train_accuracy, loss_train = train(
            model,
            dl_train,
            criterion,
            optimizer,
            epoch,
            device,
            wandb_log)

        val_metrics, loss_val = validate(
            model,
            dl_val,
            criterion,
            device)

        if save_checkpoint:
            os.makedirs(save_checkpoint_path.rsplit("/", 1)[0], exist_ok=True)
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_checkpoint_path)

        lr =  optimizer.param_groups[0]['lr']

        logger.info(f'Epoch: {epoch} '
              f' Lr: {lr:.3E} '
              f'Loss_train: {loss_train:.3E} '
              f'Loss_val: {loss_val:.3E} '
              f'Train_accuracy = [{train_accuracy:.3E}]'
              f'Val_accuracy = [{val_metrics["accuracy"]:.3E}]]'
              f'Val_instance_avg_iou = [{val_metrics["inctance_avg_iou"]:.3E}]]'
              f'Val_class_avg_iou = [{val_metrics["class_avg_iou"]:.3E}]]'
              )

        if wandb_log:
            wandb.log({'Learning_Rate': lr, 
            'Loss_train': loss_train,
            'Loss_val': loss_val,
            'Train_accuracy': train_accuracy,
            'Validation_accuracy': val_metrics["accuracy"], 
            'Validation_class_avg_iou': val_metrics['class_avg_iou'], 
            'Validation_inctance_avg_iou': val_metrics['inctance_avg_iou'], 
            'Epoch': epoch})

        
        # Early stopping
        if early_stopping:
            if val_metrics["accuracy"] > max_loss_val:
                max_loss_val = val_metrics["accuracy"]
                patience_counter = 0
                if restore_best_weights:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        }, save_checkpoint_path.rsplit("/", 1)[0]+"/best_weights.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping: patience {patience} reached")
                    if restore_best_weights:
                        best_model = torch.load(save_checkpoint_path.rsplit("/", 1)[0]+"/best_weights.pth")
                        torch.save({
                            'epoch': best_model["epoch"],
                            'model_state_dict': best_model["model_state_dict"],
                            }, save_checkpoint_path)
                        print(f"Best model at epoch {best_model['epoch']} restored")
                    break
        
    # save model to wandb
    if wandb_log:
        if config.TRAIN.WANDB_SAVE_MODEL:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(save_checkpoint_path)
            wandb.log_artifact(artifact)
            logger.info("Model saved to wandb")
        # close wandb
        wandb.finish()


def train(model, dl_train, criterion, optimizer, epoch, device, wandb_log):

    mean_correct = []
    losses = []
    model.train()
    lr = max(config.SOLVER.LR * (config.SOLVER.LR_DECAY ** (epoch // config.SOLVER.LR_STEP_SIZE)), config.SOLVER.MIN_LR)
    
    logger.info(f'Learning rate: {lr:.8f}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for idx_batch, data in tqdm(enumerate(dl_train), total=len(dl_train), smoothing=0.9):
        optimizer.zero_grad()
        points, label, target, edge_list = data["points"], data["label"], data["target"], data["edge_list"]
        points = points.data.numpy()
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        points, label, target, edge_list = points.float().to(device), label.long().to(device), target.long().to(device), edge_list.to(device)
        points = points.transpose(2, 1)
        if config.MODEL.NAME == "GPN" or config.MODEL.NAME == "GPN2":
            seg_pred, trans_feat = model(points, to_categorical(label, num_classes), edge_list)
        elif config.MODEL.NAME == "PN" or config.MODEL.NAME == "PN2":
            seg_pred, trans_feat = model(points, to_categorical(label, num_classes))

        seg_pred = seg_pred.contiguous().view(-1, part_num)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]

        correct = pred_choice.eq(target.data).cpu().sum()
        mean_correct.append(correct.item() / (batch_size * npoints))
        loss = criterion(seg_pred, target, trans_feat)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if wandb_log:
            global_step = epoch * len(dl_train) + idx_batch
            wandb.log({'Loss': loss, 'Accuracy': np.mean(mean_correct), 'Global_step': global_step})

    train_instance_acc = np.mean(mean_correct)
    logger.info('Train accuracy is: %.5f' % train_instance_acc)
    return train_instance_acc, np.mean(losses)




def validate(model, dl_val, criterion, device):
    with torch.no_grad():
        test_metrics = {}
        losses = []
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(part_num)]
        total_correct_class = [0 for _ in range(part_num)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        model = model.eval()

        for batch_id, data in tqdm(enumerate(dl_val), total=len(dl_val), smoothing=0.9):

            points, label, target, edge_list = data["points"], data["label"], data["target"], data["edge_list"]
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target, edge_list = points.float().to(device), label.long().to(device), target.long().to(device), edge_list.to(device)
            points = points.transpose(2, 1)
            if config.MODEL.NAME == "GPN" or config.MODEL.NAME == "GPN2":
                seg_pred, trans_feat = model(points, to_categorical(label, num_classes), edge_list)
            elif config.MODEL.NAME == "PN" or config.MODEL.NAME == "PN2":
                seg_pred, trans_feat = model(points, to_categorical(label, num_classes))
            
            seg_pred_loss = seg_pred.contiguous().view(-1, part_num)
            target_loss = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred_loss, target_loss, trans_feat)
            losses.append(loss.item())

            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()



            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(part_num):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
        for cat in sorted(shape_ious.keys()):
            logger.info('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)


    return test_metrics, np.mean(losses)

def set_logger(log_file):

    global logger 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger

if __name__ == "__main__":
    log_file = datetime.now().isoformat().split('.')[0] + '.txt'
    # os joint path
    log_file = os.path.join("logs", log_file)
    # if folder not exist, create it
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    set_logger(log_file)
    main()

from genericpath import exists
import os
import torch
import wandb
import yaml
from munch import Munch
from ShapeNetDataLoader import PartNormalDataset
from model import get_model, get_loss
from tqdm import tqdm
import numpy as np
import provider
from scipy.spatial import cKDTree
from visualization import visualize_point_cloud



#CONFIG
with open('GraphPointNet/config.yaml', 'rt', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = Munch.fromDict(config)

if config.MODEL.NAME == "PNPP":
    from pointnet2_model import get_model, get_loss
elif config.MODEL.NAME == "GPN":
    from model import get_model, get_loss
else:
    raise ValueError("Model not found")
device = config.DEVICE

batch_size = config.DATALOADER.BATCH_SIZE
num_workers = config.DATALOADER.NUM_WORKERS
cache = config.DATALOADER.CACHE

# criterion = config.SOLVER.CRITERION
load_checkpoint_path = config.TEST.LOAD_CHECKPOINT_PATH
visualize = config.TEST.VISUALIZE

npoints = config.DATASET.NUMBER_OF_POINTS
normal_channel = config.DATASET.NORMAL_CHANNEL

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

num_classes = 16
num_part = 50


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.to(device)
    return new_y

def main():

    #============LOAD DATA===============
    #------------------------------------
    print("Load data...")
    data_test = PartNormalDataset(split="test", npoints=npoints,normal_channel=normal_channel, config=config, cache=cache)

    # TRAIN DATA
    dl_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=data_test.my_collate)

    #============MODEL===============
    #--------------------------------
    if config.MODEL.NAME == "PNPP":
        model = get_model(num_classes=num_part)
    elif config.MODEL.NAME == "GPN":
        model = get_model(num_classes=num_classes, graph_type=config.MODEL.GRAPH_TYPE)
    model.to(device)

    #============CRITERION===============
    #------------------------------------
    criterion = get_loss().to(device)

    #============LOAD================
    #--------------------------------
    if (exists(load_checkpoint_path)):
        checkpoint = torch.load(load_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise Exception("Checkpoint not found")


    #============TEST===============
    #--------------------------------
    print("Testing...")
    test(
        model,
        dl_test,
        visualize,
        device)
    print("Testing complete")


def test(model, dl_val, visualize, device):
    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
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
            # point_graph = build_edge_index(points, num_connections=3)
            # points, point_graph = points.to(device), point_graph.to(device)
            if config.MODEL.NAME == "GPN":
                seg_pred, trans_feat = model(points, to_categorical(label, num_classes), edge_list)
            else:
                seg_pred, trans_feat = model(points, to_categorical(label, num_classes))
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

            for l in range(num_part):
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
            #visualize
            if visualize:
                for i in range(cur_batch_size):
                    name = f'{batch_id}_{i}'
                    visualize_path_prediction = os.path.join("visualize", f'{name}.ply')
                    visualize_path_prediction_gt = os.path.join("visualize", f'{name}_gt.ply')
                    os.makedirs(os.path.dirname(visualize_path_prediction), exist_ok=True)

                    prediction = cur_pred_val[i, :]
                    target_i = target[i, :]
                    points_i = points[i, :, :].cpu().data.numpy()
                    visualize_point_cloud(points_i, prediction, visualize_path_prediction)
                    visualize_point_cloud(points_i, target_i, visualize_path_prediction_gt)               



        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            print('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    print('Accuracy is: %.5f' % test_metrics['accuracy'])
    print('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
    print('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    print('Inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])

    return test_metrics

if __name__ == "__main__":
    main()
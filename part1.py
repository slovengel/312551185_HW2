import random
import os
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm.auto import tqdm
from PIL import Image
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import csv

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

NUM_CLASSES = 10 + 1
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCH = 10


class COCODetectionDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.root_dir = os.path.join('C:\\Users\\HW2\\nycu-hw2-data', data)
        self.transforms = transforms
        self.data = data

        if (data != 'test'):
            with open(self.root_dir + '.json', 'r') as f:
                coco_json = json.load(f)

            self.images = coco_json['images']

            self.image_id_to_annotations = {}  # keys strat at 1
            for ann in coco_json['annotations']:
                img_id = ann['image_id']
                if img_id not in self.image_id_to_annotations:
                    self.image_id_to_annotations[img_id] = []
                self.image_id_to_annotations[img_id].append(ann)

        else:
            self.images = []
            for name in os.listdir(self.root_dir):
                self.images.append(int(name.split('.')[0]))

    def __getitem__(self, idx):  # idx starts at 0
        image_info = self.images[idx]
        img_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        image_id = image_info['id']  # image_id starts at 1
        if (self.data != 'test'):
            boxes = []
            labels = []

            for ann in self.image_id_to_annotations.get(image_id, []):
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])  # category_id starts at 1

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {
                'boxes': boxes,
                'labels': labels,
            }

            if self.transforms:
                image = self.transforms(image)

            return image, target, image_id

        else:
            if self.transforms:
                image = self.transforms(image)

            return image, None, image_id

    def __len__(self):
        return len(self.images)


transform = Compose([
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_dataset = COCODetectionDataset('train', transforms=transform)
val_dataset = COCODetectionDataset('valid', transforms=transform)

train_dataflow = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: list(zip(*x))
)
val_dataflow = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda x: list(zip(*x))
)

model = fasterrcnn_resnet50_fpn_v2(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

num_params_requires_grad = 0
num_params = 0
for param in model.parameters():
    if param.requires_grad:
        num_params_requires_grad += param.numel()
    num_params += param.numel()
print("#Params (all):\t\t", num_params)
print("#Params (requires_grad):", num_params_requires_grad)


def train(
    model: nn.Module,
    dataflow: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
):
    model.train()

    total_loss = 0
    count = 0
    for images, targets, image_id in tqdm(dataflow, desc='train', leave=False):
        images = list(img.cuda() for img in images)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()

        optimizer.step()

        total_loss += losses.item()
        count += 1

    scheduler.step()

    avg_loss = total_loss / count
    return avg_loss


def evaluate(
    model: nn.Module,
    dataflow: DataLoader,
    gt_json_path: str,
    dt_json_path: str,
):

    results = []

    with torch.no_grad():
        for images, targets, image_ids in tqdm(
            dataflow,
            desc='eval',
            leave=False
        ):

            images = list(img.cuda() for img in images)
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            model.eval()
            outputs = model(images)

            for target, output, image_id in zip(targets, outputs, image_ids):
                image_id = int(image_id)

                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(width),
                            float(height)
                        ],
                        "score": float(score)
                    })

    with open(dt_json_path, "w") as f:
        json.dump(results, f)

    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(dt_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map = coco_eval.stats[0]  # mAP@[0.5:0.95]

    return map


optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=NUM_EPOCH,
    eta_min=1e-6
)

train_losses = []
val_losses = []
train_maps = []
val_maps = []

root_dir = 'C:\\Users\\HW2\\nycu-hw2-data\\'

model.cuda()

for epoch in range(NUM_EPOCH):

    print("Epoch",  epoch + 1, ":")
    train_loss = train(model, train_dataflow, optimizer, scheduler)
    train_map = evaluate(
        model,
        train_dataflow,
        root_dir + 'train.json',
        f"train_preds_{epoch+1}.json"
    )
    train_maps.append(train_map)
    train_losses.append(train_loss)

    val_map = evaluate(
        model,
        val_dataflow,
        root_dir + 'valid.json',
        f"val_preds_{epoch+1}.json"
    )
    val_maps.append(val_map)

    torch.save(model.state_dict(), f"fasterrcnn_{epoch+1}.pt")

with open('train_losses.csv', 'wb') as train_losses_csv:
    wr = csv.writer(train_losses_csv, quoting=csv.QUOTE_ALL)
    wr.writerow(train_losses)

with open('val_losses', 'wb') as val_losses_csv:
    wr = csv.writer(val_losses_csv, quoting=csv.QUOTE_ALL)
    wr.writerow(val_losses)

with open('train_maps', 'wb') as train_maps_csv:
    wr = csv.writer(train_maps_csv, quoting=csv.QUOTE_ALL)
    wr.writerow(train_maps)

with open('val_maps', 'wb') as val_maps_csv:
    wr = csv.writer(val_maps_csv, quoting=csv.QUOTE_ALL)
    wr.writerow(val_maps)


class COCODetectionDataset_test(Dataset):
    def __init__(self, data, transforms=None):
        self.root_dir = os.path.join('C:\\Users\\HW2\\nycu-hw2-data', data)
        self.transforms = transforms
        self.images = []
        for name in os.listdir(self.root_dir):
            self.images.append({
                'id': int(name.split('.')[0]),
                'file_name': name
            })

    def __getitem__(self, idx):  # idx starts at 0
        image_info = self.images[idx]
        img_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        image_id = image_info['id']  # image_id starts at 1
        if self.transforms:
            image = self.transforms(image)

        return image, image_id

    def __len__(self):
        return len(self.images)


test_dataset = COCODetectionDataset_test('test', transforms=transform)
test_dataflow = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda x: list(zip(*x))
)

for inputs, image_ids in test_dataflow:
    print("Showing the first testing batch (test_dataflow[0])")
    print("batch size:", len(inputs))
    for i in range(len(inputs)):
        print("image_id: ", image_ids[i])
        print(f"[inputs{i}] dtype:{inputs[i].dtype}, shape:{inputs[i].shape}")
    break


def inference(
    model: nn.Module,
    dataflow: DataLoader,
    dt_json_path: str,
):

    results = []
    model.eval()

    with torch.no_grad():
        for images, image_ids in tqdm(dataflow, desc='eval', leave=False):

            images = list(img.cuda() for img in images)

            outputs = model(images)

            for output, image_id in zip(outputs, image_ids):
                image_id = int(image_id)

                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(width),
                            float(height)
                        ],
                        "score": float(score)
                    })

    with open(dt_json_path, "w") as f:
        json.dump(results, f)


model.load_state_dict(torch.load('fasterrcnn_5.pt', weights_only=True))
model.cuda()
inference(model, test_dataflow, "test_preds_5.json")

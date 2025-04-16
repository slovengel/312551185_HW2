import os
import numpy as np
import torch
import json
import csv
from torchvision.ops import nms
from collections import defaultdict
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import matplotlib.patches as patches

image_id_global = 0


def plot_clusters(
    image_path,
    detections,
    labels,
    title="Cluster Visualization"
):

    img = plt.imread(image_path)

    colors = plt.cm.get_cmap('tab10', np.max(labels)+2)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.set_title(title)

    for det, label in zip(detections, labels):
        x, y, w, h = det["bbox"]
        color = 'red' if label == -1 else colors(label)
        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y - 3,
            f'{det["category_id"] - 1}',
            color=color,
            fontsize=20
        )

    plt.axis('off')
    plt.show()


def special_case_3_detections(detections, max_ratio=2):
    if len(detections) != 3:
        return detections

    sorted_detections = sorted(
        detections,
        key=lambda d: d["bbox"][0] + d["bbox"][2] / 2
    )

    centers = [
        d["bbox"][0] + d["bbox"][2] / 2
        for d in sorted_detections
    ]

    d1 = abs(centers[1] - centers[0])
    d2 = abs(centers[2] - centers[1])

    ratio = max(d1, d2) / (min(d1, d2) + 1e-6)  # avoid divide-by-zero

    if ratio > max_ratio:
        if d1 > d2:
            return [0, 1, 1]
        else:
            return [1, 1, 0]
    else:
        return [0, 0, 0]


def remove_outliners_DBSCAN(detections, eps=30, min_samples=1):
    if len(detections) == 0:
        return []

    boxes = np.array(detections)
    centers = np.array([
        [d["bbox"][0] + d["bbox"][2] / 2, d["bbox"][1] + d["bbox"][3] / 2]
        for d in boxes
    ])

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = clustering.labels_

    if (image_id_global <= 50):
        plot_clusters(
            "/content/" + str(image_id_global) + ".png",
            detections,
            labels,
            title="DBSCAN Clustering"
        )

    label_counts = np.bincount(labels[labels != -1])
    if len(label_counts) == 0:
        return []

    largest_cluster = np.argmax(label_counts)
    filtered = [
        d for d, label in zip(detections, labels) if label == largest_cluster
    ]

    return filtered


def remove_outliners_meanshift(detections, quantile=0.4, n_samples=None):

    filter_max = False

    if len(detections) == 0:
        return []

    if len(detections) == 1:
        labels = [0]
    if len(detections) == 2:
        labels = [0, 0]

    else:

        features = np.array([
            [
                d["bbox"][0] + d["bbox"][2] / 2,  # x_center
                d["bbox"][1] + d["bbox"][3] / 2,  # y_center
            ]
            for d in detections
        ])

        bandwidth = estimate_bandwidth(
            features,
            quantile=quantile,
            n_samples=n_samples or len(features)
        )

        if bandwidth <= 0:
            bandwidth = 30

        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        meanshift.fit(features)
        labels = meanshift.labels_

        if len(detections) == 3:
            unique_labels = np.unique(labels)
            if len(unique_labels) == 3:
                labels = special_case_3_detections(detections)

    label_counts = np.bincount(labels)
    largest_cluster = np.argmax(label_counts)
    filtered = [
        d for d, label in zip(detections, labels) if label == largest_cluster
    ]

    if len(filtered) >= 2:
        areas = [d["bbox"][2] * d["bbox"][3] for d in filtered]
        max_area = max(areas)
        min_area = min(areas)

        if max_area > 3 * min_area:
            filter_max = True
            max_index = areas.index(max_area)
            filtered.pop(max_index)
            new_label = max(labels) + 1
            labels[max_index] = new_label

    label_counts = np.bincount(labels)
    largest_cluster = np.argmax(label_counts)
    filtered = [
        d for d, label in zip(detections, labels) if label == largest_cluster
    ]

    # Visualization
    if (
        image_id_global <= 700
        and (len(filtered) < len(detections) or filter_max)
    ):
        print("image_id:", image_id_global)
        plot_clusters(
            "/content/" + str(image_id_global) + ".png",
            detections,
            labels,
            title="MeanShift Clustering"
        )

    return filtered


def recognize_number(detections, score_thresh=0.4, iou_threshold=0.2):

    detections = [d for d in detections if d["score"] >= score_thresh]

    if not detections:
        return str(-1)

    boxes = []
    scores = []
    for det in detections:
        x, y, w, h = det["bbox"]
        boxes.append([x, y, x + w, y + h])
        scores.append(det["score"])
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    keep = nms(boxes, scores, iou_threshold)

    detections = [detections[i] for i in keep]

    if not detections:
        return str(-1)

    detections = remove_outliners_meanshift(detections)
    detections = sorted(detections, key=lambda d: d["bbox"][0])

    return ''.join(str(d["category_id"] - 1) for d in detections)


path = os.path.join('/content/pred.json')
with open(path, 'r') as f:
    pred_json = json.load(f)

pred_json_grouped = defaultdict(list)

for entry in pred_json:
    pred_json_grouped[entry["image_id"]].append({
        "category_id": entry["category_id"],
        "bbox": entry["bbox"],
        "score": entry["score"]
    })

rows = [("image_id", "pred_label")]

for image_id, detections in dict(sorted(pred_json_grouped.items())).items():
    image_id_global = image_id
    number = recognize_number(detections)
    rows.append((image_id, number))

with open("pred.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

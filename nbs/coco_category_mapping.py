import numpy as np
import json

from load_data import load_nsd

with open('data/NSD/coco_img_id_to_categories.json', 'r') as f:
    img_id_to_categories = json.load(f)

all_labels = []
for v in img_id_to_categories.values():
    all_labels.extend(v)
all_labels = list(set(all_labels))
all_labels.sort()

label_idx_mapping = {label: i for i, label in enumerate(all_labels)}

num_sessions = [40, 40, 32, 30, 40, 32, 40, 30]
n_images_per_subject = [1000, 1000, 930, 907, 1000, 930, 1000, 907]
label_arr = []

for subj in range(1, 9):
    num_session = num_sessions[subj-1]
    batch_size = 16
    _, _, _, test_dataloader, _, _, _ = load_nsd(num_session, subj, 1, batch_size, batch_size, False)

    n_images = n_images_per_subject[subj-1]
    label_arr_subj = []
    for batch_idx, batch in enumerate(test_dataloader):
        coco_image_idx = batch['img_idx'].cpu().numpy()
        
        labels = [img_id_to_categories[str(coco_image_idx[i])] for i in range(len(coco_image_idx))]
        # map to index
        labels = [[label_idx_mapping[l] for l in label] for label in labels]
        labels = [np.array([1 if i in label else 0 for i in range(len(all_labels))]) for label in labels]
        labels = np.stack(labels, 0)
        label_arr_subj.append(labels)

    label_arr_subj = np.concatenate(label_arr_subj, 0)
    assert len(label_arr_subj) == n_images

    label_arr.append(label_arr_subj)

label_arr = np.concatenate(label_arr, 0)
np.save('data/NSD/coco_labels_val.npy', label_arr)
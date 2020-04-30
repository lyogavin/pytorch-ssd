import numpy as np
import pathlib
import cv2
import os
import pandas as pd
import copy
import json

FPS = 3

DEBUG = False


def sec_to_frame(sec):
    """
    Convert time index (in second) to frame index.
    0: 900
    30: 901
    """
    return (sec - 900) * FPS


NUM_FRAMES = 3
SAMPLE_RATE = FPS // NUM_FRAMES

def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    Sample frames among the corresponding clip.
    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames
    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq

class AVADataset:

    def __init__(self, root,
                 transform=None, target_transform=None,
                 dataset_type="train", balance_data=False,
                 single_frame_sec=False,
                 return_image_id=False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.single_frame_sec = single_frame_sec
        self.return_image_id = return_image_id


        self.data, self.class_names, self.class_dict = self._read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data:
            self.data = self._balance_data()
        self.ids = [info['image_id'] for info in self.data]

        self.class_stat = None

    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        # duplicate boxes to prevent corruption of dataset
        boxes = copy.copy(image_info['boxes'])
        boxes[:, 0] *= image.shape[1]
        boxes[:, 1] *= image.shape[0]
        boxes[:, 2] *= image.shape[1]
        boxes[:, 3] *= image.shape[0]
        # duplicate labels to prevent corruption of dataset
        labels = copy.copy(image_info['labels'])
        if self.return_image_id and self.transform:
                image = self.transform(image)
                boxes = None
                lables = None
        elif self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image_info['image_id'], image, boxes, labels

    def __getitem__(self, index):
        image_id, image, boxes, labels = self._getitem(index)
        if self.return_image_id:
            return image_id, image
        else:
            return image, boxes, labels

    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.return_image_id:
            if self.transform:
                image = self.transform(image)
        else:
            if self.transform:
                image, _ = self.transform(image)
        return image

    def _read_data(self):
        annotation_file = f"{self.root}/ava_{self.dataset_type}_v2.1.csv"

        print('opening ', annotation_file)
        if DEBUG:
            annotation_file = annotation_file + ".debug"
        annotations = pd.read_csv(annotation_file,
                                  names = ['video_id', 'sec_id', "XMin", "YMin", "XMax", "YMax", "class_id", "person_id"])

        class_names_dict = dict()
        class_dict = dict()

        max_class_id = 0


        with open(f"{self.root}/ava_action_list_v2.1_for_activitynet_2018.pbtxt") as f:
            for line in f:

                if "name:" in line:
                    class_name_start_pos = line.find('"')
                    class_name_end_pos = line.find('"', class_name_start_pos+1)
                    class_name = line[class_name_start_pos + 1: class_name_end_pos]

                if "id:" in line:
                    class_id_start_pos = line.find(':')
                    class_id = line[class_id_start_pos + 2:].rstrip()
                    class_id = int(class_id)

                    class_names_dict[class_id] = class_name
                    max_class_id = max(max_class_id, class_id)
                    class_dict[class_name] = class_id

        class_names = []


        for iii in range(max_class_id + 1):
            if iii in class_names_dict:
                class_names.append(class_names_dict[iii])
            else:
                class_names.append("")

        print(class_names)




        none_exist_count = 0
        data = []
        for video_id_sec_id, group in annotations.groupby(["video_id", "sec_id"]):
            video_id, sec_id = video_id_sec_id
            frame = sec_to_frame(sec_id)
            if self.single_frame_sec:
                seq = [frame]
            else:
                seq = get_sequence(frame, NUM_FRAMES // 2, SAMPLE_RATE, FPS * (15 * 60 + 1))

            for frame_id in seq:
                image_id = f"{video_id}_%06d" % frame_id
                image_file = self.root / f"{image_id}"[:-7] / f"{image_id}.jpg"
                if not os.path.exists(image_file):
                    none_exist_count += 1
                    continue

                boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
                # make labels 64 bits to satisfy the cross_entropy function
                labels = np.array(group["class_id"], dtype='int64')
                data.append({
                    'image_id': image_id,
                    'boxes': boxes,
                    'labels': labels
                })

        print('non exist frames count:', none_exist_count)
        return data, class_names, class_dict

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   f"Minimum Number of Images for a Class: {self.min_image_num}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)

    def _read_image(self, image_id):
        #image_file = self.root / self.dataset_type / f"{image_id}.jpg"
        image_file = self.root / f"{image_id}"[:-7] / f"{image_id}.jpg"
        image = cv2.imread(str(image_file))
        if image is None:
            print('none reading %s' % image_file)
            return None

        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _balance_data(self):
        label_image_indexes = [set() for _ in range(len(self.class_names))]
        for i, image in enumerate(self.data):
            for label_id in image['labels']:
                label_image_indexes[label_id].add(i)
        label_stat = [len(s) for s in label_image_indexes]
        self.min_image_num = min(label_stat[1:])
        sample_image_indexes = set()
        for image_indexes in label_image_indexes[1:]:
            image_indexes = np.array(list(image_indexes))
            sub = np.random.permutation(image_indexes)[:self.min_image_num]
            sample_image_indexes.update(sub)
        sample_data = [self.data[i] for i in sample_image_indexes]
        return sample_data



if __name__ == '__main__':
    print ('testing...')
    from torch.utils.data import DataLoader
    ds = AVADataset("/home/pi/ava_dataset/", dataset_type="val")

    #print(list(DataLoader(ds, num_workers=0)))
    
    for a in DataLoader(ds, num_workers=0):
        print([x.shape for x in a])



from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import pandas as pd

FPS = 3
def sec_to_frame(sec):
    """
    Convert time index (in second) to frame index.
    0: 900
    30: 901
    """
    return (sec - 900) * FPS

if len(sys.argv) < 4:
    print('Usage: python plot_prediction_csv.py  <label path> <dataset path> <csv path>')
    sys.exit(0)
label_path = sys.argv[1]
dataset_path = sys.argv[2]
csv_path = sys.argv[3]

class_names = [name.strip() for name in open(label_path).readlines()]

predictions = pd.read_csv(csv_path,
                          names=['video_id', 'sec_id', "XMin", "YMin", "XMax", "YMax", "class_id", "score"])


sample = predictions.sample(n=1).iloc[0]

video_id = sample['video_id']
frame_id =  sec_to_frame(sample['sec_id'])
image_id = f"{video_id}_%06d" % frame_id

image_path = f"{dataset_path}/{video_id}/{image_id}.jpg"

print('img:', image_path)


rows = predictions.loc[(predictions['video_id'] == video_id) & (predictions['sec_id'] == sample['sec_id'])]
print('rows', rows)

orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
#boxes, labels, probs = predictor.predict(image, 10, 0.4)
rows = rows.sort_values("score", ascending=False)
print('rows', rows)
rows = rows[:10]
print('rows', rows)
rows = rows[rows['score'] > 0.4]
print('rows', rows)

for inx, row in rows.iterrows():
#for i in range(boxes.size(0)):
    print('row', row)
    box = [row[ "XMin"], row["YMin"], row["XMax"], row["YMax"]] #boxes[i, :]
    print("box", box)
    cv2.rectangle(orig_image, (int(box[0]), int(box[1]), int(box[2]), int(box[3])), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label_id = row['class_id']
    label = f"{class_names[label_id]}: {row['score']:.2f}"
    cv2.putText(orig_image, label,
                (int(box[0])+ 20, int(box[1]) + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = "plot_prediction_csv_output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(rows)} objects. The output image is {path}")

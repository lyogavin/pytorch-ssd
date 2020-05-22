import argparse
import time
import glob
import json
import os
import shutil
import subprocess
import uuid
from collections import OrderedDict
import logging
from joblib import delayed
from joblib import Parallel
import pandas as pd
import shutil
import sys
from pathlib import Path
import random

FPS = 3 #30

logger = logging.getLogger()
formatter = logging.Formatter(
    '%(process)d-%(asctime)s %(levelname)s: %(message)s '
    '[in %(pathname)s:%(lineno)d]')
#handler = logging.FileHandler("./downloader.log")
handler = logging.StreamHandler(sys.stdout)

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def sec_to_frame(sec):
    """
    Convert time index (in second) to frame index.
    0: 900
    30: 901
    """
    return (sec - 900) * FPS

def create_video_folders(output_dir, tmp_dir):
    """Creates a directory for each label name in the dataset."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(f"{output_dir}/frames"):
        os.makedirs(f"{output_dir}/frames")

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        logger.info("created %s" % tmp_dir)

    Path(f"{output_dir}/donemarkers/test.done").touch()


def open_annotation(filename):
    anno = pd.read_csv(filename, index_col=0,header=None)
    return anno


NUM_FRAMES = 5
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

def download_clip_wrapper(line, tmp_dir, output_dir, i, total_count):
    """Wrapper for parallel processing purposes."""
    try:
        logger = logging.getLogger()
        formatter = logging.Formatter(
            '%(process)d-%(asctime)s %(levelname)s: %(message)s '
            '[in %(pathname)s:%(lineno)d]')
        handler = logging.FileHandler("./downloader.log")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.info("processing %d/%d %s..." % (i, total_count, line))
        anno = open_annotation(f"{output_dir}/ava_train_v2.1.csv")

        #if os.path.exists(f"{output_dir}/donemarkers/{line}.done"):
        #    logger.info(f"{output_dir}/donemarkers/{line}.done already existed, skipping...")
        #    return line
        if line[-5:] == ".webm":
            video_name = line[:-5]
        else:
            video_name = line[:-4]

        # check done by check files:

        if os.path.exists(f"{output_dir}/{video_name}"):
            extracted_file_list = glob.glob(f"{output_dir}{video_name}/{video_name}_*.jpg")

            if len(extracted_file_list) >= 2705:
                logger.info(f"2705 jpg files already found in {output_dir}{video_name}/ already existed, skipping...")
                return

        tmp_dir = "%s/%s/" % (tmp_dir, line)



        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)


        command = ['wget',
                   'https://s3.amazonaws.com/ava-dataset/trainval/%s' % line,
                   '-P',
                   tmp_dir
                   ]
        command = ' '.join(command)
        logger.info("running %s" % command)
        try:
            output = subprocess.check_output(command, shell=True,
                                             stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            logger.info("error:%s"% err)
            return

        logger.info("%s downloaded." % line)



        """
        if line[-5:] == ".webm":
            old_line = line
            line = "%s.mp4" % line[:-4]
        """


        video_output_dir = f"{output_dir}{video_name}/"
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        out_name = f"{video_output_dir}{video_name}_%06d.jpg".replace(" ", "\ ")


        result = subprocess.run(['sync', "%s/%s" % (tmp_dir, line)], stdout=subprocess.PIPE)

        """
        for ii in range(5):
            time.sleep(1)
            result = subprocess.run(['ls', '-l', "%s/%s" % (tmp_dir, line)], stdout=subprocess.PIPE)
            logger.info("ls output: %s" % result.stdout.decode('utf-8'))
        """

        #command = ['ffmpeg',  '-ss', '900', '-t', '901', '-i', "%s/%s" % (tmp_dir, old_line), "%s/15min_%s" % (tmp_dir, line)
        command = ['ffmpeg',  '-threads', '1', '-loglevel', 'info','-ss', '900', '-t', '901', '-i', "%s%s" % (tmp_dir, line),
                   '-r', '%d' % FPS, '-y', '-q:v', "1", out_name ]
        #command = ['ffmpeg',  '-i', "%s/15min_%s" % (tmp_dir, line), '-r', '30', '-q:v', "1", out_name
        command = ' '.join(command)
        logger.info("running %s" % command)
        ffoutput = ""
        for ii in range(5):
            try:
                ffoutput = subprocess.check_output(command, shell=True,
                                                 stderr=subprocess.STDOUT)
                #logger.info("output: %s" % ffoutput)
                break
            except subprocess.CalledProcessError as err:
                logger.info("error:%s"% err)
                logger.info("output:%s"% ffoutput)

        """
        extracted_file_list = glob.glob(f"{tmp_dir}/{video_name}_*.jpg")

        #logger.info("%s cut." % line)
        logger.info("%d extracted to frames of %s." % (len(extracted_file_list), out_name))


        command = ['ffmpeg',  '-i', "%s/15min_%s" % (tmp_dir, line), '-r', '30', '-q:v', "1", out_name
                   ]
        command = ' '.join(command)
        try:
            output = subprocess.check_output(command, shell=True,
                                             stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            logger.info("error:", err)
            return

        logger.info("%s extracted to frames of %s." % (line, out_name))

        saved_frame_count = 0
        for sec in anno.loc[video_name][1].unique():
            frame = sec_to_frame(sec)
            seq = get_sequence(frame, NUM_FRAMES // 2, SAMPLE_RATE, len(extracted_file_list))

            saved_frame_count += len(seq)
            #logger.info("trying to move %d files: %s" % (len(seq), str(seq)))

            for frame_id in seq:
                video_file_name = f"{video_name}_%06d.jpg" % frame_id
                try:
                    if os.path.exists(out_name % frame_id):
                        shutil.move(out_name % frame_id, f"{output_dir}/frames/{video_file_name}")
                except:
                    logger.info("error moving: %s"% sys.exc_info()[0])


        logger.info("%s saved %d frames." % (line, saved_frame_count))
        """

        #os.remove(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info("temp dir %s removed" % tmp_dir)

        Path(f"{output_dir}/donemarkers/{line}.done").touch()
        logger.info(f"{video_output_dir} done")
        return line
    except Exception as e:
        logger.info("err %s"% e)
        return line
    except:
        logger.info("Unexpected error: %s" % sys.exc_info()[0])
        return line




def main(input_csv, output_dir,num_jobs=24, tmp_dir='/tmp/ava_data'):


    # Creates folders where videos will be saved later.
    label_to_dir = create_video_folders(output_dir, tmp_dir)

    with open(f"{output_dir}/ava_file_names_trainval_v2.1.txt") as f:
      lst = [line for line in f]

    random.shuffle(lst)



    # Download all clips.
    if num_jobs == 1:
        status_lst = []
        for i, line in enumerate(lst):
            status_lst.append(download_clip_wrapper(line.strip(), tmp_dir, output_dir, i, 299))
    else:
        status_lst = Parallel(n_jobs=num_jobs)(delayed(download_clip_wrapper)(
            line.strip(), tmp_dir, output_dir, i, 299) for i, line in enumerate(f))



    # Save download report.
    with open('download_report.json', 'w') as fobj:
        fobj.write(json.dumps(status_lst))


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_csv', type=str,
                   help=('CSV file containing the following format: '
                         'YouTube Identifier,Start time,End time,Class label'))
    p.add_argument('output_dir', type=str,
                   help='Output directory where videos will be saved.')
    p.add_argument('-n', '--num-jobs', type=int, default=24)
    p.add_argument('-t', '--tmp-dir', type=str, default='/tmp/ava_data')
    main(**vars(p.parse_args()))

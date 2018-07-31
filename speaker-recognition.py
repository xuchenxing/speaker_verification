#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import itertools
import glob
import argparse
import sys
import time
import numpy as np

import pyaudio

import utils
from interface import ModelInterface


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_args():
    desc = "Speaker Recognition Command Line Tool"
    epilog = """
Wav files in each input directory will be labeled as the basename of the directory.
Note that wildcard inputs should be *quoted*, and they will be sent to glob.glob module.
Examples:
    Train (enroll a list of person named person*, and mary, with wav files under corresponding directories):
    ./speaker-recognition.py -t enroll -i "/tmp/person* ./mary" -m model.out
    Predict (predict the speaker of all wav files):
    ./speaker-recognition.py -t predict -i "./*.wav" -m model.out
"""
    parser = argparse.ArgumentParser(description=desc,epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--task',
                       help='Task to do. Either "enroll", "predict", "real", "verify"',
                       default='verify')

    parser.add_argument('-i', '--input',
                       help='Input Files(to predict) or Directories(to enroll)',
                        #default='/tmp/speaker_recognition/train/')
                        default='/tmp/speaker_recognition/predict/shiwx.wav')

    parser.add_argument('-m', '--model',
                       help='Model file to save(in enroll) or use(in predict)',
                       default='model.out')

    parser.add_argument('-p', '--personid',
                        help='input personid',
                        default='shiwx')




    ret = parser.parse_args()
    return ret

def task_enroll(input_dirs, output_model):
    m = ModelInterface()

    #get all the subdir
    train_dir = []
    for subdirs in os.walk(input_dirs):
        train_dir.append(subdirs[0])
    train_dir.remove(train_dir[0])  #去掉本身根目录

    #input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    #train_dir = itertools.chain(*(glob.glob(d) for d in input_dirs))
    #train_dir = [d for d in train_dir if os.path.isdir(d)]

    files = []
    if len(train_dir) == 0:
        print ("No valid directory found!")
        sys.exit(1)

    for d in train_dir:
        label = os.path.basename(d.rstrip('/'))
        wavs = glob.glob(d + '/*.wav')

        if len(wavs) == 0:
            print ("No wav file found in %s"%(d))
            continue
        for wav in wavs:
            try:
                fs, signal = utils.read_wav(wav)
                m.enroll(label, fs, signal)
                print("wav %s has been enrolled"%(wav))
            except Exception as e:
                print(wav + " error %s"%(e))

        m.train()
        m.dump(output_model)

# use a wav file to predict who is the speaker
def task_predict(input_files, input_model):
    start_time = time.time()
    print('开始时间：',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time)))
    m = ModelInterface.load(input_model)
    for f in glob.glob(os.path.expanduser(input_files)):
        fs, signal = utils.read_wav(f)
        label,probability = m.predict(fs, signal)
        #print(probability)
        if probability > -48 :
            print (f, '->', label)
        else:
            print (f, '->未识别到说话人')
    end_time = time.time()
    print('结束时间：',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end_time)))
    print('共耗时',end_time-start_time)

# biuld a realtime voice stream to recognize who is speaking now
def task_realtime_predict(input_model):
    print('start')
    m = ModelInterface.load(input_model)

    # set recording parameter
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    INTERVAL = 1
    INITLEN = 2

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    # fulfill the frame
    for i in range(0, int(RATE / CHUNK * INITLEN)):
        data = np.fromstring(stream.read(CHUNK), dtype=np.int16).tolist()
        frames.append(data)

    while True:
        for i in range(0, int(RATE / CHUNK * INTERVAL)):
            # 添加新的时间窗数据
            frames.append(np.fromstring(stream.read(CHUNK), dtype=np.int16).tolist())
            # 去掉最老的时间窗数据
            frames.remove(frames[0])

        framesjoin = utils.flat_array(frames)
        framesjoin = np.array(framesjoin)
        label,probability = m.predict(16000, framesjoin)
        print('当前说话人->', label)


# to verify whether the input voice and the person matched
def task_verify(input_model, input_file, personid):
    start_time = time.time()
    print('开始时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    m = ModelInterface.load(input_model)
    for f in glob.glob(os.path.expanduser(input_file)):
        fs, signal = utils.read_wav(f)
        probability = m.verify(fs, signal, personid)
        print(probability)
        if probability > -46 :
            print (f, '-> 匹配成功 ：', personid)
        else:
            print (f, '->未匹配成功')

    end_time = time.time()
    print('结束时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    print('共耗时', end_time - start_time)






if __name__ == "__main__":
    global args
    args = get_args()

    task = args.task
    if task == 'enroll':
        task_enroll(args.input, args.model)
    elif task == 'predict':
        task_predict(args.input, args.model)
    elif task == 'real':
        task_realtime_predict(args.model)
    elif task == 'verify' :
        personid = args.personid
        task_verify(args.model, args.input, args.personid)

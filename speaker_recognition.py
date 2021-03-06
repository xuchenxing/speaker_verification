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

import tornado.ioloop
import tornado.web
import json


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#全局变量
model = '/home/ai/model/model.out'
verify_voice_dir = '/home/ai/voice/verify/'
train_voice_dir = '/home/ai/voice/train/'
#model = 'model.out'
#verify_voice_dir = '/tmp/speaker_recognition/predict/'
#train_voice_dir = '/tmp/speaker_recognition/train/'

def get_args():
    desc = "Speaker Recognition Command Line Tool"
    epilog = """
Wav files in each input directory will be labeled as the basename of the directory.
Note that wildcard inputs should be *quoted*, and they will be sent to glob.glob module.
Examples:
    Train (enroll a list of person named person*, and mary, with wav files under corresponding directories):
    ./speaker_recognition.py -t enroll -i "/tmp/person* ./mary" -m model.out
    Predict (predict the speaker of all wav files):
    ./speaker_recognition.py -t predict -i "./*.wav" -m model.out
"""
    parser = argparse.ArgumentParser(description=desc,epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--task',
                       help='Task to do. Either "train_single","train_full", "predict", "real", "verify"',
                       default='verify')

    parser.add_argument('-i', '--input',
                       help='Input Files(to predict) or Directories(to enroll)',
                        #default='/tmp/speaker_recognition/train/tangyc')
                        default='/tmp/speaker_recognition/predict/yangyang.wav')

    parser.add_argument('-m', '--model',
                       help='Model file to save(in enroll) or use(in predict)',
                       default='model.out')

    parser.add_argument('-p', '--personid',
                        help='input personid',
                        default='xucx')

    ret = parser.parse_args()
    return ret

#全量训练
def task_train_full(input_dirs, output_model):
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
        return 'fail','No valid directory found!'

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

        m.train_full()
        m.dump(output_model)
    return 'success',''

#单个增量训练
def task_train_single(wav_url, person_id):

    if os.path.exists(model) :
        m = ModelInterface.load(model)
    else:
        m = ModelInterface()

    if person_id in m.features:
        return 'fail','aleady exist'

    #下载训练语音文件
    dest_dir = train_voice_dir + person_id
    if not os.path.exists(dest_dir) :
        os.makedirs(dest_dir)
    current_time = time.strftime("%Y%m%d%H%I%S", time.localtime(time.time()))
    dest_wav = dest_dir + '/' + current_time + '_' + person_id + '.wav'

    print(wav_url)
    print(dest_wav)
    utils.download_file(wav_url,dest_wav)

    #获取下载好的训练语音文件
    wavs = glob.glob(dest_dir + '/*.wav')

    if len(wavs) == 0 :
        return 'fail','no wav files under this dir'

    #train the wavs
    for wav in wavs:
        try:
            fs, signal = utils.read_wav(wav)
            m.enroll(person_id, fs, signal)
            print("wav %s has been enrolled"%(wav))
        except Exception as e:
            print(wav + " error %s"%(e))

    m.train_single(person_id)
    m.dump(model)

    return 'success',''


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
def task_verify(wav_url, person_id):
    start_time = time.time()
    print('开始时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    m = ModelInterface.load(model)

    if person_id not in m.features:
        return 'fail','current user not trained',''

    # 下载训练语音文件，
    current_time = time.strftime("%Y%m%d%H%I%S", time.localtime(time.time()))
    dest_wav = verify_voice_dir + current_time + '_' + person_id + '.wav'
    utils.download_file(wav_url, dest_wav)

    for f in glob.glob(os.path.expanduser(dest_wav)):
        fs, signal = utils.read_wav(f)
        probability = m.verify(fs, signal, person_id)
        print(probability)
        if probability > -48 :
            print (f, '-> 匹配成功 ：', person_id)
            return 'success','','yes'
        else:
            print (f, '->未匹配成功')
            return 'success','','no'

    end_time = time.time()
    print('结束时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    print('共耗时', end_time - start_time)

# 判断是否已注册声纹
def task_check_status(person_id):

    m = ModelInterface.load(model)

    if person_id not in m.features:
        return 'success','','no'
    else:
        return 'success','','yes'


def command_method():
    global args
    args = get_args()

    task = args.task
    if task == 'train_full':
        task_train_full(args.input, args.model)
    elif task == 'train_single':
        task_train_single(args.input, args.model)
    elif task == 'predict':
        task_predict(args.input, args.model)
    elif task == 'real':
        task_realtime_predict(args.model)
    elif task == 'verify' :
        personid = args.personid
        task_verify(args.model, args.input, args.personid)


def httpservice_method():
    application.listen(8888)
    print('server listening at ',8888)
    tornado.ioloop.IOLoop.instance().start()

class train_full(tornado.web.RequestHandler):
    def post(self):
        print(self.request.body)
        request = json.loads(self.request.body)
        status,reason = task_train_full(request['input_dirs'],request['model'])
        res_json = {}
        res_json['status'] = status
        res_json['reason'] = reason
        res_json['result'] = ''
        self.write(json.dumps(res_json))

class train_single(tornado.web.RequestHandler):
    def post(self):
        print(self.request.body)
        request = json.loads(self.request.body)
        status,reason = task_train_single(request['input_voice'],request['person_id'])
        res_json = {}
        res_json['status'] = status
        res_json['reason'] = reason
        res_json['result'] = ''
        self.write(json.dumps(res_json))

class check_status(tornado.web.RequestHandler):
    def post(self):
        request = json.loads(self.request.body)
        person_id = request['person_id']
        status,reason,result = task_check_status(person_id)
        res_json = {}
        res_json['status'] = status
        res_json['reason'] = reason
        res_json['result'] = result
        self.write(json.dumps(res_json))

class verify(tornado.web.RequestHandler):
    def post(self):
        print(self.request.body)
        request = json.loads(self.request.body)
        status,reason,result = task_verify(request['input_voice'],request['person_id'])
        res_json = {}
        res_json['status'] = status
        res_json['reason'] = reason
        res_json['result'] = result
        self.write(json.dumps(res_json))

application = tornado.web.Application([
    (r"/check_status", check_status),
    (r"/train_single", train_single),
    (r"/verify", verify)
])

if __name__ == "__main__":
    #command_method()
    httpservice_method()


from scipy.io import wavfile

import urllib.request as urllib


def read_wav(fname):
    fs, signal = wavfile.read(fname)
    assert len(signal.shape) == 1, "Only Support Mono Wav File!"
    return fs, signal


# flat a array
# input like :[[1, 2, 3], [4, 5], [6, 7]]
# output like : [[1, 2, 3, 4, 5, 6, 7]]
def flat_array(array) :
    for i in range(len(array)):
        if i == 0:
            continue
        array[0].extend(array[i])

    for i in range(len(array) - 1):
        array.remove(array[1])

    return array


#从网络上取文件下载到本地
def download_file(url,dest) :
    #url = 'http://10.21.19.18:8080/wav/a.wav'
    f = urllib.urlopen(url)
    data = f.read()
    with open(dest , "wb") as code:
        code.write(data)
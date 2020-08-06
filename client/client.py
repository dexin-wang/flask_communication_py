import requests
import os
import time


# url = "http://180.201.5.159:1212"
url = "http://127.0.0.1:1212"
# filename = 'test.jpg'
files = os.listdir('./demo')
file_dirs = [os.path.join('./demo', f) for f in files]

start_time = time.time()

for img in file_dirs:
    file = open(img, 'rb')
    files = {'file': (os.path.basename(img), file, 'image/ppm')}

    r = requests.post(url, files=files)
    result = r.text

    print(result)

sum_t = time.time() - start_time
print('sum time: ', sum_t)
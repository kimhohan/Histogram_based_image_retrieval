import numpy as np
import argparse
import glob
import cv2
from matplotlib import pyplot as plt
 
ap = argparse.ArgumentParser() 
ap.add_argument("-d", "--images", required = True,
    help = "Path to the directory of images") 
args = vars(ap.parse_args())
# �̹����� �ҷ��ö� ���丮�� ����� �̹������� �ҷ��������ؼ� ���

index = {} # ���� �̸��� ���� ��ųʸ��� �̹����� ������׷� ����� �̹����� ���� ���� �׷����� �����մϴ�.
images = {} # ���� �̹����� �����ϴ� ��ųʸ�
results = {} # ���̹����� hist�� ���ؼ� ���� ���絵�� �����ϴ� ��ųʸ�

for path in glob.glob(args["images"] + "/*.png"):
    filename = path[path.rfind("/") + 1:]
    image = cv2.imread(path)
    images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    index[filename] = hist
# ���丮�� �̹��������� �̸��� �о�ͼ� �̹����� �дµ� matplotlib�� RGB ������ ��������� RGB���·� ��ȭ�� �����並 �����ݴϴ�.

for (i, hist) in index.items():
    d = cv2.compareHist(index["rivendell-query.png"], hist, cv2.HISTCMP_INTERSECT)
    results[i] = d
# query�� �̹����� hist�� ���丮�� ����� �̹����� hist�� ��(Chi-Square or Intersection�� Intersection�� ���)�ؼ� ���缺�� �������ݴϴ�.

results = sorted([(v, k) for (k, v) in results.items()], reverse = True)
# ���缺�� �������� ���� ����Ѱͺ��� ������ ���ݴϴ�.

plt.subplot(1, 1, 1), plt.imshow(images["rivendell-query.png"],'gray')
plt.title("Query")
plt.xticks([]),plt.yticks([])

plt.show()
# query �̹��� ���

for (i, (score, path)) in enumerate(results):
    plt.subplot(2, 5, i + 1), plt.imshow(images[path],'gray')
    plt.title("%d" % (i+1))
    plt.xticks([]),plt.yticks([])
    if i == 9:
        break;
plt.show()
# query �̹����� ����� 10���� �̹��� ���
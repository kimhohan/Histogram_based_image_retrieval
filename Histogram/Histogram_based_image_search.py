import numpy as np
import argparse
import glob
import cv2
from matplotlib import pyplot as plt
 
ap = argparse.ArgumentParser() 
ap.add_argument("-d", "--images", required = True,
    help = "Path to the directory of images") 
args = vars(ap.parse_args())
# 이미지를 불러올때 디렉토리의 저장된 이미지들을 불러오기위해서 사용

index = {} # 파일 이름이 들어가는 딕셔너리로 이미지의 히스토그램 계산한 이미지의 색상 막대 그래프를 저장합니다.
images = {} # 실제 이미지를 저장하는 딕셔너리
results = {} # 두이미지의 hist를 비교해서 나온 유사도를 저장하는 딕셔너리

for path in glob.glob(args["images"] + "/*.png"):
    filename = path[path.rfind("/") + 1:]
    image = cv2.imread(path)
    images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    index[filename] = hist
# 디렉토리의 이미지파일의 이름만 읽어와서 이미지를 읽는데 matplotlib는 RGB 형식을 사용함으로 RGB형태로 변화후 히스토를 구해줍니다.

for (i, hist) in index.items():
    d = cv2.compareHist(index["rivendell-query.png"], hist, cv2.HISTCMP_INTERSECT)
    results[i] = d
# query의 이미지의 hist와 디렉토리에 저장된 이미지의 hist를 비교(Chi-Square or Intersection중 Intersection을 사용)해서 유사성늘 저장해줍니다.

results = sorted([(v, k) for (k, v) in results.items()], reverse = True)
# 유사성을 바탕으로 가장 비슷한것부터 정렬을 해줍니다.

plt.subplot(1, 1, 1), plt.imshow(images["rivendell-query.png"],'gray')
plt.title("Query")
plt.xticks([]),plt.yticks([])

plt.show()
# query 이미지 출력

for (i, (score, path)) in enumerate(results):
    plt.subplot(2, 5, i + 1), plt.imshow(images[path],'gray')
    plt.title("%d" % (i+1))
    plt.xticks([]),plt.yticks([])
    if i == 9:
        break;
plt.show()
# query 이미지와 비슷한 10개의 이미지 출력
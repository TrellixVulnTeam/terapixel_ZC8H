import matplotlib as mat
mat.use('GTK3Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sklearn.cluster import KMeans
import numpy as np

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

CLUSTERS = 5

print("cv2.imgread()")
#read image
img = cv2.imread('_DSC9671.jpg')


print("plt.figure()")
#plotting
fig = plt.figure(constrained_layout=True, figsize=(12, 10))
gs = fig.add_gridspec(4, 2)
ax1 = fig.add_subplot(gs[2, 0], projection='3d')
ax2 = fig.add_subplot(gs[2, 1], projection='3d')
ax3 = fig.add_subplot(gs[3, :])
ax4 = fig.add_subplot(gs[0, :])
ax5 = fig.add_subplot(gs[1, 0])
ax6 = fig.add_subplot(gs[1, 1])

ax5.hist(img.ravel(),256,[0,256])

clr = ('b','g','r')
for i,col in enumerate(clr):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    ax6.plot(histr,color = col)
    ax6.set_xlim(0,256)

#convert from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#get rgb values from image to 1D array
r, g, b = cv2.split(img)
r = r.flatten()
g = g.flatten()
b = b.flatten()

print("ax1.scatter(r, g, b)")
ax1.scatter(r, g, b)


print("img.reshape()")
img2 = img.reshape((img.shape[0] * img.shape[1], 3))

print("KMeans()")
kmeans = KMeans(n_clusters = CLUSTERS)
kmeans.fit(img2)
colors = kmeans.cluster_centers_
lables = kmeans.labels_

print("ax2.scatter()")
for label, pix in zip(lables, img2):
    ax2.scatter(pix[0], pix[1], pix[2], color = rgb_to_hex(colors[label]))

numLabels = np.arange(0, CLUSTERS + 1)
(hist, _) = np.histogram(lables, bins=numLabels)
hist = hist.astype("float")
hist /= hist.sum()
colors2 = colors
colors2 = colors2[(-hist).argsort()]
hist = hist[(-hist).argsort()]
chart = np.zeros((50, 500, 3), np.uint8)
start = 0

for i in range(CLUSTERS):
    end = start + hist[i] * 500
    # getting rgb values
    r = colors2[i][0]
    g = colors2[i][1]
    b = colors2[i][2]
    # using cv2.rectangle to plot colors
    print("cv2.rectangle()")
    cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)
    start = end

# display chart
ax3.axis("off")
print("ax3.imshow(chart)")
ax3.imshow(chart)

ax4.axis("off")
print("ax4.imshow(img)")
ax4.imshow(img)

print("plt.show()")
plt.show()

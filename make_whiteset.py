import numpy as np
import cv2
import os
import glob

if not os.path.exists("./test"):
    os.mkdir('./test')
    os.mkdir('./test/a')
    os.mkdir('./test/b')

###SETTING
folder = 'data'

class PointList():
    def __init__(self, npoints):
        self.npoints = npoints
        self.ptlist = np.empty((npoints, 2), dtype=int)
        self.pos = 0

    def add(self, x, y):
        if self.pos < self.npoints:
            self.ptlist[self.pos, :] = [x, y]
            self.pos += 1
            return True
        return False


def onMouse(event, x, y, flag, params):
    wname, img, ptlist, n= params
    if event == cv2.EVENT_MOUSEMOVE:  # マウスが移動したときにx線とy線を更新する
        img2 = np.copy(img)
        if ptlist.pos == 1:
            [a,b]=ptlist.ptlist[0]
            cv2.rectangle(img2,(a,b),(x,y),(255,0,0),2)
        cv2.imshow(wname, img2)

    if event == cv2.EVENT_LBUTTONDOWN:  # レフトボタンをクリックしたとき、ptlist配列にx,y座標を格納する
        if ptlist.add(x, y):
            #print('[%d] ( %d, %d )' % (ptlist.pos - 1, x, y))
            img2 = np.copy(img)
            cv2.imshow(wname, img)
        if ptlist.pos == 2:
            img2 = np.copy(img)
            x0=ptlist.ptlist[0][0]
            y0=ptlist.ptlist[0][1]
            x1=ptlist.ptlist[1][0]
            y1=ptlist.ptlist[1][1]


            for i in range(x1-x0):
             for j in range(y1-y0):
                    img2[y0+j,x0+i]=[255,255,255]
            cv2.imwrite("./test/b/"+'{0:04d}'.format(n)+".jpg",img2)
            cv2.imwrite("./test/a/"+'{0:04d}'.format(n)+".jpg",img)
            ptlist = PointList(2)

    # if event == cv2.EVENT_RBUTTONDOWN:
    #     img2 = np.copy(img)
    #     x0=ptlist.ptlist[0][0]
    #     y0=ptlist.ptlist[0][1]
    #     x1=ptlist.ptlist[1][0]
    #     y1=ptlist.ptlist[1][1]
    #
    #
    #     for i in range(x1-x0):
    #      for j in range(y1-y0):
    #             img2[y0+j,x0+i]=[255,255,255]
    #     cv2.imwrite("./test/a/"+'{0:04d}'.format(n)+".jpg",img2)
    #     ptlist = PointList(2)


if __name__ == '__main__':
    npoints = 2
    n=0
    files=glob.glob("./"+folder+"/*.png")
    wname = "kirinuki"
    cv2.namedWindow(wname)
    for file in files:
        img = cv2.imread(file)

        ptlist = PointList(npoints)
        n += 1

        cv2.setMouseCallback(wname, onMouse, [wname, img, ptlist,n])
        cv2.waitKey()

    cv2.destroyAllWindows()

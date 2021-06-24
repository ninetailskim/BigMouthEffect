## 利用Paddlehub制作"王大陆"滤镜
说实话我没看过王大陆参与的电视剧,但是这个表情包我还是见过的   
![](https://ai-studio-static-online.cdn.bcebos.com/c8837587719441858d4d2d7165fc2f3c2992baebd02a40ddbc19444587f34431)   
我个嘴巴是真实存在且纯天然的吗哈哈哈   
我是做不到的,我没有这种天分   
![](https://ai-studio-static-online.cdn.bcebos.com/1a17c90a4d564ecebb8cf9adfad2cec25a99986431ed40f8ad253125a7e9b618)   
但是有了Computer Vision,啥效果不能搞出来啊,废话不多说,这就整起来   
### 你可以在这里看到最后的效果
<iframe  style="width:98%;height: 800px;" src="//player.bilibili.com/player.html?aid=761139177&bvid=BV1764y1r7Z5&cid=354926346&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

### 先导入需要的库,并封装一下facelandmark吧


```python
import cv2
import numpy as np
import argparse
import copy
import math
import paddlehub as hub

class LandmarkUtils():
    def __init__(self):
        super().__init__()
        self.module = hub.Module(name="face_landmark_localization")

    def predict(self, frame):
        result = self.module.keypoint_detection(images=[frame], use_gpu=False)
        if result is not None:
            if len(result) > 0:
                return result[0]['data']
            else:
                return None
        else:
            return None
```

### 原理分析
秉承着一切从简的原则,我们直接将这个问题划分为放大镜特效+嘴部关键点检测两个部分   
我们只需要找到嘴巴的中心点,以合适的半径,做一个放大镜特效就好了   
#### 放大镜特效
![](https://ai-studio-static-online.cdn.bcebos.com/6ad2c647cd384d51b19040b51b02fadd7c03f45f93aa45768abfb677a66742cb)   
理论上是上图,也就是在x,y的位置画上nx,ny的像素
但其实寻找这个nx,ny是根据想x,y和cx,cy的距离与圆半径r的$2^{1/2}$倍有关的.


```python
def mirror(frame, cx=None, cy=None, radius=50):
    h, w = frame.shape[:2]
    if cx is None and cy is None:
        cx = w / 2
        cy = h / 2
    nx = 0
    ny = 0
    radius = int(radius)
    backFrame = copy.deepcopy(frame)
    for i in range(w):
        for j in range(h):
            tx = i - cx
            ty = j - cy
            distance = tx **2 + ty ** 2
            if distance < radius ** 2:
                nx = int(tx * distance ** 0.5 / radius)
                ny = int(ty * distance ** 0.5 / radius)
                nx = int(nx + cx)
                ny = int(ny + cy)
                if ny < h and nx < w and ny > 0 and nx > 0:
                    backFrame[j, i, :] = frame[ny, nx, :]
    return backFrame
```

### 圆心与半径选择
![](https://ai-studio-static-online.cdn.bcebos.com/61fd4341363f47c2a66b88294951b79784c9ffa670944af09008b9279bc2b4ac)   
因为没有直接的嘴巴关键点检测,这里直接使用Paddlehub中的人脸关键点监测   
我们的圆心现在使用的是52,58,63,67四个点的平均值,当然你也可以选择49,55两个点,以保证结果更稳定   
我们的半径使用的也是和52,58两个点正相关的,这里暂且使用这两个点的距离的1.5倍


```python
def magicMirrorByKP(args):
    LU = LandmarkUtils()
    cap = cv2.VideoCapture(0 if args.source == None else args.source)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if args.source == 0:
                frame = cv2.flip(frame, 1)
            res = LU.predict(frame)
            cx = None
            cy = None
            radius = 50
            if res is not None:
                res = res[0]
                cx = (res[51][0] + res[57][0] + res[62][0] + res[66][0]) / 4
                cy = (res[51][1] + res[57][1] + res[62][1] + res[66][1]) / 4
                radius = ((res[51][0] - res[57][0]) ** 2 + (res[51][1] - res[57][1]) ** 2) ** 0.5 * 1.5
                frame = mirror(frame, cx, cy, radius)
            if args.show:
                cv2.imshow("Debug", frame)
                cv2.waitKey(1)
            out.write(frame)
    cap.release()
    out.release()   
    cv2.destroyAllWindows()
```

### 最后调用一下magicMirrorByKP吧
这里我直接调用脚本吧


```python
!python magicMirror.py --source 111.mp4 --save out.mp4
```

### 总结
通过两个功能的简单组合,就完成了这个项目   
通过视频你可以发现在不张嘴的时候也会有一些特效   
这个完全可以通过嘴巴张开的程度来优化掉不需要触发的时刻   
这个就需要大家自己去尝试了哈哈哈

现在我笑起来应该是AIStudio里嘴巴最大的了   
哈哈哈哈哈哈哈哈哈哈   
![](https://ai-studio-static-online.cdn.bcebos.com/8129fe4902a347b5aadc7457ff77ec22d438a0ddbc564131b4cfc93cfbc74de1)

# 个人简介

> 百度飞桨开发者技术专家 PPDE

> 飞桨上海领航团团长

> 百度飞桨官方帮帮团、答疑团成员

> 国立清华大学18届硕士

> 以前不懂事，现在只想搞钱～欢迎一起搞哈哈哈

我在AI Studio上获得至尊等级，点亮9个徽章，来互关呀！！！<br>
[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/311006]( https://aistudio.baidu.com/aistudio/personalcenter/thirdview/311006)

B站ID： 玖尾妖熊

### 其他趣味项目：  
#### [利用Paddlehub制作端午节体感小游戏](https://aistudio.baidu.com/aistudio/projectdetail/2079016)
#### [熊猫头表情生成器[Wechaty+Paddlehub]](https://aistudio.baidu.com/aistudio/projectdetail/1869462)
#### [如何变身超级赛亚人(一)--帅气的发型](https://aistudio.baidu.com/aistudio/projectdetail/1180050)
#### [【AI创造营】是极客就坚持一百秒？](https://aistudio.baidu.com/aistudio/projectdetail/1609763)    
#### [在Aistudio，每个人都可以是影流之主[飞桨PaddleSeg]](https://aistudio.baidu.com/aistudio/projectdetail/1173812)       
#### [愣着干嘛？快来使用DQN划船啊](https://aistudio.baidu.com/aistudio/projectdetail/621831)    
#### [利用PaddleSeg偷天换日～](https://aistudio.baidu.com/aistudio/projectdetail/1403330)    
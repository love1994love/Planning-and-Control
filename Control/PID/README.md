本文件夹包含 `PID` 算法的 `Python` 实现以及 `C++` 实现



解决以下报错：<font color="red">MovieWriter ffmpeg unavailable; using Pillow instead.</font>

[解决matplotlib出现的异常：MovieWriter ffmpeg unavailable； using Pillow instead](https://blog.csdn.net/fenghefeng123/article/details/123016913)

1、下载：https://github.com/FutaAlice/ffmpeg-static-libs/releases

2、添加到环境变量：**D:\ffmpeg_3.4.2\bin\x64**

3、安装 ffmpeg 和 ffmpeg-python

```shell
pip install ffmpeg ffmpeg-python
```

4、查看是否连接成功

```
from matplotlib import animation
print(animation.writers.list())
```


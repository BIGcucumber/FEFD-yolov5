
import numpy as np
from PIL import Image



def resize(img, size):
    # 先创建一个目标大小的幕布，然后将放缩好的图片贴到中央，这样就省去了两边填充留白的麻烦。
    canvas = Image.new("RGB", size=size, color="#7777")

    target_width, target_height = size
    width, height = img.size
    offset_x = 0
    offset_y = 0
    if height > width:              # 高 是 长边
        height_ = target_height     # 直接将高调整为目标尺寸
        scale = height_ / height    # 计算高具体调整了多少，得出一个放缩比例
        width_ = int(width * scale) # 宽以相同的比例放缩
        offset_x = (target_width - width_) // 2     # 计算x方向单侧留白的距离
    else:   # 同上
        width_ = target_width
        scale = width_ / width
        height_ = int(height * scale)
        offset_y = (target_height - height_) // 2

    img = img.resize((width_, height_), Image.BILINEAR) # 将高和宽放缩
    canvas.paste(img, box=(offset_x, offset_y))         # 将放缩后的图片粘贴到幕布上
    # box参数用来确定要粘贴的图片左上角的位置。offset_x是x轴单侧留白，offset_y是y轴单侧留白，这样就能保证能将图片填充在幕布的中央

    return canvas


img= Image.open('E:/Server download/BaldPicture/51.jpg')

target_size =(500 ,300)  # 目标尺寸：宽为500，高为300
width, height =img.size
target_size_=(int(width*0.2), int(height*0.2))
res = resize(img ,target_size_)

res.save('E:/Server download/BaldPicture/51_3.jpg')
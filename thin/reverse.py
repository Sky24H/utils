from PIL import Image
import PIL.ImageOps    
import os 

i = 0

for curDir, dirs, files in os.walk('./edge'):
    for file in files:
        i += 1
        image = Image.open(os.path.join(curDir, file))

        inverted_image = PIL.ImageOps.invert(image)

        inverted_image.save(os.path.join('./edge_white', file))

import glob
from PIL import Image
from tqdm import tqdm

path = './datasets_optimized_SMIS/*/*/*/*'
imgs = glob.glob(path)

for i in tqdm(range(len(imgs))):
    img = Image.open(imgs[i]).resize((256, 256))
    img.save(imgs[i])

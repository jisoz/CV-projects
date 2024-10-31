# from fastai.vision.all import *

# path=untar_data(URLs.CAMVID)
# path.ls()

# validnames=(path/'valid.txt').read_text().split("\n")
# validnames[:5]


# pathimage=path/'images'
# pathlabels=path/'labels'


# images=get_image_files(pathimage)
# lbl=get_image_files(pathlabels)

# img=images[10]
# imgl=PILImage.create(img)
# imgl.show()


# lbls=lbl[6]
# img=PILImage.create(lbls)
# img.show()


# get_msk=lambda o:path/'labels'/f'{o.stem}_P{o.suffix}'


# msk=PILMask.create(get_msk(img))
# msk.show()
# # tensor(msk)

# import numpy as np
# codes=np.loadtxt(path/'codes.txt',dtype=str)
# codes[26]

# sz=msk.shape

# half=tuple(int(x/2) for x in sz);half


# cnvid=DataBlock(blocks=(ImageBlock,MaskBlock(codes)),
#                 get_items=get_image_files,
#                 splitter=RandomSplitter(),
#                 get_y=get_msk,
#                 batch_tfms=[*aug_transforms(size=half),Normalize.from_stats(*imagenet_stats)]
                
                
                
#                 )
# dls=cnvid.dataloaders(path/'images',bs=8)


# dls.show_batch()
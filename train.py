import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/wt_lr_net.yaml')
    model.train(data=r'datasets/data.yaml',
                cache=False,
                imgsz=640,
                epochs=5,
                single_cls=True,
                batch=8,
                close_mosaic=10,
                workers=4,
                device='cuda:0',
                # iou=0.7,
                optimizer='SGD', # using SGD
                # resume='',
                amp=False,
                task='segment',
                project='runs/train',
                name='001',
                )
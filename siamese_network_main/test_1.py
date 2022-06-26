from pathlib import Path

import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class Video(Dataset):
    def __init__(self, video_path, transform=None):
        self.cap = cv2.VideoCapture(video_path)
        self.transform = transform

    def __getitem__(self, index):
        """動画のフレームを返す。
        """
        # フレームを読み込む。
        ret, img = self.cap.read()
        # チャンネルの順番を変更する。 (B, G, R) -> (R, G, B)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # numpy 配列を PIL Image に変換する。
        img = Image.fromarray(img)

        if self.transform is not None:
            # 前処理がある場合は行う。
            img = self.transform(img)

        return img

    def __len__(self):
        """動画のフレーム数を返す。
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


# Transform を作成する。
transform = transforms.Compose([transforms.Resize(100), transforms.ToTensor()])
# Dataset を作成する。
dataset = Video("work/Siamese_Network/video/Bright_40cm.avi", transform)
# DataLoader を作成する。
dataloader = DataLoader(dataset, batch_size=1)

for batch in dataloader:
    print(batch.shape)
    
x=1
for x in range(5):
    print(x)
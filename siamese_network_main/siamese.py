

#特徴的な部分はforward関数です。入力が２つになっています。これは画像間の距離を計算できるようにするためです。
#`out1 = self.forward_one(x1)`で画像１の特徴量をDeep Learningで取得します。
class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()#親クラスのメソッドなどを呼び出すことができるようになります。
        #super(親クラスのオブジェクト, self).親クラスのメソッド # python2系での書き方
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 
        )
        o = 1600
        self.liner = nn.Sequential(nn.Linear(o, 4096), nn.Sigmoid()) #xを変更することで実行可能になった
        self.out = nn.Linear(4096, 1)
    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)#画像１の特徴量をDeep Learningで取得します。
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out
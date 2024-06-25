#coding:utf8
import torch
import torch.nn as nn
import torchvision

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class BCNN(nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.features = torchvision.models.vgg13(pretrained=True).features
        print(torchvision.models.vgg13(pretrained=True))
        self.features = nn.Sequential(*list(self.features.children())[:-1]) ##去掉最后一个池化层
        #print(self.features)
        self.fc = torch.nn.Linear(512**2, 200)
        for param in self.features.parameters():
            param.requires_grad = True
        nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, X):
        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        X = self.features(X)
        assert X.size() == (N, 512, 14, 14)
        X = X.view(N, 512, 14**2)

        # print(X.shape)
        # print(torch.transpose(X, 1, 2).shape)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (14**2)
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 200)
        return X

if __name__ == '__main__':
    model = BCNN()
    model.load_state_dict(torch.load('./models/model.pt',map_location=lambda storage,loc: storage))
    model.train(False)
    model.eval()
    input = torch.randn((1, 3, 224, 224))
    torch.onnx.export(model, input, "model.onnx", verbose=False)
    output = model(input)
    print(output.shape)


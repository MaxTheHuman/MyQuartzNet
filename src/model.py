class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride = 1):
        super(Conv_BN_ReLU, self).__init__()
        
        self.cnn = nn.Conv1d(
            in_channels, out_channels, kernel, stride=stride, padding=kernel//2
        )
        self.batch_norm = nn.modules.batchnorm.BatchNorm1d(
            num_features=out_channels
        )
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x):
        x = self.cnn(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x  # (batch, channel, feature, time)

class TCSConv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(TCSConv_BN_ReLU, self).__init__()

        self.depthwise_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel, padding=kernel//2,
            groups=4, bias=False
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, padding=0, bias=False
        )
        self.batch_norm = nn.modules.batchnorm.BatchNorm1d(
            num_features=out_channels
        )
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x  # (batch, channel, feature, time)

class TCSBlock(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, in_channels, out_channels, kernel):
        super(TCSBlock, self).__init__()
        
        self.TCS_layers = nn.Sequential(*[
            TCSConv_BN_ReLU(in_channels, out_channels, kernel)
            for i in range(R)
        ])
        # self.TCS_layer_1 = TCSConv_BN_ReLU(in_channels=256, out_channels=256, kernel=33)
        # self.TCS_layer_2 = TCSConv_BN_ReLU(in_channels=256, out_channels=256, kernel=39)
        # self.TCS_layer_3 = TCSConv_BN_ReLU(in_channels=256, out_channels=512, kernel=51)
        # self.TCS_layer_4 = TCSConv_BN_ReLU(in_channels=512, out_channels=512, kernel=63)
        # self.TCS_layer_5 = TCSConv_BN_ReLU(in_channels=512, out_channels=512, kernel=75)
        
    def forward(self, x):
        x_copy = x
        x = self.TCS_layers(x)
        # x = TCS_layer_1(x)
        # x = TCS_layer_2(x)
        # x = TCS_layer_3(x)
        # x = TCS_layer_4(x)
        # x = TCS_layer_5(x)
        x += x_copy
        return x  # (batch, channel, feature, time)

class MainBlock(nn.Module):
    def __init__(self):
        super(MainBlock, self).__init__()
        
        self.TCSBlock_1 = TCSBlock(in_channels=256, out_channels=256, kernel=33)
        self.TCSBlock_2 = TCSBlock(in_channels=256, out_channels=256, kernel=39)
        self.helper_2_3 = Conv_BN_ReLU(in_channels=256, out_channels=512, kernel=51)
        self.TCSBlock_3 = TCSBlock(in_channels=512, out_channels=512, kernel=51)
        self.TCSBlock_4 = TCSBlock(in_channels=512, out_channels=512, kernel=63)
        self.TCSBlock_5 = TCSBlock(in_channels=512, out_channels=512, kernel=75)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = self.TCSBlock_1(x)
        x = self.TCSBlock_2(x)
        x = self.helper_2_3(x)
        x = self.TCSBlock_3(x)
        x = self.TCSBlock_4(x)
        x = self.TCSBlock_5(x)
        return x  # (batch, channel, feature, time)

class ASR(nn.Module):
    def __init__(self):
        super(ASR, self).__init__()
        
        self.conv_bn_relu_1 = Conv_BN_ReLU(
            in_channels=64, out_channels=256, kernel=33, stride=2
        )
        self.main_block = MainBlock()
        self.conv_bn_relu_2 = Conv_BN_ReLU(in_channels=512, out_channels=512, kernel=87)
        self.conv_bn_relu_3 = Conv_BN_ReLU(in_channels=512, out_channels=1024, kernel=1)
        self.conv_bn_relu_4 = nn.Conv1d(in_channels=1024, out_channels=len(char2int), kernel_size=1, padding=0, dilation=2, bias=False)
    
    def forward(self, x):
        # x (batch, channel, feature, time)
        x = self.conv_bn_relu_1(x)
        x = self.main_block(x)
        x = self.conv_bn_relu_2(x)
        x = self.conv_bn_relu_3(x)
        x = self.conv_bn_relu_4(x)
        x = x.transpose(1, 2)
        return x  # (batch, channel, feature, time)

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn1 = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn1(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.n_hidden = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def forward(self, features, captions):
        captions_embed = self.embedding(captions[:, :-1]) # pass entire captions except last token at the end of each caption
        x = torch.cat((features.unsqueeze(1), captions_embed), 1) # so that at the end we have sequence of 'max_len' from RNN
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    def sample(self, inputs, states=None, max_len=20):
        output_ids = []
        for i in range(max_len):
            out, states = self.lstm(inputs, states) # processed the current input to generate the next sequence
            out = self.fc(out) # # pass through the fc to get vocab size
            _, idx = out.max(-1) # get the highest confident index
            output_ids.append(int(idx.to("cpu").item()))
            inputs = self.embedding(idx) # process the output y to be used as next input sequence
        return output_ids
    
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
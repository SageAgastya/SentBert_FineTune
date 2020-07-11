from transformers import BertModel, BertTokenizer
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(device)

    def forward(self, sent):
        tokens = self.tokenizer.tokenize(sent)
        tokens = tokens[:512]
        input_ids = torch.tensor(self.tokenizer.encode(tokens, add_special_tokens=False, add_space_before_punct_symbol=True)).unsqueeze(0).to(device)  # Batch size 1
        outputs = self.model(input_ids)
        last_hidden_states, mems = outputs[:2]  # The last hidden-state is the first element of the output tuple
        pooled = torch.sum(last_hidden_states,dim=-2) / (last_hidden_states.shape[-2])
        # print(pooled.shape)
        return pooled

class Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        # self.out = nn.Linear(2*768, 2)
        self.out = nn.Linear(3*768, 2)

    def forward(self, X1, X2):
        res = torch.cat([X1,X2,abs(X1-X2)],dim=-1)
        res = self.out(res)
        return res

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BERT()
        self.sia = Siamese()

    def forward(self, X1, X2):
        X1 = self.bert(X1)
        X2 = self.bert(X2)
        res = self.sia(X1,X2)
        return res

f1 = open("questions.txt","r",encoding="utf8")
f2 = open("answers.txt","r",encoding="utf8")
f3 = open("labels.txt","r",encoding="utf8")

net = Network().to(device)
criterion = nn.CrossEntropyLoss().cuda()
learning_rate = 0.0001
optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=0, betas=(0.9,0.99))
first = f1.readlines()[:10]
second = f2.readlines()[:10]
labels = f3.readlines()[:10]
length = len(labels)
para = net.parameters()
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
loss_values = []
epoch=0
for epoch in range(2):
    scheduler.step()
    running_loss = 0.0
    print('Epoch:', epoch, 'LR:', scheduler.get_lr())
    for i in range(length):
        s1, s2, label = first[i], second[i], labels[i]
        label = torch.tensor([int(label)]).long().to(device)
        optimizer.zero_grad()
        output = net(s1, s2)
        loss = criterion(output, label).cuda()
        print("LOSS:", loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss_values.append(running_loss/length)
f1.close()
f2.close()
f3.close()
PATH = "fine.pt"
torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_values,
            }, PATH)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot(loss_values,"b")
plt.show()

print('Finished Training')

#------------------------------------------

PATH = "finetuned.pt"
model = Network()
checkpnt = torch.load(PATH)
model.load_state_dict(checkpnt["model_state_dict"])
## test now
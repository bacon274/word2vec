import sentencepiece as spm
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tokeniser_load import *
from torch.utils.data import Dataset, DataLoader



def load_data(input_file):
    print("loading data")
    with open(input_file, 'r') as f:
        # Read the content of the file
        content = f.read()
        token_list = encode(content)
        data = []
        for i in range(2, len(token_list) - 2):
            context = [token_list[i - 2], token_list[i - 1],
                    token_list[i + 1], token_list[i + 2]]
            target = token_list[i]
            data.append((context, target))
        return data

class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        #out: 1 x emdedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()
        
        #out: 1 x vocab_size
        self.linear2 = nn.Linear(128, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim = -1)
        

    def forward(self, inputs):
        # embeds = sum(self.embeddings(inputs)).view(1,-1)
        # embeds = torch.mean(self.embeddings(inputs), dim=0)
        # print(f'inputs.shape {inputs.shape}')

        embeds = self.embeddings(inputs)  # embedded: [batch_size, context_size, embedding_dim] [32 x 4 x 100]
        # print(f'embeds.shape {embeds.shape}')
        embed_mean = embeds.mean(dim=1)  # embedded_mean: [batch_size, embedding_dim] [32 x 100]
        # print(f'embeds_mean.shape {embed_mean.shape}')
        out = self.linear1(embed_mean)  # output: [batch_size, vocab_size] [32 x 1000]
        # print(f'out.shape {out.shape}')
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_emdedding(self, word):
        word = torch.tensor(encode(word))
        return self.embeddings(word).view(1,-1)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target,dtype=torch.long)

VOCAB_SIZE = 1000
EMBEDDING_DIMENSIONS = 100
batch_size = 32

data = load_data("corpus_clean.txt")
dataset = CustomDataset(data)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = CBOW(VOCAB_SIZE,EMBEDDING_DIMENSIONS)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001) #0.001


# context = 'And God Let the'
# context_ix = encode(context)
# a = model(torch.tensor(context_ix, dtype=torch.long))


# print(f'Context: {context}\n')
# print(f'Prediction: {decode(torch.argmax(a[0]).item())}')

#TRAINING
for epoch in range(20):
    batch_count = 0
    for batch_inputs, batch_target in data_loader:
        # print(f'batch_inputs ({batch_inputs.shape}) ')
        # print(f'batch_target ({batch_target.shape}): {batch_target} ')
        log_probs = model(batch_inputs)
        
        # print(f'log_probs ({log_probs.shape}): ')
        loss = F.cross_entropy(log_probs.view(-1, log_probs.size(-1)), batch_target)

        # loss = loss_function(log_probs, batch_target)

        loss.backward()
        optimizer.step()
        if batch_count % 5000 == 0: 
            print(f'Epoch {epoch}, batch {batch_count} loss {loss}')
            context = 'let them for signs'
            context_ix = encode(context)
            with torch.no_grad():
                # Perform inference or validation without gradient calculation
                output = model(torch.tensor(context_ix, dtype=torch.long).unsqueeze(0))
                
                print(f'Context: {context}')
                print(f'Prediction: {decode(torch.argmax(output[0]).item())}')
                print("correct answer: be\n")

        batch_count += 1
        optimizer.zero_grad()





    # total_loss = 0

    # for context, target in data:
    #     context_vector = torch.tensor(context, dtype=torch.long)
    #     target_tensor = torch.tensor(target, dtype=torch.long)
    #     log_probs = model(context_vector).view(-1)
    #     total_loss += loss_function(log_probs, torch.tensor(target))


    #optimize at the end of each epoch
    # optimizer.zero_grad()
    # total_loss.backward()
    # optimizer.step()

torch.save(model.state_dict(), 'cbow_model.pth')


#TESTING
context = 'And God Let the'
context_ix = encode(context)
a = model(torch.tensor(context_ix, dtype=torch.long))

#Print result
# print(f'Raw text: {" ".join(raw_text)}\n')
print(f'Context: {context}\n')

print(f'Prediction: {decode(torch.argmax(a[0]).item())}')
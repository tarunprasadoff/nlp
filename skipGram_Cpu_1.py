import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd

wordToIndexDictionary = ''
with open('wordToIndexDictionary.txt', 'r') as f:
    for i in f.readlines():
        wordToIndexDictionary = i  # string

wordToIndexDictionary = eval(wordToIndexDictionary)  # this is orignal dict with instace dict

# define an empty list
unique_words = []

# open file and read the content in a list
with open('unique_words.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string and add item to the list
        unique_words.append(line[:-1])

vocab_size = len(unique_words)

trainset = pd.read_csv("SkipGram_Train.csv")
testset = pd.read_csv("SkipGram_Test.csv")

embedding_dim = 25
num_epochs = 50
learning_rate = 0.1
batch_size = 1000

trainloader = torch.utils.data.DataLoader(trainset.values, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset.values, batch_size=batch_size, shuffle=True)

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)
        self.activation_function = nn.LogSoftmax(dim = -1)  

    def forward(self, inputs):
        out = self.embeddings(inputs)
        out = self.linear1(out)
        return self.activation_function(out)

    def get_word_embedding(self, word):
        word = torch.tensor([wordToIndexDictionary[word]])
        return self.embeddings(word)

skipGram = SkipGram(vocab_size, embedding_dim)
results = pd.DataFrame(columns=["Epoch","Train Loss","Test Loss"])

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(skipGram.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_train_loss = 0
    total_test_loss = 0

    print(f"Epoch {epoch+1}")

    print("Training:")

    for trainload in enumerate(tqdm(trainloader)):

        y_true = trainload[1][:,1].to(device)

        y_pred = skipGram.forward(trainload[1][:,0].to(device))

        running_train_loss = loss_function(y_pred, y_true)

        optimizer.zero_grad()
        running_train_loss.backward()
        optimizer.step()

        total_train_loss += running_train_loss

    print(f'Train Loss at epoch {epoch+1}: {total_train_loss}')

    print("Testing:")

    for testload in enumerate(tqdm(testloader)):

        y_true = testload[1][:,1]

        y_pred = skipGram.forward(testload[1][:,0])

        running_test_loss = loss_function(y_pred, y_true)

        total_test_loss += running_test_loss

    print(f'Test Loss at epoch {epoch+1}: {total_test_loss}')

    results = results.append({"Epoch":epoch,"Train Loss":total_train_loss,"Test Loss":total_test_loss},ignore_index=True)

    torch.save(skipGram.state_dict, f"Results/SkipGram/Cpu/{str(learning_rate)[2:]}/{epoch+1}.pth")
    results.to_csv(f"Results/SkipGram/Cpu/{str(learning_rate)[2:]}/Loss.csv")
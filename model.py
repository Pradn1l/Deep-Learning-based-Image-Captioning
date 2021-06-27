# -*- coding: utf-8 -*-

#  @author Pradnil S Kamble

import torch 
import torch.nn as nn
import torchvision.models as models

# 1)
class EncoderCNN(nn.Module):
    #embed_size, 
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        # we are not going to train the CNN but use a pretrained model and just
        # fine tune the last layer
        self.train_CNN = train_CNN
        # the CNN model that we will use is the inception model
        # you can read about the auxilary logits in the inception model not important
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        # we are accessing the last linear layer
        # and replace that one with linear, and map that to the embed size
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        # we are removing the last linear layer and send in the in_feature output of CNN
        # and we gonna replace that last linear layer to map it to the embed size
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        # first thing we do is compute features 
        features = self.inception(images)
        # we are not training the entire network that means we are not computing gradients
        # or the backward for CNN 
        
        for name, param in self.inception.named_parameters():
            #check if fc weights in name or fc.bias in name, thats what this fc is called
            # if you print the self inception you could see
            if "fc.weight" in name or "fc.bias" in name: 
                param.requires_grad = True
                # essentially we are just fine tunning 
                # we are setting last layer to be required gradient
                # but the ones before they dont need
                
            else:
                # if you want to train CNN here it would be True too
                # but its gonna be default to just do fine tuning
                param.requires_grad = self.train_CNN
                
        return self.dropout(self.relu(features))
                
# 2) 
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super (DecoderRNN, self).__init__()
        
        # so first we need an embedding layer
        # we gonna map our word to some dimensional space to have a better representation
        # of our word
        self.embed = nn.Embedding(vocab_size, embed_size) #take index and map it there
        # here we send in embed size as input which we first run into embedding
        # its gonna mape into some hidden_size, and we have some number of layers of LSTM 
        # sort of how many LSTMS you want to stack on top of each other
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        # here we take the output from the LSTM which is the hidden_size
        # we map it to some vocab_slize, where each node represents one word in our vocabulary
        self.liner = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    # we call the features from our cnn the return thing
    # and the captions are the target captions thats in our data set
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        # we gonna concatenate features with embeddings along dimension 0
        # essentially we need to add an additional dimension here (unsqueeze 0)
        # so it is viewed as the time step and then we concatenate it with embedding that 
        # already have a time step
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        # where the embeddings are now captionss with the features of the image
        # watch LSTM video to know more
        hiddens, _ = self.lstm(embeddings)
        
        outputs = self.linear(hiddens)
        return outputs
    
# 3) Hook them together
        
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        # initialize CNN and RNN
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs
    
    # what we are doing when LSTM is using the target caption for every time step (outputs)
    # it makes some predictions which we could use for our Loss function
    # but the actual input to that hidden state that we're predicting the target captions
    # will be from the data set the actual captions
    # thats the distinction we have captions for training but not for testing
    # one way to train the model other way where we are actually evaluating 
    # or doing inference on an image that we dont have a target caption for
    
    def caption_image(self, image, vocabulary, max_length=50):
        #max lengthat that can do prediction
        # we are just doing predictions for 50 words here, if you have a longer 
        # sentence you might have to increase that max_lenght
        result_caption = []
        
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            # unsqueeze 0 so we have a dimension for the batch
            # also initialize some states
            states =None #initialize as None i.e. 0 in the beginning
            # these states are gonna be the hidden and the cell states for the LSTM
            
            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                # not unsqueeze because just one image
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1) #we are taking the word with highest probability
                
                result_caption.append(predicted.item())
                # next input 
                x = self.decoderRNN.embed(predicted).unsqueeze(0)
                
                #just to check
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
                
        # In the end we are returning sort of the string and not the indexes corresponding
        # to our prediction
        return [vocabulary.itos[idx] for idx in result_caption]
    
    #thats it for the model move to training
    
        
        

# -*- coding: utf-8 -*-

# @author Pradnil S Kamble

import os  # for loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenization: as the caption is string, to split the string based on space also advance
import torch
from torch.nn.utils.rnn import pad_sequence  # to pad every batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms


# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to a index
# 2. We need to setup a Pytorch dataset to load the data
# 3. Setup padding of every batch (all examples should be
#    of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!


# define spacy 
#download with: python -m spacy download en
spacy_eng = spacy.load("en")

# 2) Build or vocabulary

class Vocabulary: 
    #freq_threshold, if a word is repeated in a vocabulary lets say if is repeated 
    # only once then that is not important, so we can ignore that word
    # so we are saying if a word is not repeated freq_threshold times we ignore it
    def __init__(self, freq_threshold):
        #this is gonna be our dictionary, set some standard values at the begining
        # 0 will be our pad token, 1 will be our start of sentence, UNK unknown token
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        # if a word is not repeated until the threshold it is gonna map to UNK 
        # inverse of that string to index
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3 }
        self.freq_threshold = freq_threshold
        
    # to get the length of our vocabulary
    def __len__(self):
        return len(self.itos)
    
    # we create a static method so no need to initialize a self here
    
    @staticmethod
    def tokenizer_eng(text):
        # separating the text by space and tokeninzing
        # we use spacy to do some advance things 
        # first we lower case everything
        # text is the text that we send in
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
        
        # example if "I love apple" -> ["i", "love", "apple"]
    
    # we are getting list of all the captions same as self.caption.tolist()
    def build_vocabulary(self, sentence_list):
        # count how many times a specific word repeats
        frequencies = {}
        # if its over threshold we include it if not then we ignore it
        # we gonna start with index 4 as we have alread include 3 tokens up
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies: 
                    frequencies[word] = 1
                else: 
                    frequencies[word] +=1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word]=idx
                    self.itos[idx] = word
                    # next one is then gonna be the 5th one
                    idx +=1 
                    
    def numericalize(self, text): 
        # convert the text into numericalize value
        tokenized_text = self.tokenizer_eng(text)
        # call the tokenizer
        
        # if token is included, and surpasses then we do stoi
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
            ]



# 1) we start with the second step

class FlickrDataset(Dataset):
    # root_dir: dir of the images
    # frequency threshold hyper param
    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        # df: data frame
        # image name and caption was separated with comma so it is a csv comma separated value
        self.df = pd.read_csv(caption_file)
        self.transform = transform
        
        # Get image and caption columns, first row name image
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        # Initialize vocabulary and build one
        self.vocab = Vocabulary(freq_threshold)
        # caption.tolist() will send all the captions that we have
        self.vocab.build_vocabulary(self.captions.tolist())
    
    # so that we can have length of our data set
    def __len__(self):
        return len(self.df)
    
    #data loader in pytorch 
    # index inbetween, strictlys less than the length of the actual dataset
    # index is there to say how to obtain a single sample, a single image with corresponding caption
    def __getitem__(self, index):
        #single caption of that image
        caption = self.captions[index]
        img_id = self.imgs[index]
        # actual image based on that image id will be 
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        
        if self.transform is not None:
            # Actual transformations that we might wanna do on our images
            img = self.transform(img)
            
        # we want to convert this in numericalized version
        # stoi is string to index, first ask for the index of start token start of sentence
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        #sort of coverting each word to a index in our vocabulary
        numericalized_caption += self.vocab.numericalize(caption)
        # EOS end token, end of sentence
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        # return the image and a numericalized caption converted to a tensor
        
        return img, torch.tensor(numericalized_caption)
    
# Sequence lengths have to be same, but caption length can be different for 
# different examples, so in our batch we need to have same
# so check the highest possiblity and then pad upto that 
# but we might pad unneccesarily 
# So how to pad so we check what is the longest lenght in our batch 
# and then pad everything to that batch
        
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        # we get list of list of examples 
        # batch is list of examples we wanna have 
        # by each example we get image and caption
        # unsqueeze 0 is the extra dimension for the batch
        imgs = [item[0].unsqueeze(0) for item in batch]
        # torch . catenate, so we catenate all the images at dimensions 0
        imgs = torch.cat(imgs, dim=0)
        # targets are the captions 
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value = self.pad_idx)
        
        # targets are now padded
        return imgs, targets
        
# at the end we want a data loader that is doing all this for us which we can then 
# send in to our model
        
def get_Loader (
        root_folder,
        annotation_file, #catption_file
        transform,
        batch_size = 32,
        num_workers = 8,
        shuffle = True, # when working with time-series data dont do this
        pin_memory = True,
        ):
        
        # just initialize our data set
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)
    # the pad index 
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory = pin_memory,
        # this is where we call our collate function
        collate_fn =MyCollate(pad_idx = pad_idx),
        )
    
    return loader

# define transforms 

def main():
    # transforms = transforms.Compose() to add several to resize your model
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
            ]
        )
    
    dataloader = get_Loader("/Users/DELL/Desktop/MS3/Image Captioning/images", 
                            annotation_file="/Users/DELL/Desktop/MS3/Image Captioning/captions.txt",
                            transform = transform)
    
    for idx, (imgs, captions) in enumerate(dataloader):
        print(imgs.shape)
        print(captions.shape)
        
if __name__ == "__main__":
    main()
    # Out put 
    # torch.Size([32, 3, 224, 224]) rgb 224 x 224, batch 32
    # and torch.Size([29, 32]) 32 example each converted to numerical values of len 29
    # because maximum sequence len was 26 of that batch
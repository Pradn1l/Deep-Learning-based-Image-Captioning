# @author Pradnil S Kamble


import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN

# utils function its pretty basic , print_examples : taking 5 example images
# small test case scenario, it has 5 examples

def train():
    # Dat augmentation
    transform = transforms.Compose(
        [
            transforms.Resize((356,356)),
            # this crop is because inception takes input 299x299
            transforms.RandomCrop((299,299)),
            transforms.ToTensor(),
            # inception uses 0.5 on all of them
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),            
            ]
        )
    
    train_loader, dataset = get_loader(
        root_folder="/Users/DELL/Desktop/MS3/Image Captioning/images",
        annotation_file="/Users/DELL/Desktop/MS3/Image Captioning/captions.txt",
        transform = transform,
        num_workers = 2,
        )
    
    #model configurations not that important
    torch.backends.cudnn.benchmark = True
    # this will give us some performance boost
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    
    # Hyperparameters 
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)  #thats what we get from the get loader
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100
    
    #for tensorboard 
    writer = SummaryWriter("runs/flickr")
    step = 0
    
    # intialize mode, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer= optim.Adam(model.parameters(), lr = learning_rate)
    
    #from the Utils, step for the tensorboard, so that loss function continues where it ended
    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
        
    model.train()
    
    for epoch in range(num_epochs):
    # Uncomment the line below to see a couple of test cases
    # these are our  test examples
    # print_examples(model, device, dataset)
        if save_model:
            checkpoint = {
                # save everything each epoch
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
        
        save_checkpoint(checkpoint)
        
    
    # we need to go through the data loader
    
    for idx, (imgs, captions) in enumerate(train_loader):
        imgs = imgs.to(device)
        captions = captions.to(device)
        # send in captions all except the last one, because we want the model to predict
        # the end token
        outputs = model(imgs, captions[:-1])
        # why we do this explained in sequence to sequence
        # because we are predicting each example for a particular timestep
        # one example might have 20 words, each word has its logits
        # but criterion only expects 2 dimensions
        # if (N, 10) meaning 1 example has 10 probabilities, targets would be N
        # but here we would have (seq_len, N, vocabulary_size) each word has a corresponding
        # probability in vocab size, but targets are just (seq_len,N) so we concatenate both
        # so that we can use each time step as one example
        # caption.reshape (-1) because it is gonna take a single dimension
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.resphape(-1))
        
        writer.add_scalar("Training loss", loss.item(), global_step = step)
        step += 1
        
        optimizer.zero_grad()
        loss.backward(loss)
        optimizer.step()
        
if __name__ == "__main__":
    train()
        
        
    
    
    
    
 


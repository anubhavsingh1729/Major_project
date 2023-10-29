from django.shortcuts import render,redirect
from django.conf import settings
from . models import Post
from . forms import Imageupload

def index(request):
    post = Post.objects.all()
    return render(request,'index.html',{'post':post})

#--------------------------------------------------------------------------------------------------------
import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from imagecaptioning.build_vocab import Vocabulary
from imagecaptioning.model import EncoderCNN, DecoderRNN
from PIL import Image

num_layers = 1
embed_size = 256
hidden_size=512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_path = os.path.join(settings.BASE_DIR,'imagecaptioning/data/vocab.pkl')
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image


def main(img):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    vocab_path = os.path.join(settings.BASE_DIR,'imagecaptioning/data/vocab.pkl')
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # Build models
    encoder = EncoderCNN(embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load("../image_captioning/models/encoder-5-3000.pkl"))
    decoder.load_state_dict(torch.load("../image_captioning/models/decoder-5-3000.pkl"))

    # Prepare an image
    image = load_image(img, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    return(sentence)
    #image = Image.open(img)
    #plt.imshow(np.asarray(image))
#--------------------------------------------------------------------------------------------------------
def caption(image):
    sent = main(image)
    return sent

def uploadimg(request):
    if request.method == 'POST':
        form = Imageupload(request.POST, request.FILES)
        if form.is_valid():
            m = Post()
            m.image = form.cleaned_data['image']
            m.title = caption(m.image.url)
            m.save()
            return redirect('/')
    else:
        form = Imageupload()
    return render(request,'forms.html',{form:'form'})
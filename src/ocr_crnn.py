import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, n_layers=3, dropout=0.2):
        super(CRNN, self).__init__()
        backbone = timm.create_model("resnet152", in_chans=1, pretrained=False)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            512, hidden_size, n_layers, bidirectional=True,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.mapSeq(x)
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        x = self.out(x)
        return x.permute(1, 0, 2)

chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz²³µ¹¼½¾ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝßàáâãäåæçèéêëìíîïñòóôõöøùúûüýÿĀāĂăĄąĆćċČčĐĒēĖėęĚěġĩĪİıĺĽŁŃńňŉŌōŎőŒŘřŚśşŠšŢţŤũŪūŮůŰźŻżŽžƃƉƠơƬƯưǏǒǪǵȂȇȊȍȎȏȓȕȗțȮəɮʃʰˉˢˣΔΡΣΦΩίαβελνπστϋабвгдежиклмнопрстуъыьэײبرل٢٣ڮڽۊᴐᴰᴱᴴᴵᴸᴹᴺᵀᵃᵈᵉᵐᵒᵗᵛᶜᶻᶾḂḎḕḢḻṁṅṫẏẢẤẦầẬẮẶẾỀềỆọỐốồỘỢό⁰⁵⁷⁸⁹ⁿ₁₃₄ₓₘₙₜℇℒΩℱ⅓⅔ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ①②❶➀➁➂➃➄➅➋➌➍➎ⱽ卐ꝋ𝕵-'
vocab_size = len(chars)
char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

crnn_model = CRNN(vocab_size=vocab_size).to(device)
checkpoint = torch.load('../ocr_crnn.pt', map_location=device) # Import crnn model
crnn_model.load_state_dict(checkpoint['model_state_dict'])
crnn_model.eval()
crnn_transform = transforms.Compose([
    transforms.Resize((100, 420)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def decode(encoded_sequences, idx_to_char, blank_char='-'):
    decoded_sequences = []
    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None
        for token in seq:
            if token != 0:
                char = idx_to_char.get(token.item(), '')
                if char != blank_char:
                    if char != prev_char or prev_char == blank_char:
                        decoded_label.append(char)
                prev_char = char
        decoded_sequences.append("".join(decoded_label))
    return decoded_sequences

def run_text_recognition_crnn(image_input) -> list:
    if isinstance(image_input, np.ndarray):
        image_input = Image.fromarray(image_input)
    if image_input.mode != 'RGB':
        image_input = image_input.convert('RGB')
    transformed_image = crnn_transform(image_input).unsqueeze(0).to(device)
    crnn_model.eval()
    with torch.no_grad():
        logits = crnn_model(transformed_image).log_softmax(2)
    predictions = logits.permute(1, 0, 2).argmax(2)
    decoded_text = decode(predictions, idx_to_char)
    
    return decoded_text[0] if decoded_text else ""
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms

def ensure_pil_uint8(img):
    # Accepts PIL or torch.Tensor
    if isinstance(img, torch.Tensor):
        # img: CxHxW
        if img.dtype.is_floating_point():
            img = (img.clamp(0, 1) * 255).to(torch.uint8)
        elif img.dtype != torch.uint8:
            img = img.to(torch.uint8)
        img = to_pil_image(img)
    return img

def patch_image(x):

    B, C, H, W = x.shape
    patch_size = H // 2

    stride = patch_size // 2
    #print(stride)
    patches = []
    for i in range(3):
        for j in range(3):
            patch = x[:, :,
                      stride * i:patch_size + (stride * i),
                      stride * j:patch_size + (stride * j)]
            patches.append(patch)  # [B, C, patch_h, patch_w]
    return patches

def patch_image_3x3(x):
    # x: (B, C, H, W) — assume square or already center-cropped square
    B, C, H, W = x.shape
    patch_h, patch_w = H // 3, W // 3
    patches = []
    for i in range(3):
        for j in range(3):
            ph0, ph1 = i*patch_h, (i+1)*patch_h
            pw0, pw1 = j*patch_w, (j+1)*patch_w
            patches.append(x[:, :, ph0:ph1, pw0:pw1])
    # list of 9 tensors, each (B, C, patch_h, patch_w)
    return patches

def output_to_Tensor(outputs):
    # coerce to a Tensor
    if hasattr(outputs, "last_hidden_state"):           # HF ModelOutput
        cls = outputs.last_hidden_state
    elif isinstance(outputs, dict):                      # dict of tensors
        cls = outputs.get("last_hidden_state", None)
        if cls is None:
            cls = next(v for v in outputs.values() if isinstance(v, torch.Tensor))
    elif isinstance(outputs, (tuple, list)):             # tuple/list -> first tensor
        cls = next(v for v in outputs if isinstance(v, torch.Tensor))
    elif isinstance(outputs, torch.Tensor):              # already a tensor
        cls = outputs
    else:
        raise TypeError(f"Unsupported encoder output: {type(outputs)}")
    return cls

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_features_rad_dino(dataset, image_encoder, radDinoType = 1):
    image_encoder.eval().to(device)

    all_features = []
    feat_dim = None

    with torch.no_grad():
        for i in range(len(dataset)):
            image, label = dataset[i]
            image = image.unsqueeze(0).to(device)
            # Avoid double-rescale warnings: pass PIL/uint8
            #image = ensure_pil_uint8(image)
            
            #patches = patch_image(image)
            #images_stacked = torch.cat(patches, dim=0)

            #print(images_stacked.shape)
            cls_token = image_encoder(image) 
            cls_token = output_to_Tensor(cls_token)

            #if radDinoType == 2:
                #print(cls_token.shape)
                #cls_token = cls_token.last_hidden_state
                #cls_token = cls_token[:, 0] 

            if feat_dim is None:
                print(cls_token.shape)
            #patch_embeddings = patch_tokens.mean(dim=[2, 3])                 

            #feats = torch.cat([cls_token, patch_embeddings], dim=1)          
            feats = cls_token.squeeze(0).cpu()    
            #feats = cls_token.cpu()                                  
            label = torch.as_tensor(label, dtype=torch.float32).cpu()        

            if feat_dim is None:
                feat_dim = feats.shape[0]   # 1536
                print("Feature dim:", feat_dim)

            all_features.append({
                'image_feat': feats,        # CPU, (D,)
                'label': label              # CPU, (C,)
            })

    return all_features, feat_dim

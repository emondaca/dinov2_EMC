import torch.nn as nn
from rad_dino.utils import safetensors_to_state_dict
import torch
import torch.nn.functional as F
from transformers import pipeline
from transformers import AutoModel, AutoConfig
from dinov2.layers import DINOHead

def load_raddinoMaira1(RadDino_src, RadDinoWeights):
    m = torch.hub.load(RadDino_src, "dinov2_vitb14", source="local")
    print(m)
    sd = safetensors_to_state_dict(RadDinoWeights)
    m.load_state_dict(sd, strict=True)
    return m

def load_raddinoMaira2():
    name = "microsoft/rad-dino-maira-2"

    cfg = AutoConfig.from_pretrained(name)
    backbone = AutoModel.from_pretrained(name, config=cfg)
    return backbone

def pool_tokens(x):
    return x[:, 0] if x.dim() == 3 and x.size(1) > 0 else x.mean(dim=1)

def _get_block_module(m, idx: int):

    if hasattr(m, "blocks"):
        return m.blocks[idx]

    if hasattr(m, "encoder") and hasattr(m.encoder, "layer"):
        return m.encoder.layer[idx]
    raise AttributeError("Could not locate transformer blocks on the model.")

def _get_final_norm(m):

    if hasattr(m, "norm"):
        return m.norm

    if hasattr(m, "layernorm"):
        return m.layernorm

    return nn.Identity()

class RadDINOFirst11Extractor(nn.Module):
    def __init__(self, RadDino_src, RadDinoWeights, radDinoType = 1):
        super().__init__()
        self.radDinoType = radDinoType
        if radDinoType == 1:
            self.m = load_raddinoMaira1(RadDino_src, RadDinoWeights)
        else: self.m = load_raddinoMaira2()
        for p in self.m.parameters():
            p.requires_grad = False
        self.m.eval()

    @torch.no_grad()
    def forward(self, x):
        feat = {}

        def hook(_m, _inp, out):
            feat["x"] = out


        layer11_minus1 = _get_block_module(self.m, 10)
        h = layer11_minus1.register_forward_hook(hook)

        try:
            _ = self.m(x)
        finally:
            h.remove()

        return feat["x"]


class RadDINOLastBlockClassifier(nn.Module):
    def __init__(self, RadDino_src, RadDinoWeights, num_classes, in_dim = 768, radDinoType = 1):
        super().__init__()
        if radDinoType == 1:
            m = load_raddinoMaira1(RadDino_src, RadDinoWeights)
        else: m = load_raddinoMaira2()

        self.last_block = _get_block_module(m, 11)  
        self.final_norm = _get_final_norm(m)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_dim, num_classes)
        )


    def forward(self, tokens_11):
        x = self.last_block(tokens_11)  
        if isinstance(x, (tuple, list)):
            x = x[0]
            
        x = self.final_norm(x)   
        if isinstance(x, (tuple, list)):
            x = x[0]      

        x = pool_tokens(x)             
        return self.classifier(x)   



# RadDino PreTrained Head ---------------------------------------
class radDino_LastBlock_PlusHead(nn.Module):
    def __init__(self, fusion_dim,RadDino_src, RadDinoWeights, head_dir, num_classes=3):
        super().__init__()

        m = load_raddinoMaira1(RadDino_src, RadDinoWeights)

        self.last_block = _get_block_module(m, 11)  
        self.final_norm = _get_final_norm(m)

        self.rad_dino_head_gh = DINOHead(
            in_dim=fusion_dim,
            out_dim=65536,
            hidden_dim=2048,
            bottleneck_dim=256,
            nlayers=3,
        )
        head_state_dict = safetensors_to_state_dict(head_dir)
        self.rad_dino_head_gh.load_state_dict(head_state_dict, strict=True)

        #old_last_mlp = self.rad_dino_head_gh.mlp[2]
        #in_features_last_mlp = old_last_mlp.in_features  

        #new_out_dim = 768 

        #self.rad_dino_head_gh.mlp[2] = nn.Linear(
        #    in_features_last_mlp,
        #    num_classes
        #)
        
        #self.classifier = nn.Sequential(
        #    nn.Dropout(0.3),
        #    nn.Linear(new_out_dim, num_classes)
        #)

        in_features = self.rad_dino_head_gh.last_layer.in_features
        self.rad_dino_head_gh.last_layer = nn.Linear(in_features, num_classes)

        
        for name, param in self.rad_dino_head_gh.named_parameters():
            param.requires_grad = False  
        for param in self.rad_dino_head_gh.last_layer.parameters():
            param.requires_grad = True   

        #for param in self.rad_dino_head_gh.mlp[0].parameters():
        #    param.requires_grad = False

        #for param in self.rad_dino_head_gh.mlp[1].parameters():
        #    param.requires_grad = False

        #for param in self.rad_dino_head_gh.mlp[2].parameters():
        #    param.requires_grad = True

        #for param in self.rad_dino_head_gh.last_layer.parameters():
        #    param.requires_grad = True

    def forward(self, tokens_11):
        x = self.last_block(tokens_11)      
        x = self.final_norm(x)       
        x = pool_tokens(x)
        x = self.rad_dino_head_gh(x)   
        return x







# This demo needs to be run from the repo folder.
# python demo/fake_gan/run.py
import os
import random
import gradio as gr
import itertools
from PIL import Image, ImageFont, ImageDraw
import sys

from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
import torchvision.transforms as T
import gradio as gr


sys.path.append("source")

# import DirectedDiffusion



class LocalBlend:
    
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        affine = False
        if affine:
            affine_transfomer = T.RandomAffine(degrees=(0, 0), translate=(0, 0), scale=(1, 1))
            maps[1] = affine_transfomer(maps[1])
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th


        
        
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

def save_attn(attn_repalce_new, dim=0):
        attn_repalce_new_view = attn_repalce_new[0,:,:,dim]
        image = 255 * attn_repalce_new_view / attn_repalce_new_view.max()
        image = image.cpu()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image.save('test.png')
        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
        
    def modify_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] >= 16 ** 2 and place_in_unet == 'up':
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        global mask2
        global attn_editor_bundle
        global len_tokens 
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            # self.all_cross_attn.append(attn[0,attn.shape[0]//2:])
            if attn[0].ndim ==3:
                attn[0] = attn[0].unsqueeze(0)
            attn_base, attn_repalce = attn[0], attn[1:]
            if attn.shape[0] > 1:
                k = 1
            else:
                k = 0

            if is_cross:
                affine = True
                replace = attn_editor_bundle['replace_attention']
                load = True

                if not os.path.exists(attn_editor_bundle["prompt"][0]+ " " + str(attn_editor_bundle["seed"])):
                    attn_editor_bundle["save_latents"] = True
                    os.mkdir(attn_editor_bundle["prompt"][0]+ " " + str(attn_editor_bundle["seed"]))
                if attn_editor_bundle["save_latents"]:
                    attn_editor_bundle["edit_latents"] = False
                    self.all_cross_attn.append(attn[0])
                    affine = False
                    replace = False
                    load = False
                # print('cross ' , attn.shape , place_in_unet, ' ', self.cur_step)
                # alpha_words = self.cross_replace_alpha[self.cur_step]
                # attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn_repalce_new = attn_base

                if load:
                    cur_map = self.maps[0]
                    self.maps.pop(0)
                    attn_repalce_new = cur_map
                    attn_repalce_new = attn_repalce_new.unsqueeze(0)
                # print('attn ' , attn[0].shape)
                # print(cur_map.shape)

                if affine:
                    attn_repalce_new = attn_repalce_new.reshape(-1, int(np.sqrt(attn_repalce_new.shape[2])), int(np.sqrt(attn_repalce_new.shape[2])), attn_repalce_new.shape[-1])
                    # save_attn(attn_repalce_new)
                    attn_repalce_new = attn_repalce_new.permute(0, 3, 1, 2)
                    # attn_repalce_new2 = attn_repalce_new.deepcopy()
                    # attn_repalce_new = affine_transfomer(attn_repalce_new)
                    if attn_editor_bundle["translate"][0] != 0.0:
                        attn_repalce_new = T.functional.affine(attn_repalce_new, translate=(float(attn_editor_bundle["translate"][0]*attn_repalce_new.shape[2]/64), 0), scale=1, shear=[0, 0], angle = 0)
                    if attn_editor_bundle["scale"][0] != 1.0:
                        attn_repalce_new = T.functional.affine(attn_repalce_new, translate=(0, 0), scale=float(attn_editor_bundle["scale"][0]), shear=[0, 0], angle = 0)
                    if attn_editor_bundle["shear"][0] != 0.0:
                        attn_repalce_new = T.functional.affine(attn_repalce_new, translate=(0, 0), scale=1, shear=[float(attn_editor_bundle["shear"][0]), 0], angle = 0)
                    if attn_editor_bundle["angle"][0] != 0.0:
                        attn_repalce_new = T.functional.affine(attn_repalce_new, translate=(0, 0), scale=1, shear=[0, 0], angle = float(attn_editor_bundle["angle"][0]))

                    attn_repalce_new = attn_repalce_new.permute(0, 2, 3, 1)
                    attn_repalce_new = attn_repalce_new.reshape(attn[0].shape)
                # 
                if replace and  self.cur_step >= attn_editor_bundle["step_to_replace"]:
                    # attn[1:,:,:,:] = self.replace_cross_attention(attn_repalce_new.squeeze(0), attn_repalce_new)
                    for ids in attn_editor_bundle["edit_index"][0]:
                        # index = in_token_ids[ids]
                        if attn_repalce_new.ndim == 3:
                            attn_repalce_new = attn_repalce_new.unsqueeze(0)
                        attn[k,:,:,int(ids)] = attn_repalce_new[:,:,:,int(ids)]
                    
                    attn[k:,:,:,len_tokens :len_tokens +int(attn_editor_bundle["num_trailing_attn"][0])] = attn_repalce_new[:,:,:,len_tokens :len_tokens +int(attn_editor_bundle["num_trailing_attn"][0])]
                    # attn[1:,:,:,10:] = torch.zeros_like(attn[1:,:,:,10:])
                    # attn_repalce_new[:,:,:,5:10]
            else:
                # print('self ' , attn.shape , place_in_unet, ' ', self.cur_step)
                # attn_repalce_new = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
                # attn_repalce_new = attn_repalce
                # mask = torch.zeros_like(attn_repalce_new)
                # mask[:,:,int(attn_repalce_new.shape[2]*2000/4096):int(attn_repalce_new.shape[2]*3500/4096),int(attn_repalce_new.shape[3]*1000/4096):int(attn_repalce_new.shape[3]*3500/4096)] = 1.0
                # mask2 = 255*mask.squeeze(0)[0,:,:]

                # mask3 = attn_repalce_new.mean(1, keepdim=False).sum(1, keepdim=False)
                # attn_mask = mask3.reshape(1, int(np.sqrt(mask3.shape[1])), int(np.sqrt(mask3.shape[1])))
                # attn_mask = attn_mask.unsqueeze(1).repeat(1, 3, 1, 1).int().float()

                # attn_repalce_new = attn_repalce_new * mask
                affine = False
                if affine and self.cur_step >= 10 :
                    # attn_repalce_new = attn_repalce_new.squeeze(0)
                    # attn_repalce_new2 = attn_repalce_new.deepcopy()
                    # attn_repalce_new = affine_transfomer(attn_repalce_new)
                    # if self.cur_step and show and place_in_unet == 'up':
                    #     show_self_attention_comp(controller, res=16, from_where=(place_in_unet,), select = 1)
                    #     show = False
                    attn_repalce_new = attn_repalce_new.reshape(-1, attn_repalce_new.shape[2], int(np.sqrt(attn_repalce_new.shape[2])), int(np.sqrt(attn_repalce_new.shape[2])))
                    attn_repalce_new = T.functional.affine(attn_repalce_new, translate=(-10, 0), scale=1.0, shear=[0, 0], angle = 0)
                    attn_repalce_new = attn_repalce_new.permute(0, 2, 3, 1)
                    attn_repalce_new = attn_repalce_new.reshape(-1, int(attn_repalce_new.shape[3]), int(np.sqrt(attn_repalce_new.shape[3])), int(np.sqrt(attn_repalce_new.shape[3])))
                    
                    attn_repalce_new = T.functional.affine(attn_repalce_new, translate=(-10, 0), scale=1.0, shear=[0, 0], angle = 0)
                    
                    attn_repalce_new = attn_repalce_new.permute(0, 2, 3, 1)
                    attn_repalce_new = attn_repalce_new.reshape(attn[1:].shape)
                    # attn[1:,0,int(attn_repalce_new.shape[2]*0/4096):int(attn_repalce_new.shape[2]*1800/4096),int(attn_repalce_new.shape[3]*0/4096):int(attn_repalce_new.shape[3]*1800/4096)] =  attn_repalce_new [:,0,int(attn_repalce_new.shape[2]*0/4096):int(attn_repalce_new.shape[2]*1800/4096),int(attn_repalce_new.shape[3]*0/4096):int(attn_repalce_new.shape[3]*1800/4096)]
                # attn_repalce_new2 = self.modify_self_attention(attn_base, attn_repalce_new, place_in_unet)
                    # attn[1:,:,:,:] = attn_repalce_new*mask + attn[1:,:,:,:]*(1-mask)
                    attn[1:,:,:,:] = attn_repalce_new
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        # self.all_cross_attn = []
        # self.maps = torch.load('./cross_maps/'+str(self.cur_step+1)+'.pt')
                
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller

def saveimage(self, latent):
    latents = 1 / 0.18215 * latent.detach()
    image = self.model.vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype(np.uint8)
    img = Image.fromarray(image)
    img.save("test.png")
    return img


def saveimagegrad(self, latent):
    latents = 1 / 0.18215 * latent.detach()
    image = self.model.vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
    image = (image * 255).astype(np.uint8)
    img = Image.fromarray(image)
    img.save("test.png")
    return img

def savelatent(self, latent):
    latents = 1 / 0.18215 * latent.detach()
    # image = self.model.vae.decode(latents)['sample']
    # image = (image / 2 + 0.5).clamp(0, 1)
    image = latents.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype(np.uint8)
    img = Image.fromarray(image)
    img.save("testlatent.png")
    return img

def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(attn_editor_bundle["prompt"]), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_word)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(attn_editor_bundle["prompt"][select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))

def view_image_ui(images):
    return images
def show_cross_attention_ui(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(attn_editor_bundle["prompt"][select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)+20):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        if i < len(tokens):
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        else:
            image = ptp_utils.text_under_image(image, decoder(int(tokens[-1])))
        images.append(image)
    # ptp_utils.to_images(np.stack(images, axis=0))
    images = ptp_utils.to_images(np.stack(images, axis=0), num_rows=3)
    return images

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))


### null text inversion
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image



@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image', 
    attn_editor_bundle = {}
):
    global all_attns
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    all_cross = []
    all_attns = []
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        controller.all_cross_attn =[]
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        # load = True
        if not attn_editor_bundle["save_latents"] and os.path.exists(attn_editor_bundle["prompt"][0] +" " + str(attn_editor_bundle["seed"])):
            controller.maps = torch.load(attn_editor_bundle["prompt"][0] +" " + str(attn_editor_bundle["seed"])+'/'+str(controller.cur_step+1)+'.pt')
        # latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        latents = ptp_utils.diffusion_step_modified(model, controller, latents, context, t, guidance_scale, low_resource=False, attn_editor_bundle=attn_editor_bundle)
        all_attns.append(show_cross_attention_ui(controller, res=16, from_where=("up", "down"), select = 0))
        # save = False
        if attn_editor_bundle["save_latents"] and os.path.exists(attn_editor_bundle["prompt"][0] + " " + str(attn_editor_bundle["seed"])) and hasattr(controller, 'all_cross_attn'):
            # if not os.path.exists(attn_editor_bundle["prompt"][0]):
            #     os.mkdir(attn_editor_bundle["prompt"][0])
            torch.save(controller.all_cross_attn,attn_editor_bundle["prompt"][0]+" " + str(attn_editor_bundle["seed"])+"/"+str(controller.cur_step)+'.pt')
            # all_cross.append(controller.all_cross_attn)
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None,  attn_editor_bundle = {}, verbose=True):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, x_t = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator, uncond_embeddings=uncond_embeddings)

        print("with prompt-to-prompt")
    else:
        attn_editor_bundle["edit_latents"] = False
        images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings, attn_editor_bundle =attn_editor_bundle)
        images_to_draw = ptp_utils.to_images(images)
        # images2, x_t2 = text2image_ldm_stable_second(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings)
    # # if verbose:
    #     ptp_utils.view_images(images)
    return images, x_t, images_to_draw

def run_and_display2(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True, attn_editor_bundle = {}):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        # images, x_t = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator, uncond_embeddings=uncond_embeddings)
        print("with prompt-to-prompt")

    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings, attn_editor_bundle = attn_editor_bundle)
    if verbose:
        images2 = ptp_utils.to_images(images)
    return images, x_t, images2



# prompt
# boundingbox
# prompt indices for region
# number of trailing attention
# number of DD steps
# gaussian coefficient
# seed
EXAMPLES = [
    [
        "A painting of a tiger, on the wall in the living room",
        "0.2,0.6,0.0,0.5",
        "1,5",
        5,
        15,
        1.0,
        2094889,
        0,
    ],
    [
        "A photo of a dog in a beach",
        "0.0,0.5,0.0,0.5",
        "5",
        10,
        20,
        5.0,
        8880,
        5,
    ],
    [
        "A red cube above a blue sphere",
        "0.4,0.7,0.0,0.5 0.4,0.7,0.5,1.0",
        "2,3 6,7",
        10,
        20,
        1.0,
        1213698,
        0,
    ],
]


# model_bundle = DirectedDiffusion.AttnEditorUtils.load_all_models(
    # model_path_diffusion="assets/models/stable-diffusion-v1-4"
# )

ALL_OUTPUT = []

def attention_run(
    in_prompt,
    in_bb,
    in_token_ids,
    in_slider_trailings,
    in_slider_ddsteps,
    in_slider_gcoef,
    in_seed,
    step_to_replace,
    edit_latents,
    save_latents,
    is_draw_bbox,
    mask,
    translate,
    scale,
    shear,
    angle
):
    global x_t
    global attn_editor_bundle
    str_arg_to_val = lambda arg, f: [
        [f(b) for b in a.split(",")] for a in arg.split(" ")
    ]
    roi = str_arg_to_val(in_bb, float)
    attn_editor_bundle = {
        "edit_index": str_arg_to_val(in_token_ids, int),
        "roi": roi,
        "seed": int(in_seed),
        "num_trailing_attn": [in_slider_trailings] * len(roi),
        "num_affected_steps": in_slider_ddsteps,
        "noise_scale": [in_slider_gcoef] * len(roi),
        "edit_latents": edit_latents,
        "replace_attention": replace_attention,
        "step_to_replace": step_to_replace,
        "save_latents": save_latents,
        "prompt": [in_prompt],
        "translate": str_arg_to_val(translate, float)[0],
        "scale": str_arg_to_val(scale, float)[0],
        "shear": str_arg_to_val(shear, float)[0],
        "angle": str_arg_to_val(angle, float)[0]
    }
    # img = DirectedDiffusion.Diffusion.stablediffusion(
    #     model_bundle,
    #     attn_editor_bundle=attn_editor_bundle,
    #     guidance_scale=7.5,
    #     prompt=in_prompt,
    #     steps=50,
    #     seed=in_seed,
    #     is_save_attn=False,
    #     is_save_recons=False,
    # )
    # if is_draw_bbox and in_slider_ddsteps > 0:
    #     for r in roi:
    #         x0, y0, x1, y1 = [int(r_ * 512) for r_ in r]
    #         image_editable = ImageDraw.Draw(img)
    #         image_editable.rectangle(
    #             xy=[x0, x1, y0, y1], outline=(255, 0, 0, 255), width=5
    #         )




    # g_cpu = torch.Generator().manual_seed(8888)
    g_cpu = torch.Generator().manual_seed(int(in_seed))
    # g_cpu = torch.Generator().manual_seed(1234)
    # prompts = ["A painting of a squirrel eating a burger"]
    prompts = [in_prompt]
    controller = AttentionStore()
    image, x_t, image_to_draw = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu,  attn_editor_bundle = attn_editor_bundle)
    # mask2 = gr.Image(value=image_to_draw, label="Image for brushing with mask1", show_label=False, elem_id="img2maskimg1", interactive=True, type="pil", tool="sketch", image_mode="RGBA").style(height=480)
    # mask.change(change_image, inputs=[mask2], outputs=[mask2])
    return image_to_draw


def resize_mask(mask):
    resized_mask = mask['mask'].resize((64,64), Image.NEAREST)
    return torch.from_numpy(np.array(resized_mask)).permute(2,0,1).cuda()
def attention_edit(
    in_prompt,
    in_bb,
    in_token_ids,
    in_slider_trailings,
    in_slider_ddsteps,
    in_slider_gcoef,
    in_seed,
    step_to_replace,
    edit_latents,
    replace_attention,
    save_latents,
    is_draw_bbox,
    mask,
    translate,
    scale,
    shear,
    angle
):
    global mask2
    global attn_editor_bundle
    global x_t
    global len_tokens
    len_tokens = len(tokenizer.encode(in_prompt))
    str_arg_to_val = lambda arg, f: [
        [f(b) for b in a.split(",")] for a in arg.split(" ")
    ]
    roi = str_arg_to_val(in_bb, float)
    attn_editor_bundle = {
        "edit_index": str_arg_to_val(in_token_ids, int),
        "roi": roi,
        "seed": int(in_seed),
        "num_trailing_attn": [in_slider_trailings] * len(roi),
        "num_affected_steps": in_slider_ddsteps,
        "noise_scale": [in_slider_gcoef] * len(roi),
        "edit_latents": edit_latents,
        "replace_attention": replace_attention,
        "step_to_replace": step_to_replace,
        "save_latents": save_latents,
        "prompt": [in_prompt],
        "translate": str_arg_to_val(translate, float)[0],
        "scale": str_arg_to_val(scale, float)[0],
        "shear": str_arg_to_val(shear, float)[0],
        "angle": str_arg_to_val(angle, float)[0]
    }
    # img = DirectedDiffusion.Diffusion.stablediffusion(
    #     model_bundle,
    #     attn_editor_bundle=attn_editor_bundle,
    #     guidance_scale=7.5,
    #     prompt=in_prompt,
    #     steps=50,
    #     seed=in_seed,
    #     is_save_attn=False,
    #     is_save_recons=False,
    # )
    # if is_draw_bbox and in_slider_ddsteps > 0:
    #     for r in roi:
    #         x0, y0, x1, y1 = [int(r_ * 512) for r_ in r]
    #         image_editable = ImageDraw.Draw(img)
    #         image_editable.rectangle(
    #             xy=[x0, x1, y0, y1], outline=(255, 0, 0, 255), width=5
    #         )




    # g_cpu = torch.Generator().manual_seed(8888)
    g_cpu = torch.Generator().manual_seed(int(in_seed))
    # g_cpu = torch.Generator().manual_seed(1234)
    # prompts = ["A painting of a squirrel eating a burger"]
    # prompts = ["A photo of a dog in a beach"]
    # controller = AttentionStore()
    # image, x_t, image_to_draw = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu)
    # if draw_mask:
    #     mask2 = gr.Image(value=image_to_draw, label="Image for brushing with mask1", show_label=False, elem_id="img2maskimg1", interactive=True, type="pil", tool="sketch", image_mode="RGBA").style(height=480)

    # show_cross_attention(controller, res=16, from_where=("up", "down"))

    # prompts = ["A painting of a squirrel eating a burger",
    #         "A painting of a squirrel eating a burger"]
    # prompts = [in_prompt, in_prompt]
    prompts = [in_prompt]
    mask2 = mask
    # mask2 = edit_mask(mask2)
    # prompts = ["A photo of a dog in a beach",
    # "A photo of a dog in a beach"]
    # controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=0.4)
    controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=1.0, self_replace_steps=1.0)
    images2, x_t2, images_to_show = run_and_display2(prompts, controller, latent=x_t, run_baseline=True, attn_editor_bundle=attn_editor_bundle)
    # show_cross_attention(controller, res=16, from_where=("up", "down"), select = 1)
    # show_self_attention_comp(controller, res=16, from_where=("up", "down"), select = 1)
    return images_to_show



def run_it(
    in_prompt,
    in_bb,
    in_token_ids,
    in_slider_trailings,
    in_slider_ddsteps,
    in_slider_gcoef,
    in_seed,
    step_to_replace,
    edit_latents,
    replace_attention,
    is_draw_bbox,
    save_latents,
    mask,
    translate,
    scale,
    shear,
    angle,
    progress=gr.Progress(),
):
    global ALL_OUTPUT
    global num_trailing_attn
    num_affected_steps = [in_slider_ddsteps]
    noise_scale = [in_slider_gcoef]
    num_trailing_attn = [in_slider_trailings]
    # if draw_mask:
    #     num_affected_steps = [5, 10]
    #     noise_scale = [1.0, 1.5, 2.5]
    #     num_trailing_attn = [10, 20, 30, 40]

    param_list = [num_affected_steps, noise_scale, num_trailing_attn]
    param_list = list(itertools.product(*param_list))

    results = []
    progress(0, desc="Starting...")
    for i, element in enumerate(progress.tqdm(param_list)):
        print("=========== Arguments ============")
        print("Prompt:", in_prompt)
        # print("BoundingBox:", in_bb)
        print("Token indices:", in_token_ids)
        print("Num Trialings:", element[2])
        # print("Num DD steps:", element[0])
        # print("Gaussian coef:", element[1])
        print("Seed:", in_seed)
        # print("Prompt:", in_prompt)
        print("===================================")
        img = attention_run(
            in_prompt=in_prompt,
            in_bb=in_bb,
            in_token_ids=in_token_ids,
            in_slider_trailings=element[2],
            in_slider_ddsteps=element[0],
            in_slider_gcoef=element[1],
            in_seed=in_seed,
            step_to_replace=step_to_replace,
            edit_latents=edit_latents,
            save_latents = save_latents,
            is_draw_bbox=is_draw_bbox,
            mask = mask,
            translate = translate,
            scale = scale,
            shear = shear,
            angle = angle
        )

    return img


def edit_it(
    in_prompt,
    in_bb,
    in_token_ids,
    in_slider_trailings,
    in_slider_ddsteps,
    in_slider_gcoef,
    in_seed,
    step_to_replace,
    edit_latents,
    replace_attention,
    is_draw_bbox,
    save_latents,
    mask,
    translate,
    scale,
    shear,
    angle,
    progress=gr.Progress(),
):
    global ALL_OUTPUT
    num_affected_steps = [in_slider_ddsteps]
    noise_scale = [in_slider_gcoef]
    num_trailing_attn = [in_slider_trailings]
    # if draw_mask:
    #     num_affected_steps = [5, 10]
    #     noise_scale = [1.0, 1.5, 2.5]
    #     num_trailing_attn = [10, 20, 30, 40]

    param_list = [num_affected_steps, noise_scale, num_trailing_attn]
    param_list = list(itertools.product(*param_list))

    results = []
    progress(0, desc="Starting...")
    for i, element in enumerate(progress.tqdm(param_list)):
        print("=========== Arguments ============")
        print("Prompt:", in_prompt)
        print("Token indices:", in_token_ids)
        print("Num Trialings:", element[2])
        print("Num DD steps:", element[0])
        print("Gaussian coef:", element[1])
        print("Seed:", in_seed)
        print("===================================")
        img = attention_edit(
            in_prompt=in_prompt,
            in_bb=in_bb,
            in_token_ids=in_token_ids,
            in_slider_trailings=element[2],
            in_slider_ddsteps=element[0],
            in_slider_gcoef=element[1],
            in_seed=in_seed,
            step_to_replace=step_to_replace,
            edit_latents=edit_latents,
            replace_attention=replace_attention,
            save_latents = save_latents,
            is_draw_bbox=is_draw_bbox,
            mask = mask,
            translate = translate,
            scale = scale,
            shear = shear,
            angle = angle
        )
        results.append(
            (
                img,
                "#Trailing:{},#DDSteps:{},GaussianCoef:{}".format(
                    element[2], element[0], element[1]
                ),
            )
        )
    ALL_OUTPUT += results
    return ALL_OUTPUT

def clean_gallery():
    global ALL_OUTPUT
    ALL_OUTPUT = []
    return ALL_OUTPUT

def show_attention():
    controller = AttentionReplace(attn_editor_bundle["prompt"], NUM_DIFFUSION_STEPS, cross_replace_steps=1.0, self_replace_steps=1.0)
    show_cross_attention(controller, res=16, from_where=("up", "down"), select = 1)

def show_attention_ui():
    global all_attns
    results = []
    for i in range(len(all_attns)):
        results.append(
            (
                all_attns[i],
                "#Step:{}".format(i),
            )
        )
    return results
    


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Attention Editor
    """
    )
    with gr.Row(variant="panel"):
        with gr.Column(variant="compact"):
            in_prompt = gr.Textbox(
                label="Enter your prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
            ).style(
                container=False,
            )
            with gr.Row(variant="compact"):
                in_bb = gr.Textbox(
                    label="Bounding box",
                    show_label=True,
                    max_lines=1,
                    placeholder="e.g., 0.1,0.5,0.3,0.6",
                    visible=False
                )
                in_token_ids = gr.Textbox(
                    label="Token indices",
                    show_label=True,
                    max_lines=1,
                    placeholder="e.g., 1,2,3",
                )
                # replace_step = gr.Textbox(
                #     label="replace step",
                #     show_label=True,
                #     max_lines=1,
                #     placeholder="e.g., 5",
                # )
                in_seed = gr.Number(
                    value=2483964026821236, label="Random seed", interactive=True
                )
                step_to_replace = gr.Number(
                    value=0, label="step_to_replace", interactive=True
                )
            with gr.Row(variant="compact"):
                translate = gr.Textbox(value="0.0", label="translate", interactive=True)
                shear = gr.Textbox(value = "0.0", label="shear", interactive=True)
                scale = gr.Textbox(value="1.0", label="scale", interactive=True)
                angle = gr.Textbox(value="0.0", label="angle", interactive=True)

            with gr.Row(variant="compact"):
                save_latents = gr.Checkbox(
                    value=False,
                    label="save latents for editing?",
                )
                is_draw_bbox = gr.Checkbox(
                    value=True,
                    label="To draw the bounding box?",
                )
                edit_latents = gr.Checkbox(
                    value=True,
                    label="To transform the latent",
                )
                replace_attention = gr.Checkbox(
                    value=True,
                    label="To replace attention maps?",
                )
            with gr.Row(variant="compact"):
                in_slider_trailings = gr.Slider(
                    minimum=0, maximum=77, value=10, step=1, label="#trailings"
                )
                in_slider_ddsteps = gr.Slider(
                    minimum=0, maximum=30, value=10, step=1, label="#DDSteps", visible=False
                )
                in_slider_gcoef = gr.Slider(
                    minimum=0, maximum=10, value=1.0, step=0.1, label="GaussianCoef", visible=False
                )
            
            with gr.Row(variant="compact"):
                btn_run = gr.Button("Generate image").style(full_width=True)
                btn_edit = gr.Button("Edit image").style(full_width=True)
                btn_clean = gr.Button("Clean Gallery").style(full_width=True)
                btn_show_attention = gr.Button("show attention").style(full_width=True)

            # gr.Markdown(
            #     """ Note:
            #     1) Please click one of the examples below for quick setup.
            #     2) if #DDsteps==0, it means the SD process runs without DD.
            #     """
            # )
        # image = Image.new('RGB', (512, 512))
    mask = gr.Image(label="Image for brushing with mask1", show_label=False, elem_id="img2maskimg1", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA").style(height=512)
    # global all_attns
    # all_attns = []
        # gallery = gr.Gallery(
        #     label="Generated images", show_label=False, elem_id="gallery"
        # ).style(grid=[2], height="auto")

    # create a mask that user can draw in gradio
    with gr.Row(variant="panel"):
        with gr.Column(variant="compact"):
                    gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery").style(grid=[2], height="auto")
                    gallery_attention = gr.Gallery(label="attention images", show_label=True, elem_id="attentions").style(grid=[5], height="auto")
            
        args = [
            in_prompt,
            in_bb,
            in_token_ids,
            in_slider_trailings,
            in_slider_ddsteps,
            in_slider_gcoef,
            in_seed,
            step_to_replace,
            edit_latents,
            replace_attention,
            is_draw_bbox,
            save_latents,
            mask,
            translate,
            scale,
            shear,
            angle
        ]
        
        btn_run.click(run_it, inputs=args, outputs=mask)
        btn_edit.click(edit_it, inputs=args, outputs=gallery)
        btn_clean.click(clean_gallery, outputs=gallery)
        btn_show_attention.click(show_attention_ui, outputs=gallery_attention)
    examples = gr.Examples(
        examples=EXAMPLES,
        inputs=args,
    )

if __name__ == "__main__":
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    MY_TOKEN = ''
    LOW_RESOURCE = False 
    NUM_DDIM_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(device)
    # try:
    #     ldm_stable.disable_xformers_memory_efficient_attention()
    # except AttributeError:
    #     print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer
    NUM_DIFFUSION_STEPS = 50
    # prompts = ["A photo of a dog in a beach",
    # "A photo of a dog in a beach"]
    # prompts = ["A photo of a dog in a beach"]
    demo.queue().launch()

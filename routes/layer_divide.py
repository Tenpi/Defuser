import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage import color
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import copy
import pickle
import psd_tools
from psd_tools.psd import PSD
import pytoshop
from pytoshop import layers
from pytoshop.enums import BlendMode
from .segmentate import get_raw_mask
from .functions import get_models_dir, next_index
import pathlib
import shutil

dirname = os.path.dirname(__file__)
if "_internal" in dirname: dirname = os.path.join(dirname, "../")
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def assign_tile(row, tile_width, tile_height):
    tile_x = row["x_l"] // tile_width
    tile_y = row["y_l"] // tile_height
    return f"tile_{tile_y}_{tile_x}"

def get_foreground(img, h_split, v_split, n_cluster, alpha, th_rate):
    df = rgb2df(img)
    image_width = img.shape[1] 
    image_height = img.shape[0]             
    mask = get_raw_mask(img)
    mask = (mask * 255).astype(np.uint8)
    mask = mask.repeat(3, axis=2)

    num_horizontal_splits = h_split
    num_vertical_splits = v_split
    tile_width = image_width // num_horizontal_splits
    tile_height = image_height // num_vertical_splits

    df["tile"] = df.apply(assign_tile, args=(tile_width, tile_height), axis=1)

    cls = MiniBatchKMeans(n_clusters=n_cluster, batch_size=100)
    cls.fit(df[["r","g","b"]])
    df["label"] = cls.labels_

    mask_df = rgb2df(mask)
    mask_df["bg_label"] = (mask_df["r"] > alpha) & (mask_df["g"] > alpha) & (mask_df["b"] > alpha)

    img_df = df.copy()
    img_df["bg_label"] = mask_df["bg_label"]
    img_df["label"] = img_df["label"].astype(str) + "-" + img_df["tile"]
    bg_rate = img_df.groupby("label").sum()["bg_label"]/img_df.groupby("label").count()["bg_label"]
    img_df["bg_cls"] = (img_df["label"].isin(bg_rate[bg_rate > th_rate].index)).astype(int)
    img_df["a"] = 255

    bg_df = img_df[img_df["bg_cls"] == 0]
    fg_df = img_df[img_df["bg_cls"] != 0] 

    return [fg_df, bg_df]

def skimage_rgb2lab(rgb):
    return color.rgb2lab(rgb.reshape(1,1,3))

def rgb2df(img):
  h, w, _ = img.shape
  x_l, y_l = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
  r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
  df = pd.DataFrame({
      "x_l": x_l.ravel(),
      "y_l": y_l.ravel(),
      "r": r.ravel(),
      "g": g.ravel(),
      "b": b.ravel(),
  })
  return df

def mask2df(mask):
  h, w = mask.shape
  x_l, y_l = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
  flg = mask.astype(int)
  df = pd.DataFrame({
      "x_l_m": x_l.ravel(),
      "y_l_m": y_l.ravel(),
      "m_flg": flg.ravel(),
  })
  return df

def rgba2df(img):
  h, w, _ = img.shape
  x_l, y_l = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
  r, g, b, a = img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]
  df = pd.DataFrame({
      "x_l": x_l.ravel(),
      "y_l": y_l.ravel(),
      "r": r.ravel(),
      "g": g.ravel(),
      "b": b.ravel(),
      "a": a.ravel()
  })
  return df

def hsv2df(img):
    x_l, y_l = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij")
    h, s, v = np.transpose(img, (2, 0, 1))
    df = pd.DataFrame({"x_l": x_l.flatten(), "y_l": y_l.flatten(), "h": h.flatten(), "s": s.flatten(), "v": v.flatten()})
    return df

def df2rgba(img_df):
  r_img = img_df.pivot_table(index="x_l", columns="y_l",values= "r").reset_index(drop=True).values
  g_img = img_df.pivot_table(index="x_l", columns="y_l",values= "g").reset_index(drop=True).values
  b_img = img_df.pivot_table(index="x_l", columns="y_l",values= "b").reset_index(drop=True).values
  a_img = img_df.pivot_table(index="x_l", columns="y_l",values= "a").reset_index(drop=True).values
  df_img = np.stack([r_img, g_img, b_img, a_img], 2).astype(np.uint8)
  return df_img

def df2bgra(img_df):
  r_img = img_df.pivot_table(index="x_l", columns="y_l",values= "r").reset_index(drop=True).values
  g_img = img_df.pivot_table(index="x_l", columns="y_l",values= "g").reset_index(drop=True).values
  b_img = img_df.pivot_table(index="x_l", columns="y_l",values= "b").reset_index(drop=True).values
  a_img = img_df.pivot_table(index="x_l", columns="y_l",values= "a").reset_index(drop=True).values
  df_img = np.stack([b_img, g_img, r_img, a_img], 2).astype(np.uint8)
  return df_img

def df2rgb(img_df):
  r_img = img_df.pivot_table(index="x_l", columns="y_l",values= "r").reset_index(drop=True).values
  g_img = img_df.pivot_table(index="x_l", columns="y_l",values= "g").reset_index(drop=True).values
  b_img = img_df.pivot_table(index="x_l", columns="y_l",values= "b").reset_index(drop=True).values
  df_img = np.stack([r_img, g_img, b_img], 2).astype(np.uint8)
  return df_img

def pil2cv(image):
  new_image = np.array(image, dtype=np.uint8)
  if new_image.ndim == 2:
      pass
  elif new_image.shape[2] == 3:
      new_image = new_image[:, :, ::-1]
  elif new_image.shape[2] == 4:
      new_image = new_image[:, :, [2, 1, 0, 3]]
  return new_image

def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image

def get_cls_update(ciede_df, threshold, cls2counts):
    set_list = [frozenset({cls, tgt}) for cls, tgt in ciede_df[ciede_df["ciede2000"] < threshold][["cls_no", "tgt_no"]].to_numpy()]
    merge_set = []
    while set_list:
        set_a = set_list.pop()
        merged = False
        for i, set_b in enumerate(merge_set):
            if set_a & set_b:
                merge_set[i] |= set_a
                merged = True
                break
        if not merged:
            merge_set.append(set_a)
    merge_dict = {}
    for merge in merge_set:
        max_cls = max(merge, key=cls2counts.get)
        for cls in merge:
            if cls != max_cls:
                merge_dict[cls] = max_cls
    return merge_dict

def calc_ciede(mean_list, cls_list):
  cls_no = []
  tgt_no = []
  ciede_list = []
  for i in range(len(mean_list)):
    img_1 = np.array(mean_list[i][:3])
    for j in range(len(mean_list)):
      if i == j:
        continue
      img_2 = np.array(mean_list[j][:3])
      ciede = color.deltaE_ciede2000(skimage_rgb2lab(img_1), skimage_rgb2lab(img_2))[0][0]
      ciede_list.append(ciede)
      cls_no.append(cls_list[i])
      tgt_no.append(cls_list[j])
  ciede_df = pd.DataFrame({"cls_no": cls_no, "tgt_no": tgt_no, "ciede2000": ciede_list})
  return ciede_df

def get_mask_cls(df, cls_no):
  mask = df.copy()
  mask.loc[df["label"] != cls_no, ["r","g","b"]] = 0
  mask.loc[df["label"] == cls_no, ["r","g","b"]] = 255
  mask = cv2.cvtColor(df2rgba(mask).astype(np.uint8), cv2.COLOR_RGBA2GRAY)
  return mask

def fill_mean_color(img_df, mask):
  df_img = df2rgba(img_df).astype(np.uint8)
  if len(df_img.shape) == 3:
      mask = np.repeat(mask[:, :, np.newaxis], df_img.shape[-1], axis=-1)
  masked_img = np.where(mask == 0, 0, df_img)
  mean = np.mean(masked_img[mask != 0].reshape(-1, df_img.shape[-1]), axis=0)

  img_df["r"] = mean[0]
  img_df["g"] = mean[1]
  img_df["b"] = mean[2]
  return img_df, mean

def get_blur_cls(img, cls, size):
  blur_img = cv2.blur(img, (size, size))
  blur_df = rgba2df(blur_img)
  blur_df["label"] = cls
  img_list = []
  mean_list = []
  cls_list = list(cls.unique())
  for cls_no in tqdm(cls_list):
    mask = get_mask_cls(blur_df, cls_no)
    img_df = blur_df.copy()
    img_df.loc[blur_df["label"] != cls_no, ["a"]] = 0 
    img_df, mean = fill_mean_color(img_df, mask)
    df_img = df2rgba(img_df).astype(np.uint8)
    img_list.append(df_img)
    mean_list.append(mean)
  return img_list, mean_list, cls_list

def get_cls_update(ciede_df, df, threshold):
    set_list = [frozenset({cls, tgt}) for cls, tgt in ciede_df[ciede_df["ciede2000"] < threshold][["cls_no", "tgt_no"]].to_numpy()]
    merge_set = []
    while set_list:
        set_a = set_list.pop()
        merged = False
        for i, set_b in enumerate(merge_set):
            if set_a & set_b:
                merge_set[i] |= set_a
                merged = True
                break
        if not merged:
            merge_set.append(set_a)
    merge_dict = {}
    for merge in merge_set:
        cls_counts = {cls: len(df[df["label"] == cls]) for cls in merge}
        max_cls = max(cls_counts, key=cls_counts.get)
        for cls in merge:
            merge_dict[cls] = max_cls
    return merge_dict

def get_color_dict(mean_list, cls_list):
  color_dict = {}
  for idx, mean in enumerate(mean_list):
    color_dict.update({cls_list[idx]:{"r":mean[0],"g":mean[1],"b":mean[2], }})
  return color_dict

def get_update_df(df, merge_dict, mean_list, cls_list):
  update_df = df.copy()
  update_df["label"] = update_df["label"].apply(lambda x: x if x not in merge_dict.keys() else merge_dict[x])
  color_dict = get_color_dict(mean_list, cls_list)
  update_df["r"] = update_df.apply(lambda x: color_dict[x["label"]]["r"], axis=1)
  update_df["g"] = update_df.apply(lambda x: color_dict[x["label"]]["g"], axis=1)
  update_df["b"] = update_df.apply(lambda x: color_dict[x["label"]]["b"], axis=1)    
  return update_df, color_dict

def split_img_df(df, show=False):
  img_list = []
  for cls_no in tqdm(list(df["label"].unique())):
    img_df = df.copy()
    img_df.loc[df["label"] != cls_no, ["a"]] = 0 
    df_img = df2rgba(img_df).astype(np.uint8)
    img_list.append(df_img)
  return img_list

def get_base(img, loops, cls_num, threshold, size, h_split, v_split, n_cluster, alpha, th_rate, bg_split=True):
  if bg_split == False:
    df = rgba2df(img)
    df_list = [df]
  else:
    df_list = get_foreground(img, h_split, v_split, n_cluster, alpha, th_rate)

  output_list = []

  for idx, df in enumerate(df_list):
    output_df = df.copy()
    cls = MiniBatchKMeans(n_clusters = cls_num)
    cls.fit(df[["r","g","b"]])
    df["label"] = cls.labels_ 
    df["label"] = df["label"].astype(str) + f"_{idx}"
    for i in range(loops):
      if i !=0:
        img = df2rgba(df).astype(np.uint8)
      blur_list, mean_list, cls_list = get_blur_cls(img, df["label"], size)
      ciede_df = calc_ciede(mean_list, cls_list)
      merge_dict = get_cls_update(ciede_df, df, threshold)
      update_df, color_dict = get_update_df(df, merge_dict, mean_list, cls_list)
      df = update_df
    output_df["label"] = df["label"]
    output_df["layer_no"] = idx 
    output_list.append(output_df)

  output_df = pd.concat(output_list).sort_index()

  mean_list = []
  cls_list = list(output_df["label"].unique())
  for cls_no in tqdm(cls_list):
    mask = get_mask_cls(output_df, cls_no)
    img_df = output_df.copy()
    img_df.loc[output_df["label"] != cls_no, ["a"]] = 0 
    img_df, mean = fill_mean_color(img_df, mask)
    mean_list.append(mean)

  color_dict = get_color_dict(mean_list, cls_list)
  output_df["r"] = output_df.apply(lambda x: color_dict[x["label"]]["r"], axis=1)
  output_df["g"] = output_df.apply(lambda x: color_dict[x["label"]]["g"], axis=1)
  output_df["b"] = output_df.apply(lambda x: color_dict[x["label"]]["b"], axis=1)  
  
  return output_df

def set_label(x, idx):
  if x["m_flg"] == True:
    return idx
  else:
    return x["label"]

def mode_fast(series):
    return series.mode().iloc[0]

def get_seg_base(input_image, masks, th):
  df = rgba2df(input_image)
  df["label"] = -1
  for idx, mask in tqdm(enumerate(masks)):
    if int(mask["area"] < th):
      continue
    mask_df = mask2df(mask["segmentation"])
    df = df.merge(mask_df, left_on=["x_l", "y_l"], right_on=["x_l_m", "y_l_m"], how="inner")
    df["label"] = np.where(df["m_flg"] == True, idx, df["label"])
    df.drop(columns=["x_l_m", "y_l_m", "m_flg"], inplace=True)

  df["r"] = df.groupby("label")["r"].transform(mode_fast)
  df["g"] = df.groupby("label")["g"].transform(mode_fast)
  df["b"] = df.groupby("label")["b"].transform(mode_fast)
  return df

def get_normal_layer(input_image, df):
  base_layer_list = split_img_df(df, show=False)

  org_df = rgba2df(input_image)
  hsv_df = hsv2df(cv2.cvtColor(df2rgba(df).astype(np.uint8), cv2.COLOR_RGB2HSV))
  hsv_org = hsv2df(cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV))

  hsv_org["bright_flg"] = hsv_df["v"] < hsv_org["v"]
  bright_df = org_df.copy()
  bright_df["bright_flg"] = hsv_org["bright_flg"]
  bright_df["a"] = np.where(bright_df["bright_flg"] == True, 255, 0)
  bright_df["label"] = df["label"]
  bright_layer_list = split_img_df(bright_df, show=False)

  hsv_org["shadow_flg"] = hsv_df["v"] >= hsv_org["v"]
  shadow_df = rgba2df(input_image)
  shadow_df["shadow_flg"] = hsv_org["shadow_flg"]
  shadow_df["a"] = np.where(shadow_df["shadow_flg"] == True, 255, 0)
  shadow_df["label"] = df["label"]
  shadow_layer_list = split_img_df(shadow_df, show=True)
    
  return base_layer_list, bright_layer_list, shadow_layer_list

def get_composite_layer(input_image, df):
  base_layer_list = split_img_df(df, show=False)

  org_df = rgba2df(input_image)

  org_df["r"] = org_df["r"].apply(lambda x:int(x))
  org_df["g"] = org_df["g"].apply(lambda x:int(x))
  org_df["b"] = org_df["b"].apply(lambda x:int(x))

  org_df["diff_r"] = df["r"] - org_df["r"]
  org_df["diff_g"] = df["g"] - org_df["g"]
  org_df["diff_b"] = df["b"] - org_df["b"]
  
  org_df["shadow_flg"] = org_df.apply(
    lambda x: True if x["diff_r"] >= 0 and x["diff_g"] >= 0 and x["diff_b"] >= 0 else False,
    axis=1
  )
  org_df["screen_flg"] = org_df.apply(
    lambda x: True if x["diff_r"] < 0 and x["diff_g"] < 0 and x["diff_b"] < 0 else False,
    axis=1
  )

  shadow_df = org_df.copy()
  shadow_df["a"] = org_df.apply(lambda x: 255 if x["shadow_flg"] == True else 0, axis=1)
  
  shadow_df["r"] = shadow_df["r"].apply(lambda x: x*255)
  shadow_df["g"] = shadow_df["g"].apply(lambda x: x*255)
  shadow_df["b"] = shadow_df["b"].apply(lambda x: x*255)

  shadow_df["r"] = (shadow_df["r"])/df["r"]
  shadow_df["g"] = (shadow_df["g"])/df["g"]
  shadow_df["b"] = (shadow_df["b"])/df["b"]
  
  shadow_df["label"] = df["label"]
  shadow_layer_list = split_img_df(shadow_df, show=True)

  screen_df = org_df.copy()

  screen_df["a"] = screen_df["screen_flg"].apply(lambda x: 255 if x == True else 0)

  screen_df["r"] = (screen_df["r"] - df["r"])/(1 - df["r"]/255) 
  screen_df["g"] = (screen_df["g"] - df["g"])/(1 - df["g"]/255) 
  screen_df["b"] = (screen_df["b"] - df["b"])/(1 - df["b"]/255) 

  screen_df["label"] = df["label"]
  screen_layer_list = split_img_df(screen_df, show=True)
  
  addition_df = org_df.copy()
  addition_df["a"] = addition_df.apply(lambda x: 255 if x["screen_flg"] == False and x["shadow_flg"] == False else 0, axis=1)

  addition_df["r"] = org_df["r"] - df["r"] 
  addition_df["g"] = org_df["g"] - df["g"] 
  addition_df["b"] = org_df["b"] - df["b"]  

  addition_df["r"] = addition_df["r"].apply(lambda x: 0 if x < 0 else x)
  addition_df["g"] = addition_df["g"].apply(lambda x: 0 if x < 0 else x)
  addition_df["b"] = addition_df["b"].apply(lambda x: 0 if x < 0 else x)

  addition_df["label"] = df["label"]

  addition_layer_list = split_img_df(addition_df, show=True)

  subtract_df = org_df.copy()
  subtract_df["a"] = subtract_df.apply(lambda x: 255 if x["screen_flg"] == False and x["shadow_flg"] == False else 0, axis=1)

  subtract_df["r"] = df["r"] - org_df["r"]   
  subtract_df["g"] = df["g"] - org_df["g"] 
  subtract_df["b"] = df["b"] - org_df["b"]

  subtract_df["r"] = subtract_df["r"].apply(lambda x: 0 if x < 0 else x)
  subtract_df["g"] = subtract_df["g"].apply(lambda x: 0 if x < 0 else x)
  subtract_df["b"] = subtract_df["b"].apply(lambda x: 0 if x < 0 else x)

  subtract_df["label"] = df["label"]

  subtract_layer_list = split_img_df(subtract_df, show=True)
  return base_layer_list, shadow_layer_list, screen_layer_list, addition_layer_list, subtract_layer_list

def add_psd(psd, img, name, mode):
  layer_1 = layers.ChannelImageData(image=img[:, :, 3], compression=1)
  layer0 = layers.ChannelImageData(image=img[:, :, 0], compression=1)
  layer1 = layers.ChannelImageData(image=img[:, :, 1], compression=1)
  layer2 = layers.ChannelImageData(image=img[:, :, 2], compression=1)

  new_layer = layers.LayerRecord(channels={-1: layer_1, 0: layer0, 1: layer1, 2: layer2},
                                  top=0, bottom=img.shape[0], left=0, right=img.shape[1],
                                  blend_mode=mode,
                                  name=name,
                                  opacity=255,
                                  )
  #gp = nested_layers.Group()
  #gp.layers = [new_layer]
  psd.layer_and_mask_info.layer_info.layer_records.append(new_layer)
  return psd

def load_masks(output_dir):
  pkl_path = os.path.join(output_dir, "tmp", "sorted_masks.pkl")
  with open(os.path.normpath(pkl_path), "rb") as f:
    masks = pickle.load(f)
  return masks

def save_psd(input_image, layers, names, modes, output_dir, layer_mode):
  input_cv = pil2cv(input_image)
  input_np = np.array(input_image)[..., :3].transpose(2, 0, 1)
  image_data = pytoshop.image_data.ImageData(channels=input_np)
  psd = pytoshop.core.PsdFile(num_channels=3, height=input_cv.shape[0], width=input_cv.shape[1], image_data=image_data)
  if layer_mode == "normal":
    for idx, output in enumerate(layers[0]):
      psd = add_psd(psd, layers[0][idx], names[0] + str(idx), modes[0])
      psd = add_psd(psd, layers[1][idx], names[1] + str(idx), modes[1])
      psd = add_psd(psd, layers[2][idx], names[2] + str(idx), modes[2])
  else:
    for idx, output in enumerate(layers[0]):
      psd = add_psd(psd, layers[0][idx], names[0] + str(idx), modes[0])
      psd = add_psd(psd, layers[1][idx], names[1] + str(idx), modes[1])
      psd = add_psd(psd, layers[2][idx], names[2] + str(idx), modes[2])
      psd = add_psd(psd, layers[3][idx], names[3] + str(idx), modes[3])
      psd = add_psd(psd, layers[4][idx], names[4] + str(idx), modes[4])

  name = f"image{next_index(output_dir)}"
  with open(os.path.normpath(f"{output_dir}/{name}.psd"), "wb") as fd2:
      psd.write(fd2)

  return f"{output_dir}/{name}.psd"

def divide_folder(psd_path, mode):
  assets_dir = os.path.join(dirname, "../dist/assets/images")
  with open(os.path.normpath(f"{assets_dir}/empty.psd"), "rb") as fd:
    psd_base = PSD.read(fd)
  with open(os.path.normpath(psd_path), "rb") as fd:
    psd_image = PSD.read(fd)

  if mode == "normal":
     add_num = 3
  else:
     add_num = 5

  base_records_list = list(psd_base.layer_and_mask_information.layer_info.layer_records)
  image_records_list = list(psd_image.layer_and_mask_information.layer_info.layer_records)

  merge_list = []
  for idx, record in enumerate(image_records_list):
      if idx % add_num == 0:
          merge_list.append(base_records_list[0])
      merge_list.append(record)
      if idx % add_num == (add_num - 1):
          merge_list.append(base_records_list[2])

  psd_image.layer_and_mask_information.layer_info.layer_records = psd_tools.psd.layer_and_mask.LayerRecords(merge_list)
  psd_image.layer_and_mask_information.layer_info.layer_count = len(psd_image.layer_and_mask_information.layer_info.layer_records)

  folder_channel = psd_base.layer_and_mask_information.layer_info.channel_image_data[0]
  image_channel = psd_image.layer_and_mask_information.layer_info.channel_image_data

  channel_list = []
  for idx, channel in enumerate(image_channel):
      if idx % add_num == 0:
          channel_list.append(folder_channel)
      channel_list.append(channel)
      if idx % add_num == (add_num - 1):
          channel_list.append(folder_channel)

  psd_image.layer_and_mask_information.layer_info.channel_image_data =  psd_tools.psd.layer_and_mask.ChannelImageData(channel_list)
  with open(os.path.normpath(psd_path), "wb") as fd:
      psd_image.write(fd)

  return psd_path

def get_mask_generator(pred_iou_thresh, stability_score_thresh, min_mask_region_area):
    global device
    sam_checkpoint = os.path.join(get_models_dir(), "misc/sam.pth")
    model_type = "default"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    if device == "mps": device = "cpu"
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )

    return mask_generator

def get_masks(image, mask_generator):
    masks = mask_generator.generate(image)
    return masks

def show_anns(image, masks, output_dir):
    if len(masks) == 0:
        return
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    with open(os.path.normpath(f"{output_dir}/tmp/sorted_masks.pkl"), "wb") as f:
        pickle.dump(sorted_masks, f)
    polygons = []
    color = []
    mask_list = []
    for mask in sorted_masks:
        m = mask["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        img = np.dstack((img*255, m*255*0.35))
        img = img.astype(np.uint8)

        mask_list.append(img)

    base_mask = image
    for mask in mask_list:
        base_mask = Image.alpha_composite(base_mask, Image.fromarray(mask))

    return base_mask

def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)

def segment_image(input_image, output_dir, pred_iou_thresh, stability_score_thresh, min_mask_region_area):
    mask_generator = get_mask_generator(pred_iou_thresh, stability_score_thresh,min_mask_region_area)
    masks = get_masks(pil2cv(input_image), mask_generator)
    input_image.putalpha(255)
    masked_image = show_anns(input_image, masks, output_dir)
    return masked_image

def divide_layer(divide_mode, input_image, output_dir, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg, area_th):
    if divide_mode == "segment_mode":
        return segment_divide(input_image, output_dir, layer_mode, area_th)
    elif divide_mode == "color_base_mode":
        return color_base_divide(input_image, output_dir, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg)

def segment_divide(input_image, output_dir, layer_mode, area_th):
    image = pil2cv(input_image)
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    masks = load_masks(output_dir)

    df = get_seg_base(input_image, masks, area_th)

    base_image = cv2pil(df2bgra(df))
    image = cv2pil(image)
    if layer_mode == "composite":
        base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = get_composite_layer(input_image, df)
        filename = save_psd(
            input_image,
            [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
            ["base", "screen", "multiply", "subtract", "addition"],
            [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
            output_dir,
            layer_mode

        )
        base_layer_list = [cv2pil(layer) for layer in base_layer_list]
        divide_folder(filename, layer_mode)
        return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
    elif layer_mode == "normal":
        base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(input_image, df)
        filename = save_psd(
            input_image,
            [base_layer_list, bright_layer_list, shadow_layer_list],
            ["base", "bright", "shadow"],
            [BlendMode.normal, BlendMode.normal, BlendMode.normal],
            output_dir,
            layer_mode
        )

        divide_folder(filename, layer_mode)
        return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
    else:
        return None

def color_base_divide(input_image, output_dir, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg):
    image = pil2cv(input_image)
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    df = get_base(input_image, loops, init_cluster, ciede_threshold, blur_size, h_split, v_split, n_cluster, alpha, th_rate, split_bg)

    base_image = cv2pil(df2bgra(df))
    image = cv2pil(image)
    if layer_mode == "composite":
        base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = get_composite_layer(input_image, df)
        filename = save_psd(
            input_image,
            [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
            ["base", "screen", "multiply", "subtract", "addition"],
            [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
            output_dir,
            layer_mode,
        )
        base_layer_list = [cv2pil(layer) for layer in base_layer_list]
        divide_folder(filename, layer_mode)
        return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
    elif layer_mode == "normal":
        base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(input_image, df)
        filename = save_psd(
            input_image,
            [base_layer_list, bright_layer_list, shadow_layer_list],
            ["base", "bright", "shadow"],
            [BlendMode.normal, BlendMode.normal, BlendMode.normal],
            output_dir,
            layer_mode,
        )
        divide_folder(filename, layer_mode)
        return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
    else:
        return None
    
def layer_divide(input_image, output_dir, divide_mode, loops, clusters, cluster_threshold, blur_size, layer_mode, area):
    if not divide_mode: divide_mode = "color_base_mode"
    if not loops: loops = 3
    if not clusters: clusters = 10
    if not cluster_threshold: cluster_threshold = 15
    if not blur_size: blur_size = 5
    if not layer_mode: layer_mode = "composite"
    if not area: area = 20000
    h_split = 256
    v_split = 256
    n_cluster = 500
    alpha = 100
    th_rate = 0.1
    split_bg = True

    output = ""
    if divide_mode == "color_base_mode":
        output_images, base_layers, bright_layers, shadow_layers, output_file = divide_layer(divide_mode, input_image, output_dir,
        loops, clusters, cluster_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg, area)
        output = output_file
    elif divide_mode == "segment_mode":
        pred_iou_thresh = 0.8
        stability_score_thresh = 0.8
        min_mask_region_area = 100
        os.makedirs(f"{output_dir}/tmp", exist_ok=True)
        segment_image(input_image, output_dir, pred_iou_thresh, stability_score_thresh, min_mask_region_area)
        output_images, base_layers, bright_layers, shadow_layers, output_file = divide_layer(divide_mode, input_image, output_dir,
        loops, clusters, cluster_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg, area)
        shutil.rmtree(f"{output_dir}/tmp")
        output = output_file

    return output
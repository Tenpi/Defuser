import argparse
import pathlib
import io
from PIL import Image
import numpy as np
import io
import time
import struct
import pywt
import os
from enum import Enum

class SubsampleOptions(Enum):
  FOUR_FOUR_TWO = 0
  FOUR_TWO_ZERO = 1

def subsample(
    yuv_img: np.ndarray,
    u=True,
    v=False,
    subsample_type: SubsampleOptions = SubsampleOptions.FOUR_TWO_ZERO,
  ):
  if subsample_type == SubsampleOptions.FOUR_FOUR_TWO:
    return ValueError("Not yet implemented")

  channels = []
  if u:
    channels.append(1)
  if v:
    channels.append(2)

  subsampled_image = yuv_img.copy()
  cols, rows, _ = yuv_img.shape
  for channel in channels:

    # Horizontal copy
    print("~~~~ Horizontal ~~~~~")
    print(subsampled_image[:, :, channel].shape)
    print(subsampled_image[:, 1:rows // 2 * 2:2, channel].shape)
    print(subsampled_image[:, ::2, channel].shape)
    last_source_row = rows // 2 * 2
    subsampled_image[:, 1::2, channel] = subsampled_image[:,
                                                          :last_source_row:2, channel]

    # if subsample_type == SubsampleOptions.FOUR_FOUR_TWO:
    #   continue

    # Vertical copy
    print("~~~~ Vertical ~~~~~")
    print(subsampled_image[1::2, :, channel].shape)
    print(subsampled_image[::2, :, channel].shape)
    last_source_col = cols // 2 * 2
    subsampled_image[1::2, :,
                     channel] = subsampled_image[:last_source_col:2, :, channel]

  return subsampled_image

def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
  m = np.array([
      [0.29900, -0.14713, 0.615],
      [0.58700, -0.28886, -0.51499],
      [0.11400, 0.436, -0.10001]
    ])

  # rgb = rgb.astype(float)
  yuv = np.dot(rgb, m)
  yuv[:, :, 1:] += 127.5
  yuv = np.clip(yuv, 0, 255)
  # return yuv
  return np.round(yuv).astype(int)


def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
  yuv = yuv.astype(float)
  m = np.array([
      [1.000, 1.000, 1.000],
      [0.000, -0.39465, 2.03211],
      [1.13983, -0.58060, 0.000],
    ])

  yuv[:, :, 1:] -= 127.5
  rgb = np.dot(yuv, m)
  rgb = np.clip(rgb, 0, 255)
  return np.round(rgb).astype(int)


class EmbedMaxDct(object):
  def __init__(self, watermarks=[], scales=[0, 36, 36], block=4):
    self._watermarks = watermarks
    self._wmLen = len(watermarks)
    self._scales = scales
    self._block = block

  def encode_rgb(self, rgb: np.ndarray) -> np.ndarray:
    yuv = rgb_to_yuv(rgb)
    yuv = subsample(yuv) 
    encoded = self._encode_yuv(yuv)
    return yuv_to_rgb(encoded)

  def _encode_yuv(self, yuv: np.ndarray) -> np.ndarray:
    rows, columns, _ = yuv.shape

    for channel in range(2):
      if self._scales[channel] <= 0:
        continue

      last_processed_row = rows // self._block * self._block
      last_processed_col = columns // self._block * self._block

      ca1, (h1, v1, d1) = pywt.dwt2(
          yuv[:last_processed_row, :last_processed_col, channel], 'haar')

      self.encode_frame(ca1, scale=self._scales[channel])

      yuv[:last_processed_row, :last_processed_col, channel, ] = pywt.idwt2(
          (ca1, (v1, h1, d1)), 'haar')

    return yuv

  def encode_frame(self, frame, scale):
    '''
    frame is a matrix (M, N)

    we get K (watermark bits size) blocks (self._block x self._block)

    For i-th block, we encode watermark[i] bit into it
    '''
    (row, col) = frame.shape
    num = 0

    for i in range(row // self._block):
      for j in range(col // self._block):
        block = frame[i * self._block: i * self._block + self._block,
                      j * self._block: j * self._block + self._block]
        wmBit = self._watermarks[(num % self._wmLen)]

        diffusedBlock = self.diffuse_dct_matrix(block, wmBit, scale)

        frame[i * self._block: i * self._block + self._block,
              j * self._block: j * self._block + self._block] = diffusedBlock

        num = num + 1

  def diffuse_dct_matrix(self, block, wmBit, scale):
    """
    To embed a 1, add 0.75
    To embed a 0, add 0.25
    """
    pos = np.argmax(abs(block.flatten()[1:])) + 1
    i, j = pos // self._block, pos % self._block

    val = block[i][j]

    if val >= 0.0:
      block[i][j] = (val // scale + 0.25 + (0.5 * wmBit)) * scale
    else:
      val = abs(val)
      block[i][j] = -1.0 * (val // scale + 0.25 + (0.5 * wmBit)) * scale
    return block


class DecodeMaxDct(object):
  def __init__(self, wm_length, scales=[0, 36, 36], block=4):
    self._wmLen = wm_length
    self._scales = scales
    self._block = block

  def decode_rgb(self, rgb: np.ndarray) -> np.ndarray:
    rows, columns, __name__ = rgb.shape

    yuv = rgb_to_yuv(rgb)

    scores = [[] for i in range(self._wmLen)]
    for channel in range(2):
      if self._scales[channel] <= 0:
        continue

      last_processed_row = rows // self._block * self._block
      last_processed_col = columns // self._block * self._block

      ca1, (_, _, _) = pywt.dwt2(
          yuv[:last_processed_row, :last_processed_col, channel], 'haar')

      scores = self.decode_frame(ca1, self._scales[channel], scores)

    avgScores = list(map(lambda l: np.array(l).mean(), scores))

    bits = (np.array(avgScores) * 255 > 127)
    return bits

  def decode_frame(self, frame, scale, scores):
    (row, col) = frame.shape
    num = 0

    for i in range(row // self._block):
      for j in range(col // self._block):
        block = frame[i * self._block: i * self._block + self._block,
                      j * self._block: j * self._block + self._block]

        wmBit = num % self._wmLen
        score = self.infer_dct_matrix(block, scale)

        scores[wmBit].append(score)
        num = num + 1

    return scores

  def infer_dct_matrix(self, block, scale):
    pos = np.argmax(abs(block.flatten()[1:])) + 1
    i, j = pos // self._block, pos % self._block

    val = block[i][j]
    if val < 0:
      val = abs(val)

    if (val % scale) > 0.5 * scale:
      return 1
    else:
      return 0


class WatermarkEncoder(object):
  def __init__(self, content=b''):
    seq = np.array([n for n in content], dtype=np.uint8)
    self._watermarks = list(np.unpackbits(seq))
    self._wmLen = len(self._watermarks)

  def get_length(self):
    return self._wmLen

  def max_dwt_encode(self, rgb: np.ndarray) -> np.ndarray:
    rows, columns, _ = rgb.shape

    if rows * columns < 256 * 256:
      raise RuntimeError(
          'image too small, should be larger than 256x256')

    embed = EmbedMaxDct(self._watermarks)
    return embed.encode_rgb(rgb)


class WatermarkDecoder(object):
  def __init__(self, wm_length=0):
    self._wmLen = wm_length

  def _reconstruct_bytes(self, bits):
    nums = np.packbits(bits)
    bstr = b''
    for i in range(self._wmLen // 8):
      bstr += struct.pack('>B', nums[i])
    return bstr

  def decode(self, rgb) -> bytes:
    rows, columns, _ = rgb.shape
    if rows * columns < 256 * 256:
      raise RuntimeError(
          'image too small, should be larger than 256x256')

    bits = []
    embed = DecodeMaxDct(wm_length=self._wmLen)
    bits = embed.decode_rgb(rgb)
    return self._reconstruct_bytes(bits)

def apply_watermark(img_buffer: bytes, jpeg_quality: int = 75,watermark: str = "SDV2"):
  if jpeg_quality < 0 or jpeg_quality > 100:
    raise ValueError("jpeg_quality must be between 0 and 100")

  # Convert image bytes to numpy array
  img = _bytes_to_nparray(img_buffer)

  # Encode watermark into image
  wm_encoder = WatermarkEncoder(watermark.encode('utf-8', 'replace'))

  encoded_img = wm_encoder.max_dwt_encode(img)
  # Convert numpy array to image bytes
  encoded_img = Image.fromarray(encoded_img.astype(np.uint8), 'RGB')

  encoded_img_bytes_png = io.BytesIO()
  encoded_img_bytes_jpg = io.BytesIO()
  encoded_img_bytes_webp = io.BytesIO()

  encoded_img.save(encoded_img_bytes_jpg, format="jpeg",
                   quality=jpeg_quality, subsampling=0)
  encoded_img.save(encoded_img_bytes_png, format="png")
  encoded_img.save(encoded_img_bytes_webp, format="webp")

  return encoded_img_bytes_jpg, encoded_img_bytes_png, encoded_img_bytes_webp


def decode_wm(encoded_img_buffer: io.BytesIO, wm_length=32) -> str:
  encoded_img_bytes = encoded_img_buffer.getvalue()

  # Convert image bytes to numpy array
  encoded_img = _bytes_to_nparray(encoded_img_bytes)

  # Decode watermark from image
  wm_decoder = WatermarkDecoder(wm_length=wm_length)
  watermark = wm_decoder.decode(encoded_img)
  decoded = watermark.decode('utf-8', 'replace')
  return decoded

def _bytes_to_nparray(bytes: bytes) -> np.array:
  start_time = time.time()
  # Convert image bytes to PIL image object
  img = Image.open(io.BytesIO(bytes))

  width, height = img.size
  # logging.info(
  #   f"image size: {width} x {height} = {(width * height):,} pixels")
  pixels = width * height

  if pixels > 4096 * 4096:
    raise ValueError("Image is too large. Max size is 4096 x 4096 pixels.")
  if pixels < 256 * 256:
    raise ValueError("Image is too small. Should be larger than 256x256 ")

  # preprocess_time = time.time()
  # logging.info(
  #   f"time to handle pre-processing: {preprocess_time - start_time}")

  # logging.info(f"image mode: {img.mode}")
  if img.mode != 'RGB':
    img = img.convert('RGB')

  # convert_time = time.time()
  # logging.info(f"time to convert to RGB:  {convert_time - preprocess_time}")

  # Convert PIL image object to numpy array
  return np.asarray(img)

def encode_watermark(input, output, watermark, quality = 100):
    watermark = watermark if watermark else "SDV2"
    quality = int(quality) if quality else 100
    ext = pathlib.Path(output).suffix
    img = io.BytesIO(open(os.path.normpath(input), "rb").read())
    watermark_bytes = apply_watermark(img.read(), quality, watermark)
    if ext == ".jpg" or ext == ".jpeg":
        image = Image.open(watermark_bytes[0])
        image.save(output, "jpeg")
    elif ext == ".png":
        image = Image.open(watermark_bytes[1])
        image.save(output, "png")
    else:
        image = Image.open(watermark_bytes[2])
        image.save(output, "webp")

def decode_watermark(input, output, length):
    length = int(length) if length else 4
    img = io.BytesIO(open(os.path.normpath(input), "rb").read())
    decoded = decode_wm(img, wm_length=length * 8)
    file = open(os.path.normpath(output), "w")
    file.write(decoded)
    file.close()
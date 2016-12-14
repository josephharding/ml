
import struct


def get_big_endian_int(s):
  return struct.unpack(">i", s)[0]

def get_big_endian_byte(s):
  return struct.unpack(">B", s)[0]

def write_image(name, pixels):
  # seems the pixels are in reverse order
  pixels = pixels[::-1]

  bf = open(name, 'wb')
  # create the BMP header 
  header = ''
  header += struct.pack('<BB', 66, 77) # B ascii code, M ascii code
  header += struct.pack('<L', 40 + 14 + (n_rows * n_cols * 3)) # file size
  header += struct.pack('<L', 0) # reserved
  header += struct.pack('<L', 54) # offset
  header += struct.pack('<L', 40) # head length
  header += struct.pack('<L', n_cols) # width
  header += struct.pack('<L', n_rows) # height
  header += struct.pack('<H', 1) # colorplanes
  header += struct.pack('<H', 24) # colordepth
  header += struct.pack('<L', 0) # compression
  header += struct.pack('<L', n_rows * n_cols * 24) # image size
  header += struct.pack('<L', 2835) # res_hor
  header += struct.pack('<L', 2835) # res_ver
  header += struct.pack('<L', 0) # palette
  header += struct.pack('<L', 0) # importantcolors

  # NOTE: not adding any padding b/c these images _happen_ to be evenly divisible by 4
  body = ''
  for r in range(n_rows):
    for c in range(n_cols):
      idx = n_cols - 1 - c + (r * n_cols) # walking the rows in reverse to correct source image
      body += struct.pack('<BBB', pixels[idx], pixels[idx], pixels[idx]) # adds 84 bytes total

  bf.write(header + body)
  bf.close()


if __name__ == '__main__':

  # open a binary file
  f = open('./t10k-images-idx3-ubyte', 'rb')
  try:
    magic_num = get_big_endian_int(f.read(4))
    n_images = get_big_endian_int(f.read(4))
    n_rows = get_big_endian_int(f.read(4))
    n_cols = get_big_endian_int(f.read(4))
    
    print("magic num was {m}, {n} images with {r} rows and {c} columns".format(m=magic_num, n=n_images, r=n_rows, c=n_cols)) 

    # write the first image into a bitmap file 
    for n in range(n_images): 
      pixels = [] 
      for i in range(n_rows * n_cols):    
        pixels.append(get_big_endian_byte(f.read(1)))
      
      write_image('images/image{n}.bmp'.format(n=n),  pixels)


  finally:    
    f.close()

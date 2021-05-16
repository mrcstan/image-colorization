import cv2
import numpy as np
import utils as utils

# Target image
#im_in = './input_images/cats_res.png'
im_in = './input_images/vintage1.jpg'

im_in = cv2.imread(im_in, 0)/255.0; # read as gray scale image
im_in = np.stack((im_in,) * 3, axis=-1)

# Existing marked image if available
#im_marked = './input_images/im_marked_gui.pvng'
#im_marked = cv2.cvtColor(cv2.imread(im_marked), cv2.COLOR_BGR2RGB)/255.0
im_marked = None

# Color source image. Use this to provide an image where the color is to be taken from
im_src_color = './input_images/cats_low_m.png'
im_src_color = cv2.cvtColor(cv2.imread(im_src_color), cv2.COLOR_BGR2RGB)/255.0
#im_src_color = None

utils.scribble_color(im_in, im_marked_in=im_marked, im_src_color=im_src_color,
                    hwsz=1, solver='spsolve', atol=1e-6, btol=1e-6, iter_lim=10000)

# Cannot run matplotlib plt after destroying the tkinter window or master

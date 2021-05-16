import cv2
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

import tkinter as tk
import tkinter.colorchooser as colorchooser
from matplotlib import pyplot as plt
import matplotlib as mplib
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
mplib.use("TkAgg"); # need this to avoid AttributeError: 'FigureCanvasTkAgg' object has no attribute 'manager'

def BGR2YUV(im):
    return cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_BGR2YUV)/255.0

def YUV2RGB(im):
    return cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_YUV2RGB)/255.0

def RGB2BGR_UINT8(im):
    #return cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return (im[:, :, [2, 1, 0]]*255).astype(np.uint8)

def RGB2BGR(im):
    return im[:, :, [2, 1, 0]]

'''
def BGR2YIQ(im):
    #im2 = np.zeros(im.shape, dtype=np.uint8)
    #im2[:, :, 0] = 0.299*im[:, :, 2] + 0.587*im[:, :, 1] + 0.114*im[:, :, 0]
    #im2[:, :, 1] = 0.596*im[:, :, 2] - 0.275*im[:, :, 1] - 0.321*im[:, :, 0]
    #im2[:, :, 2] = 0.212*im[:, :, 2] - 0.523*im[:, :, 1] + 0.311*im[:, :, 0]
    #return im2
    transform = np.array([[0.114020904255103, 0.587043074451121, 0.298936021293775],
                          [-0.321557107325010,  -0.274388635745789, 0.595945743070799],
                          [0.311413349996145,  -0.522910690302974, 0.211497340306828]])
    return np.dot(im, transform.T.copy())
'''
'''
# Inverse of RGB2YIQ. Produces negative rgb values
def YIQ2RGB(im):
    transform = np.array([[1.0, 0.956, 0.621],
                          [1.0, -0.272, -0.647],
                          [1.0, -1.106, 1.703]])
    return np.dot(im, transform.T.copy())
'''
'''
# From MATLAB ntsc2rgb
def YIQ2BGR(im):
    transform = np.array([[ 1.0, 0.0, 1.0],
                          [ 1.0, 0.0, 0.0],
                          [ 1.0, 0.956,  0.364650616559013]])
    return np.dot(im, transform.T.copy())
'''
'''
# From MATLAB ntsc2rgb
def YIQ2RGB(im):
    transform = np.array([[1.0, 0.956, 0.364650616559013],
                          [1.0, 0.0, 0.0],
                          [1.0, 0.0, 1]])
    return np.dot(im, transform.T.copy())
'''
'''
# Does not work properly due to rounding errors
def BGR2YUV(im):
    transform = np.array([[0.114, 0.587, 0.299],
                          [0.436,  -0.28886, -0.14713],
                          [-0.10001, -0.51499, 0.615]])
    return np.dot(im, transform.T.copy())
# Does not work properly due to rounding errors
def YUV2RGB(im):
    transform = np.array([[1, 0, 1.13983],
                          [1, -0.39465, -0.5806],
                          [1, 2.03211, 0]])
    return np.dot(im, transform.T.copy())

'''

'''

# marked by color_mask 
# hwsz: 2*hwsz+1 is the window size around a pixel
'''
def color_image(im_in, im_marked, hwsz=1, tol_variance = 2e-6,
                solver='spsolve', atol=1e-6, btol=1e-6, iter_lim=10000):
    '''
    # Color im_in based on the colors in im_marked
    :param im_in: BGR input image
    :param im_marked: BGR input image with color markers
    :param hwsz: distance within which a pixel is considered neighbor
    :param tol_variance:
    :return:
    '''
    np.seterr(divide='raise', invalid='raise')

    n_rows = im_in.shape[0]
    n_cols = im_in.shape[1]
    assert n_rows == im_marked.shape[0]
    assert n_cols == im_marked.shape[1]
    assert solver.lower() == 'spsolve' or solver.lower() == 'lsqr'
    # -------------------------------------------------------------------------------------------------
    # Prepare images
    # mask where pixels are colored. Must convert to signed integer to avoid error in substraction when
    # difference is negative
    color_mask = (np.sum(np.abs(im_in - im_marked), axis=2) > 0.01)
    plt.figure()
    plt.imshow(color_mask.astype(np.uint8), cmap='gray')
    #color_mask = (np.sum(np.abs(im_in.astype(np.int8) - im_marked.astype(np.int8)), axis=2) > 1)
    im_in2 = BGR2YUV(im_in)
    #print('im_in2 = ', im_in2.min(), im_in2.max())
    im_marked2 = BGR2YUV(im_marked)
    #print('im_marked2 = ', im_marked2.min(), im_marked2.max())
    im_LC = im_marked2.copy()
    im_LC[:, :, 0] = im_in2[:, :, 0]
    # -------------------------------------------------------------------------------------------------
    # initialize variables for creating the sparse matrix
    wsz = 2*hwsz+1; # window size in one-direction
    wsz2 = wsz ** 2
    n_px = n_rows*n_cols
    pix2var = np.reshape(np.arange(n_px), (n_rows, n_cols))
    n_ii = n_px*wsz2
    ii = np.zeros(n_ii, dtype=np.uint64)
    jj = np.zeros(n_ii, dtype=np.uint64)
    val = np.zeros(n_ii)
    ii_ind = -1; # for (I, J, val)
    eqn_ind = -1;

    im_out = im_LC.copy()

    for i in range(n_rows):
        for j in range(n_cols):
            eqn_ind += 1
            #print('(i,j', i, j)
            if not color_mask[i,j]:
                # Reinitialize variables for window
                px_w = np.zeros(wsz2); # for storing pixels in a window
                ind_w = -1
                for r in range(max(0, i-hwsz), min(i+hwsz+1, n_rows)):
                    for s in range(max(0, j-hwsz), min(j+hwsz+1, n_cols)):
                        if (r != i or s != j):
                            ii_ind += 1
                            ind_w += 1
                            ii[ii_ind] = eqn_ind
                            jj[ii_ind] = pix2var[r, s]
                            px_w[ind_w] = im_LC[r, s, 0]
                            #print('r, s, im_LC', r, s, im_LC[r, s, 0])
                            #print('i, j, r, s = ', i, j, r, s)
                ind_w += 1
                px_w[ind_w] = im_LC[i, j, 0]
                variance = np.var(px_w[:(ind_w+1)]); # variance of pixel value in the window
                px_w = px_w[:ind_w]

                csig = 0.6*variance
                mgv = np.min((px_w - im_LC[i, j, 0])** 2);
                if (csig < (-mgv / np.log(0.01))):
                    csig = -mgv / np.log(0.01)
                if (csig < 0.000002):
                    csig = 0.000002
                '''
                if variance < tol_variance:
                    px_w[:] = 1./ind_w
                else:
                    px_w = np.exp(-(px_w - im_LC[i, j  , 0])**2/variance)
                    sum_px_w = np.sum(px_w)
                    px_w = px_w/sum_px_w
                '''
                px_w = np.exp(-(px_w - im_LC[i, j, 0]) ** 2 / csig)
                sum_px_w = np.sum(px_w)
                px_w = px_w / sum_px_w
                ii_ind1 = ii_ind + 1
                val[(ii_ind1 - ind_w):ii_ind1] = -px_w

            # pixel should maintain original intensity as much as possible
            ii_ind += 1
            ii[ii_ind] = eqn_ind
            jj[ii_ind] = pix2var[i, j]
            val[ii_ind] = 1

    # Construct sparse matrix
    ii_ind1 = ii_ind + 1
    ii = ii[:ii_ind1]
    jj = jj[:ii_ind1]
    val = val[:ii_ind1]

    
    A = sparse.coo_matrix((val, (ii, jj)), shape=(n_px, n_px)).tocsr()
    b = np.zeros(n_px)
    mask_vector = color_mask.ravel()

    print('Solving equations')
    if solver.lower() == 'lsqr':
        lsqr = linalg.lsqr
    elif solver.lower() == 'spsolve':
        solve = linalg.spsolve

    for col in range(1, im_LC.shape[2]):
        b[mask_vector] = im_LC[:, :, col].ravel()[mask_vector]
        if solver.lower() == 'lsqr':
            soln, istop, itn, r1norm = lsqr(A, b, atol=atol, btol=btol, iter_lim=iter_lim, show=False)[:4]
            print('lsqr stopping conditions:', istop, itn, r1norm)
        elif solver.lower() == 'spsolve':
            soln = solve(A, b)
        im_out[:, :, col] = soln.reshape(n_rows, n_cols)
        #print('soln min, max = ',soln.min(), soln.max())
        #print('1st soln, last soln = ', soln[0], soln[-1])

    im_out = YUV2RGB(im_out)
    return im_out



def scribble_color(im, im_marked_in=None, im_src_color=None,
                   hwsz=1, solver='spsolve', atol=1e-6, btol=1e-6, iter_lim=10000):
    '''
    :param im: RGB image represented by floating point numbers from 0 to 1
    param im_marked_in: RGB image represented by floating point numbers from 0 to 1
    :return:
    '''

    #assert len(im.shape) == 2
    scribble_color.xs = []; # line segment end points
    scribble_color.ys = []; # line segment end points
    scribble_color.lnwidth = 5; # line width in terms of pixel scale
    #scribble_color.im_marked = np.repeat((im[:, :, None]*255).astype(np.uint8), 3, axis=2)
    if im_marked_in is not None:
        scribble_color.im_marked = (im_marked_in*255).astype(np.uint8)
    else:
        scribble_color.im_marked = (im*255).astype(np.uint8)

    scribble_color.im_out = np.ones(im.shape)
    scribble_color.im_marked_prev = scribble_color.im_marked.copy()

    if im_src_color is None:
        scribble_color.im_src_color = (np.ones(im.shape)*255).astype(np.uint8)
    else:
        scribble_color.im_src_color = (im_src_color*255).astype(np.uint8)

    fig_in = plt.figure('Input')
    plt.imshow(scribble_color.im_marked_prev)
    plt.title('Input image. Click image to create line segments')
    #plt.subplots_adjust(bottom=0.5)
    fig_in.tight_layout()
    def create_xlab(lnwidth):
        xlabstr =  'c or click "color" button: change current color,\n' \
                + 'v: pick color from right source image, b: back to input image\n' \
                + 'z: del current line, n: fix color for current line and start new line\n'\
                + '-(=): decrease(increase) line width, current width = %d\n' % (
                  lnwidth) + 'o: get colored image, s: save, r: reset, q: quit'
        return xlabstr
    plt.xlabel(create_xlab(scribble_color.lnwidth))

    fig_other = plt.figure('Other')
    plt.title('Source color image')
    plt.imshow(scribble_color.im_src_color)
    #plt.subplots_adjust(bottom=0.5)
    scribble_color.cur_fig = fig_in
    fig_other.tight_layout()
    master = tk.Tk()
    master.attributes('-fullscreen', True)
    frame = tk.Frame(master)
    frame.pack(side = tk.TOP)
    frame_other = tk.Frame(master)
    frame_other.pack(side = tk.TOP)

    canvas_other = FigureCanvasTkAgg(fig_other, master=master)  # A tk.DrawingArea.
    canvas_other.draw()
    toolbar_other = NavigationToolbar2Tk(canvas_other, frame_other)
    toolbar_other.update()
    canvas_other.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
    #canvas_other.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
    #canvas_other._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    canvas = FigureCanvasTkAgg(fig_in, master=master)  # A tk.DrawingArea.
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    #canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    scribble_color.linecolor = np.array([1.0, 0.0, 0.0])
    def on_mouse_press(event):
        x = event.xdata
        y = event.ydata
        if not x or not y:
            return
        scribble_color.xs.append(x)
        scribble_color.ys.append(y)
        plt.figure('Input')
        '''
        if len(scribble_color.xs) > 1:
            line, = plt.plot(scribble_color.xs, scribble_color.ys, 'r-')
        else:
            line, = plt.plot(scribble_color.xs, scribble_color.ys, 'r.')
        scribble_color.lines.append(line)
        '''

        draw_line(scribble_color.im_marked, scribble_color.xs, scribble_color.ys, scribble_color.linecolor, scribble_color.lnwidth)
        plt.imshow(scribble_color.im_marked)
        canvas.draw()
        #canvas.get_tk_widget().update_idletasks()

    def on_mouse_press_other(event):
        x = event.xdata
        y = event.ydata
        if not x or not y:
            return
        x = np.round(x).astype(np.uint64)
        y = np.round(y).astype(np.uint64)
        scribble_color.linecolor = im_src_color[y, x, :]
        refresh_color_box()

    #def on_close(event):
    #    _quit()

    def draw_line(im, xs, ys, linecolor, linewdith):
        linecolor = (linecolor * 255).astype(np.uint8)
        xs = np.round(xs)
        ys = np.round(ys)
        linecolor = (int(linecolor[0]), int(linecolor[1]), int(linecolor[2]))
        # duplicate point if only 1 point is given
        if len(xs) == 1:
            xs = [xs[0], xs[0]]
            ys = [ys[0], ys[0]]
        for i in range(len(xs) - 1):
            pt1 = (int(xs[i]), int(ys[i]))
            pt2 = (int(xs[i+1]), int(ys[i+1]))
            cv2.line(im, pt1, pt2, linecolor, thickness=linewdith, lineType=8)

    def call_askcolor():
        color = colorchooser.askcolor(title="select color")
        # color[0] = three channel representation. color[1] = Hex representation
        scribble_color.linecolor = np.floor(np.array(color[0]))/255.0
        scribble_color.linecolor[scribble_color.linecolor > 1] = 1.0
        refresh_color_box()

    def refresh_color_box():
        #hexcolor = '#%02x%02x%02x' % scribble_color.linecolor
        hexcolor = mplib.colors.to_hex(scribble_color.linecolor)
        canvas_color.create_rectangle(50, 0, 100, 50, fill=hexcolor)
        canvas_color.pack(side=tk.BOTTOM)

    def on_key_press(event):
        print("You pressed {}".format(event.key))
        if event.key == 'z':
            # undo drawn lines
            scribble_color.im_marked = scribble_color.im_marked_prev.copy()
            plt.figure('Input')
            plt.imshow(scribble_color.im_marked)
            scribble_color.xs = []
            scribble_color.ys = []
            canvas.draw()
        elif event.key == 'n':
            # undo drawn lines
            plt.figure('Input')
            draw_line(scribble_color.im_marked, scribble_color.xs, scribble_color.ys, scribble_color.linecolor, scribble_color.lnwidth)
            scribble_color.im_marked_prev = scribble_color.im_marked.copy()
            plt.imshow(scribble_color.im_marked_prev)
            scribble_color.xs = []
            scribble_color.ys = []
            canvas.draw()
        elif event.key == 'c':
            print('Call ask color')
            call_askcolor()
        elif event.key == 'v':
            print('Show source color image if available')
            scribble_color.cur_fig = fig_other
            plt.title('Source color image')
            plt.imshow(scribble_color.im_src_color)
            canvas_other.draw()
            fig_in.canvas.mpl_disconnect(scribble_color.ci_mouse_press)
            scribble_color.co_mouse_press = fig_other.canvas.mpl_connect('button_press_event', on_mouse_press_other)
        elif event.key == 'b':
            if (scribble_color.co_mouse_press is not None):
                fig_other.canvas.mpl_disconnect(scribble_color.co_mouse_press)
            fig_in.canvas.mpl_connect('button_press_event', on_mouse_press)
            plt.figure('Input')
            plt.title('Input image. Click image to create line segments')
            plt.xlabel(create_xlab(scribble_color.lnwidth))
            plt.imshow(scribble_color.im_marked)
            canvas.draw()
        elif event.key == '-':
            scribble_color.lnwidth -= 1
            scribble_color.lnwidth = max(scribble_color.lnwidth, 1)
            plt.figure('Input')
            plt.xlabel(create_xlab(scribble_color.lnwidth))
            canvas.draw()
        elif event.key == '=':
            scribble_color.lnwidth += 1
            plt.figure('Input')
            plt.xlabel(create_xlab(scribble_color.lnwidth))
            canvas.draw()
        elif event.key == 'o':
            print('Coloring image ...')
            scribble_color.im_out = color_image(RGB2BGR(im), RGB2BGR(scribble_color.im_marked/255.0), hwsz=hwsz,
                                       solver=solver, atol=atol, btol=btol, iter_lim=iter_lim)
            print('im_out min, max =', scribble_color.im_out.min(), scribble_color.im_out.max())
            plt.figure('Other')
            plt.imshow(scribble_color.im_out)
            plt.title('Output image')
            canvas_other.draw()
        elif event.key == 's':
            print('Saving images ...')
            cv2.imwrite('./output_images/im_marked.png', cv2.cvtColor(scribble_color.im_marked, cv2.COLOR_RGB2BGR))
            cv2.imwrite('./output_images/im_out.png', RGB2BGR_UINT8(scribble_color.im_out))
        elif event.key == 'r':
            scribble_color.xs = []
            scribble_color.ys = []
            scribble_color.im_marked = (im * 255).astype(np.uint8)
            scribble_color.im_marked_prev = scribble_color.im_marked.copy()
            plt.figure('Input')
            plt.imshow(scribble_color.im_marked)
            canvas.draw()
        elif event.key == 'q':
            _quit()

        key_press_handler(event, canvas, toolbar)

    def _quit():
        # canvas_other.get_tk_widget().destroy()
        #  canvas.get_tk_widget().destroy()
        master.quit()  # does not stop mainloop
        master.destroy()  # ends mainloop


        # Fatal Python Error: PyEval_RestoreThread: NULL tstate


    # Create an hard reference to the callback not to be cleared by the garbage
    # collector
    scribble_color.ci_mouse_press = fig_in.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig_in.canvas.mpl_connect('key_press_event', on_key_press)
    scribble_color.co_mouse_press = None

    #fig_in.canvas.mpl_connect('close_event', on_close)

    #cprint(color)

    hexcolor =  mplib.colors.to_hex(scribble_color.linecolor)
    canvas_color = tk.Canvas(master, width=200, height=100)
    canvas_color.create_rectangle(50, 0, 100, 50, fill=hexcolor)
    #canvas_overall.place(relx=0.5, rely=0.5, anchor=tk.S)
    canvas_color.pack(side=tk.BOTTOM)
    button = tk.Button(master, text="Color", command= call_askcolor)
    button.pack(side=tk.BOTTOM)

    master.mainloop()
    return

def scribble_color2(im, im_marked_in=None, im_src_color=None,
                   hwsz=1, solver='spsolve', atol=1e-6, btol=1e-6, iter_lim=10000):
    '''
    :param im: RGB image represented by floating point numbers from 0 to 1
    param im_marked_in: RGB image represented by floating point numbers from 0 to 1
    :return:
    '''
    fig_in = plt.figure('Input')
    plt.title('Input image. Click image to create line segments')

    fig_other = plt.figure('Other')
    plt.title('Source color image')

    master = tk.Tk()
    master.attributes('-fullscreen', True)

    canvas_other = FigureCanvasTkAgg(fig_other, master=master)  # A tk.DrawingArea.
    canvas_other.draw()

    canvas = FigureCanvasTkAgg(fig_in, master=master)  # A tk.DrawingArea.
    canvas.draw()

    def _quit():
     #   canvas_other.get_tk_widget().destroy()
      #  canvas.get_tk_widget().destroy()
       master.quit()  # does not stop mainloop
       master.destroy()  # ends mainloop

    button = tk.Button(master, text="Quit", command= _quit)
    button.pack(side=tk.BOTTOM)

    master.mainloop()
    return im, im
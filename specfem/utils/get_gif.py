import os
import argparse
from PIL import Image

def get_gif_from_imgs(outfiles_path):
    """
    Fetches all the images in OUTPUT FILES and creates a .GIF out of them
    @arg outfiles_path: path to the OUTPUT_FILES folder
    @out Returns .GIF file saved in the same directory
    """

    # Fetch .jpg images
    list_dir = os.listdir(outfiles_path)
    for f in list_dir[:]:  # NB: list_dir[:] makes a copy
        if not(f.endswith('.jpg')):
            list_dir.remove(f)
    
    # 
    img, *imgs = [Image.open(os.path.join(outfiles_path, f)) for f in sorted(list_dir)]
    img.save(fp=os.path.join(outfiles_path, 'forward_gif.gif'), format='GIF', append_images=imgs, 
             save_all=True, duration=200, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates .GIF of the simulation')
    parser.add_argument('outfiles_path', type=str,
                        default='./OUTPUT_FILES/', 
                        nargs='?', const=1)
    args = parser.parse_args()

    get_gif_from_imgs(args.outfiles_path)
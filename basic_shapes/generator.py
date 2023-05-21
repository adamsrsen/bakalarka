import argparse
import sys
import os
from random import randint
from PIL import Image, ImageDraw

def pick_random(arr):
    return arr[randint(0, len(arr) - 1)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=100)
    parser.add_argument("-x", "--width", type=int, default=32)
    parser.add_argument("-y", "--height", type=int, default=32)
    parser.add_argument("-d", "--directory", type=str, default="./input")
    parser.add_argument("-t", "--type", type=lambda x: x.split(","), default="circle,square")
    parser.add_argument("-v", "--variance", type=int, default=0)
    parser.add_argument("-r", "--radius", type=int, default=12)
    parser.add_argument("-c", "--color", type=lambda x: x.split(","), default="white")
    parser.add_argument("-b", "--background", type=lambda x: x.split(","), default="black")
    parser.add_argument("-f", "--color-format", default="L")
    sys.argv.pop(0)
    args = parser.parse_args(sys.argv)

    for shape in args.type:
        path = f"{args.directory}/{shape}"
        if not os.path.exists(path):
            os.mkdir(path)

        for n in range(args.size):
            image = Image.new(args.color_format, (args.width, args.height), pick_random(args.background))
            draw = ImageDraw.Draw(image)
            r = args.radius + randint(-args.variance, args.variance)
            x,y = randint(0, args.width - r), randint(0, args.height - r),
            if shape == "circle":
                draw.ellipse((x,y,x+r,y+r),fill=pick_random(args.color))
            elif shape == "square":
                draw.rectangle((x,y,x+r,y+r),fill=pick_random(args.color))
            image.save(f"{path}/{n}.png")


if __name__ == "__main__":
    main()
"""Insert a styled skill-chain chapter before the original playground entrance.

The original 159.33 s October overview is required as --base. Its video packets
are copied unchanged: only the inserted chapter is encoded. Pillow and either
ffmpeg or imageio-ffmpeg are required. The default cut is an existing keyframe
and blank chapter boundary at 134.4 seconds.
"""
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base', type=Path, required=True)
parser.add_argument('--chain', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--cut', type=float, default=134.4)
args = parser.parse_args()
ffmpeg = shutil.which('ffmpeg')
if not ffmpeg:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

def run(*arguments):
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                    *map(str, arguments)], check=True)

font_file = Path(__file__).parent / 'fonts/Inter.ttf'
def font(size, weight=400):
    result = ImageFont.truetype(str(font_file), size)
    axes = result.get_variation_axes()
    result.set_variation_by_axes([14 if axis['name'] == b'Optical size' else weight
                                 for axis in axes])
    return result

chain_duration = 1340 / 60
lead = 0.5
insert_duration = chain_duration + 1.0
phases = [('open the door', 0, 7), ('walk', 7, 10),
          ('pick up the box', 10, 14), ('carry the box', 14, 20),
          ('put the box down on the floor', 20, chain_duration)]
with tempfile.TemporaryDirectory(prefix='fiber-chain-edit-') as directory:
    tmp = Path(directory)
    # Match the original montage's 208 px margins, Inter type, subdued colors,
    # and the #f3f5f9 background at the playground chapter boundary.
    decoration = Image.new('RGBA', (1920, 1080))
    shadow = Image.new('RGBA', decoration.size)
    ImageDraw.Draw(shadow).rounded_rectangle((208, 142, 1712, 988), radius=12,
                                             fill=(15, 23, 42, 22))
    decoration.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(10)))
    d = ImageDraw.Draw(decoration)
    d.rounded_rectangle((207, 135, 1713, 983), radius=13, fill='#dce1e9')
    d.text((208, 42), 'Chaining skills in simulation', font=font(42, 650), fill='#101828')
    d.text((1712, 62), 'One planner · one continuous rollout', font=font(20),
           fill='#667085', anchor='ra')
    decoration.save(tmp / 'decoration.png')
    mask = Image.new('L', (1504, 846))
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 1503, 845), radius=12, fill=255)
    mask.save(tmp / 'mask.png')
    inputs = ['-i', str(args.chain), '-loop', '1', '-framerate', '30', '-i', str(tmp/'decoration.png'),
              '-loop', '1', '-framerate', '30', '-i', str(tmp/'mask.png')]
    for i, (label, _, _) in enumerate(phases):
        image = Image.new('RGBA', (1920, 1080))
        draw = ImageDraw.Draw(image)
        draw.text((208, 1000), 'Language command', font=font(17), fill='#667085')
        draw.text((208, 1026), '“' + label + '”', font=font(27, 550), fill='#101828')
        image.save(tmp / f'command-{i}.png')
        inputs += ['-loop', '1', '-framerate', '30', '-i', str(tmp/f'command-{i}.png')]
    Image.new('RGB', (1920, 1080), '#f3f5f9').save(tmp/'background.png')
    inputs += ['-loop', '1', '-framerate', '30', '-i', str(tmp/'background.png')]
    filters = [f'[8:v]format=rgba,trim=duration={insert_duration}[bg]',
               '[bg][1:v]overlay=format=rgb:shortest=1[decorated]',
               f'[0:v]fps=30,scale=1504:846:flags=lanczos,setsar=1,format=rgba,'
               f'tpad=start_mode=clone:start_duration={lead}:stop_mode=clone:stop_duration=0.5[scene]',
               '[scene][2:v]alphamerge[rounded]',
               '[decorated][rounded]overlay=208:136:format=rgb:shortest=1[c0]']
    for i, (_, start, end) in enumerate(phases):
        begin = 0 if i == 0 else start + lead
        finish = insert_duration if i == 4 else end + lead
        filters += [f'[{i+3}:v]format=rgba,fade=t=in:st={begin}:d=0.18:alpha=1,'
                    f'fade=t=out:st={finish-0.18}:d=0.18:alpha=1[label{i}]',
                    f'[c{i}][label{i}]overlay=format=rgb:shortest=1[c{i+1}]']
    filters += [f'[c5]fade=t=in:st=0:d=0.45:color=0xf3f5f9,'
                f'fade=t=out:st={insert_duration-0.45}:d=0.45:color=0xf3f5f9,'
                'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[v]']
    run(*inputs, '-filter_complex', ';'.join(filters), '-map', '[v]', '-an',
        '-t', insert_duration, '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level:v', '4.0',
        '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
        '-video_track_timescale', '15360', '-movflags', '+faststart', tmp/'chapter.mp4')
    # A keyframe split retains all of the original animation and image quality.
    run('-i', args.base, '-t', args.cut, '-map', '0:v', '-c:v', 'copy', '-an', tmp/'head.mp4')
    run('-ss', args.cut, '-i', args.base, '-map', '0:v', '-c:v', 'copy', '-an', tmp/'tail.mp4')
    packet_text = subprocess.check_output([ffmpeg, '-v', 'error', '-i', str(tmp/'head.mp4'),
                                           '-map', '0:v', '-c:v', 'copy', '-f', 'framecrc', '-']).decode()
    denominator = int(re.search(r'#tb 0: 1/(\d+)', packet_text).group(1))
    packets = [line.split(',') for line in packet_text.splitlines() if not line.startswith('#')]
    head_duration = max(int(row[2]) + int(row[3]) for row in packets) / denominator
    (tmp/'concat.txt').write_text(f"file 'head.mp4'\nduration {head_duration:.9f}\nfile 'chapter.mp4'\nduration {insert_duration:.9f}\nfile 'tail.mp4'\n")
    run('-f', 'concat', '-safe', '0', '-i', tmp/'concat.txt', '-c:v', 'copy', '-an', tmp/'video.mp4')
    # Keep original audio synchronized on either side of the new silent chapter.
    audio = (f'[1:a]asplit=2[a0][a1];'
             f'[a0]atrim=end={args.cut},asetpts=PTS-STARTPTS,'
             f'afade=t=out:st={args.cut-0.3}:d=0.3[head];'
             f'anullsrc=r=48000:cl=stereo,atrim=duration={insert_duration+head_duration-args.cut}[gap];'
             f'[a1]atrim=start={args.cut},asetpts=PTS-STARTPTS,afade=t=in:d=0.3[tail];'
             '[head][gap][tail]concat=n=3:v=0:a=1[a]')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run('-i', tmp/'video.mp4', '-i', args.base, '-filter_complex', audio,
        '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
        '-movflags', '+faststart', '-map_metadata', '-1', args.output)
print(args.output)

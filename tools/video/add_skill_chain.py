"""Insert the complete local skill-chain rollout into the published overview.

Requires ffmpeg (or imageio-ffmpeg) and Pillow. Retains the original soundtrack
on either side, fading to the silent simulation insert without repeating audio.
The base must be the original 159.33 s overview, saved outside the site checkout.
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base', type=Path, required=True)
parser.add_argument('--chain', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--cut', type=float, default=73.0)
parser.add_argument('--resume', type=float, default=74.0,
                    help='Skip the existing montage transition when resuming.')
args = parser.parse_args()
ffmpeg = shutil.which('ffmpeg')
if not ffmpeg:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
font_path = '/System/Library/Fonts/Helvetica.ttc'
if not Path(font_path).exists():
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = lambda size: ImageFont.truetype(font_path, size)
# Source is the site's existing, continuous 1340-frame, 60 Hz chain render.
chain_duration = 1340 / 60
base_duration = 4780 / 30
fade = 0.4
phases = [('Open door', 0, 7.0), ('Go through', 7.0, 10.0),
          ('Pick up box', 10.0, 14.0), ('Carry back', 14.0, 20.0),
          ('Place down', 20.0, chain_duration)]
with tempfile.TemporaryDirectory(prefix='fiber-chain-edit-') as tmp:
    tmp = Path(tmp)
    inputs = ['-i', str(args.base), '-i', str(args.chain)]
    for active, _ in enumerate(phases):
        overlay = Image.new('RGBA', (1920, 1080), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, 1920, 118), fill='white')
        draw.text((72, 34), 'Skill chaining in simulation', font=font(42), fill='#111827')
        draw.text((1470, 49), 'One continuous rollout', font=font(23), fill='#667085')
        draw.rectangle((0, 960, 1920, 1080), fill='white')
        for i, (label, _, _) in enumerate(phases):
            x = 72 + i * 356
            color = '#2455d6' if i == active else '#758094'
            if i == active:
                draw.rounded_rectangle((x, 983, x+328, 1045), radius=16, fill='#edf2ff')
            draw.text((x+18, 998), label, font=font(26), fill=color)
            if i < 4:
                draw.text((x+335, 1000), '›', font=font(24), fill='#b0b8c7')
        image = tmp / f'phase-{active}.png'
        overlay.save(image)
        inputs += ['-loop', '1', '-i', str(image)]
    before = args.cut
    filters = [f'[0:v]split=2[head0][tail0]',
               f'[head0]trim=end={before},setpts=PTS-STARTPTS,fps=30,settb=1/30,format=yuv420p[head]',
               f'[tail0]trim=start={args.resume},setpts=PTS-STARTPTS,fps=30,settb=1/30,format=yuv420p[tail]',
               f'[1:v]fps=30,trim=duration={chain_duration},setpts=PTS-STARTPTS,format=yuv420p[c0]']
    for i, (_, start, end) in enumerate(phases):
        enable = f'gte(t,{start})' if i == len(phases)-1 else f'gte(t,{start})*lt(t,{end})'
        filters += [f'[c{i}][{i+2}:v]overlay=0:0:enable=\'{enable}\':shortest=1[c{i+1}]']
    filters += [f'[c5]fps=30,settb=1/30[chain];[head][chain]xfade=transition=fade:duration={fade}:offset={before-fade}[joined]',
                f'[joined]fps=30,settb=1/30[joined30];[joined30][tail]xfade=transition=fade:duration={fade}:offset={before+chain_duration-2*fade},format=yuv420p[v]',
                f'[0:a]asplit=2[a0][a1]',
                f'[a0]atrim=end={before-fade},asetpts=PTS-STARTPTS,afade=t=out:st={before-0.9}:d=0.5[ah]',
                f'anullsrc=r=48000:cl=stereo,atrim=duration={chain_duration-fade},asetpts=PTS-STARTPTS[silence]',
                f'[a1]atrim=start={args.resume},asetpts=PTS-STARTPTS,afade=t=in:d=0.5[at]',
                '[ah][silence][at]concat=n=3:v=0:a=1[a]']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, '-hide_banner', '-loglevel', 'warning', '-y', *inputs,
               '-filter_complex', ';'.join(filters), '-map', '[v]', '-map', '[a]',
               '-c:v', 'libx264', '-preset', 'medium', '-crf', '27', '-pix_fmt', 'yuv420p',
               '-r', '30', '-c:a', 'aac', '-b:a', '96k', '-map_metadata', '-1',
               '-movflags', '+faststart', '-t', str(base_duration+chain_duration-2*fade-(args.resume-args.cut)),
               str(args.output)]
    subprocess.run(command, check=True)
print(args.output)

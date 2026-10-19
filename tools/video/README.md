# Overview video edit

The current full-generation source was on the unavailable cluster at
`/home/ubuntu/Assets/fiber_demo/`; it is not in the local website checkout.
The older scripts under local `Assets/icra_demo/` generate the September video.

`add_skill_chain.py` edits the saved, untouched October overview. It inserts the
complete simulation chain at the existing 134.4 s blank/keyframe boundary,
immediately before the original playground entrance. The original video packets
are copied unchanged, retaining the original animation and encoding quality.
Only the new chapter is rendered. It matches the original Inter typography,
208 px side margins, muted colors and framed composition, with fades and one
command caption. The original soundtrack is retained on either side; the new
chapter has no narration or music.

From `/Users/yutanglin/Desktop/FAR`, using Python with Pillow and imageio-ffmpeg:

```sh
python fiberaa.github.io/tools/video/add_skill_chain.py \
  --base Assets/demo_edits/fiber_demo_before_chain.mp4 \
  --chain fiberaa.github.io/static/videos/sim/chain.mp4 \
  --output Assets/demo_edits/fiber_demo_chain_before_playground_v2.mp4
```

Keep the untouched base outside the website repository. Do not use an already
edited overview as the base. The default split depends on this original video's
keyframe boundary. Inter is bundled under the SIL Open Font License in `fonts/`.

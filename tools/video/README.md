# Overview video edit

The original October overview generation source was on the unavailable cluster.
`add_skill_chain.py` reproducibly inserts the site's existing complete simulation
chain into the locally saved overview, with phase labels and 0.4-second fades.
It skips the original montage transition at 73–74 seconds. Original audio stays
aligned on both sides; the simulation insert is silent, with audio fades.

From `/Users/yutanglin/Desktop/FAR`, using Python with Pillow and imageio-ffmpeg:

```sh
python fiberaa.github.io/tools/video/add_skill_chain.py \
  --base Assets/demo_edits/fiber_demo_before_chain.mp4 \
  --chain fiberaa.github.io/static/videos/sim/chain.mp4 \
  --output Assets/demo_edits/fiber_demo_with_chain.mp4
```

Keep the base outside the website repository. Do not use the already edited
published overview as the base: that would insert the segment twice.

The historical full-generation scripts also exist locally under
`Assets/icra_demo/render_icra_demo_v{1,5}.sh`; they produce the September demo,
not the current October overview.

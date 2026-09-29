"""Apply three box sizes after exporting the isolated pickup scenes.

Usage: python3 resize_pickup_cases.py playground/models/scenes/pickup
"""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    folder = parser.parse_args().folder
    # Full dimensions in metres; medium retains the authentic low-support size.
    sizes = [
        ("Small", [.17, .16, .215]),
        ("Medium", [.185, .272, .275]),
        ("Large", [.24, .32, .34]),
    ]
    manifest = json.loads((folder / "cases.json").read_text())
    for items in manifest["layouts"].values():
        for index, item in enumerate(items):
            label, dimensions = sizes[index]
            cfg_path = folder / item["config"]
            cfg = json.loads(cfg_path.read_text())
            tree = ET.parse(folder / cfg["xml"])
            geom = tree.find(".//geom[@name='pickup_object_box']")
            half = [v / 2 for v in dimensions]
            geom.set("size", " ".join(map(str, half)))
            # The free body origin stays at the bottom to preserve surface placement.
            geom.set("pos", f"0 0 {half[2]}")
            geom.set("mass", ".5")
            geom.set("condim", "4")
            # The large box needs a platform that covers its rotated footprint.
            if index == 2 and item.get("surface") == "support":
                support = tree.find(".//geom[@name='pickup_support']")
                support_half = [0.25, 0.30, float(support.get("size").split()[2])]
                support.set("size", " ".join(map(str, support_half)))
                support_visual = next(v for v in cfg["visuals"] if v.get("name") == "pickup_support")
                support_visual["size"] = support_half
                cfg["provenance"]["support_dimensions_m"] = [v * 2 for v in support_half]
            tree.write(folder / cfg["xml"], encoding="unicode")
            visual = next(v for v in cfg["visuals"] if v["id"] == cfg["object_geom"])
            visual["size"] = half
            q = cfg["qpos"]
            w, x, y, z = q[3:7]
            yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
            cfg["pickup_approach"] = {"position": q[:2], "yaw": yaw}
            cfg["pickup_spawn"] = {
                "radius_min": .9, "radius_max": 1.5,
                "sector_degrees": 65, "heading_offset_degrees": 60,
            }
            cfg["boxes"][0].update(size_label=label, dimensions=dimensions)
            cfg["provenance"].update(
                box_dimensions_m=dimensions, box_size=label,
                spawn="Random 0.9–1.5 m radius with varied side and heading",
            )
            cfg_path.write_text(json.dumps(cfg, separators=(",", ":")))
            item.update(size_label=label, dimensions=dimensions)
    (folder / "cases.json").write_text(json.dumps(manifest, separators=(",", ":")))
    print("Sized nine selections: Small 17×16×21.5, Medium 18.5×27.2×27.5, Large 24×32×34 cm")


if __name__ == "__main__":
    main()

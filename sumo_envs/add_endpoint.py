from pathlib import Path

file_path = Path("/sumo_envs/LONG_GANG/env/osm.rou.xml")
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# add route path here
target_prefixes = [
    '<route edges="1125684496#0 1125684496#1 1125597092#1',
    #'<route edges="1125695392#0'
]

modified_lines = []
for line in lines:
    line_strip = line.strip()
    replaced = False
    for prefix in target_prefixes:
        if line_strip.startswith(prefix):
            indent = line[:line.find('<')]
            modified_lines.append(f'{indent}<route edges="{prefix[14:]}" />\n')
            replaced = True
            break
    if not replaced:
        modified_lines.append(line)

output_path = "/sumo_envs/LONG_GANG/env/osm.rou.xml"
with open(output_path, "w", encoding="utf-8") as f:
    f.writelines(modified_lines)

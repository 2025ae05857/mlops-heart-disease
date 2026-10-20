import re

with open("report.md", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Remove any trailing or duplicate "Visual Artifacts & Figures" sections at the bottom
content = re.sub(r"\n+## Visual Artifacts & Figures.*", "", content, flags=re.DOTALL)

# 2. Clean up any duplicated figure lines throughout the file
clean_lines = []
seen_images = set()

for line in content.splitlines():
    # Detect markdown image lines
    if line.strip().startswith("!["):
        img_match = re.search(r"\((.*?)\)", line)
        if img_match:
            img_path = img_match.group(1)
            if img_path in seen_images:
                continue  # Skip duplicate image embed
            seen_images.add(img_path)
    clean_lines.append(line)

fixed_content = "\n".join(clean_lines)

with open("report.md", "w", encoding="utf-8") as f:
    f.write(fixed_content)

print("SUCCESS: Removed all duplicate figures from report.md!")

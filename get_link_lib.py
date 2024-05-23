from pathlib import Path
with open("out.log") as f:
    lines = f.readlines()

link_list = []
link_dirs = []
link_dict = {}
for l in lines:
    l_list = l.split()
    link_files = [Path(l_i) for l_i in l_list if l_i.endswith(".a")]
    for p in link_files:
        k = str(p.parent)
        if k not in link_dict:
            link_dict[k] = set()
        link_dict[k].add(p.name)
print(link_dict)

for k, v in link_dict.items():
    print(k)
    for v_i in v:
        print(v_i)
    



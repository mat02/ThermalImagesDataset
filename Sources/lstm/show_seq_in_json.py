import json

folds = 0, 5
files = [
    'train_{}.json',
    'val_{}.json',
]

with open('folds_2.txt', 'w') as out:

    for k in range(folds[0], folds[1]):
        for f in files:
            f = f.format(k)
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
            except:
                continue
            
            sequences = set()
            for s in data['normal']:
                name = s['path']
                sequences.add(name)
            
            out.write(f"Sequences in file {f}:\n")
            out.writelines([s+'\n' for s in sequences])
            out.write("\n")

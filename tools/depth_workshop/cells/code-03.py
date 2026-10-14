# @title Infrastructure: write and verify the carried source (about 290 KB; no need to read it)
CARRIED_FILES = @@CARRIED_FILES@@
CARRIED_HASHES = @@CARRIED_HASHES@@
for name, text in CARRIED_FILES.items():
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    if hashlib.sha256(path.read_bytes()).hexdigest() != CARRIED_HASHES[name]:
        raise RuntimeError('Carried source integrity failure: ' + name)
print('Separate Depth Anything and ZoeDepth source namespaces materialized and verified.')
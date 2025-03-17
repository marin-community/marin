import matplotlib.font_manager
flist = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

font_names = []

for fname in flist:
    try:
        print(matplotlib.font_manager.FontProperties(fname=fname).get_name())
        font_names.append(matplotlib.font_manager.FontProperties(fname=fname).get_name())
    except:
        print(f"Error with {fname}")

print(sorted(set(font_names)))
import os
import shutil

root = os.path.dirname(os.path.realpath(__file__))
### we form 1623 partitions of characters (regardless of the alphabets);
### and each task is composed of 5 characters (classes) sampled from the 1623 tasks, labeled from 0-4
### the task for meta-training is sampled from the first 1200 characters;
### and the task for meta-testing is sampled from the last 423 characters.

### first rename the data dir to 50_alphabets
### then run 
idx = 0
for f in os.listdir("raw/50_alphabets"):
    print(f)
    for character_class in os.listdir(os.path.join("raw/50_alphabets", f)):
        character_images = os.listdir("raw/50_alphabets/" + f + "/" + character_class)
        os.system("mkdir raw/1623_characters/"+str(idx))
        # os.system("cp raw/50_alphabets/"+f+"/"+ re.sub(r'\)',r'\)', re.sub(r'\(', r'\(', character_class))+"/*"+" raw/1623_characters/"+str(idx))
        # 由于 class 有 ( )
        for png_file in os.listdir(os.sep.join(('raw', '50_alphabets', f, character_class))):
            if png_file.endswith('.png'):
                shutil.copy2(os.sep.join(('raw', '50_alphabets', f, character_class, png_file)), "raw/1623_characters/" + str(idx))
        idx += 1
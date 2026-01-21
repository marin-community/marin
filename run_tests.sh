 uv run pytest lib/levanter/tests/test_train_image_anyres.py -vs > out1.txt 2>&1
 uv run pytest lib/levanter/tests/test_train_image.py -vs > out12.txt 2>&1 
uv run pytest lib/levanter/tests/test_llava_onevision.py -vs > out2.txt 2>&1
uv run pytest lib/levanter/tests/test_llava_onevision_wo_anyres.py -vs > out3.txt 2>&1
uv run pytest lib/levanter/tests/test_image.py -vs > out4.txt 2>&1
uv run pytest lib/levanter/tests/test_image_anyres.py -vs > out5.txt 2>&1
Use under BSD Lisence.
Run python sentiBank.py, it will explain itself. CPU/GPU are both supported.
You can run the example image by:
python sentiBank.py test_image.jpg

The output should be a json file, containing 2,089 ranked concept scores, and a 4096-dimension feature (fc7).

Under windows 7+, you can probably run above command directly. Otherwise you may need to compile Caffe. 
Before compiling, put the extract_nfeatures.cpp under caffe/tools. After compiling, copy/link caffe/build/tools/extract_nfeatures.bin or exe to DeepSentiBank folder.
There is also a .m file that can read the raw feature file (fc7.dat prob.dat) generated from the executable extract_nfeatures. into matlab.

## Note:

-(1) Due to the size limit, caffe_sentibank_train_iter_25000 should be downloaded from the sentibank website. 

-(2) libopenblas.dll should be unzipped. 

## `Overall steps:`

- (i) Use gen_path_file.py to generate path file: path_file.txt
- (ii) Use sentiBank.py to obtain ANP json file: path_file.json
- (iii) Use jsonparsing_top10.py to extract top 10 ANPs for each image


Please cite:
@article{chen2014deepsentibank,
  title={Deepsentibank: Visual sentiment concept classification with deep convolutional neural networks},
  author={Chen, Tao and Borth, Damian and Darrell, Trevor and Chang, Shih-Fu},
  journal={arXiv preprint arXiv:1410.8586},
  year={2014}
}

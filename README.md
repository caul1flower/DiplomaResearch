# DiplomaResearch

You may install the dependencies using the following command:
> pip install -r requirements.txt

You can download the **dataset** currently used in the evaluation script from the following link:
[NYU-v2](https://drive.google.com/file/d/1osYRaDfMYuyiTkJwDbKl3kHwyevDLsZf/view?usp=sharing)

All original **pretrained models** can be found here: <a href="https://drive.google.com/drive/folders/17mCRfsNj0f_BNY3viHcR6M1camCVoAb8?usp=sharing">here</a>.

To evaluate the model, **specify the paths** to the dataset and pretrained model while running the script. 

Additionally, you can adjust parameters such as --scale and --num_feats. 
_Note: These parameters must match the ones used during the training of the pretrained model, so be careful. (~~not me spending a few hours to figure that out~~_)

Example command:
> python evaluation_script.py --scale 16 --num_feats 40 --root_dir nyu_data --model_dir model/SGNet_X16_R.pth

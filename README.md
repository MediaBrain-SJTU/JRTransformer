# Joint-Relation Transformer for Multi-Person Motion Prediction
**Abstract**: Multi-person motion prediction is a challenging problem due to the dependency of motion on both individual past movements and interactions with other people. Transformer-based methods have shown promising results on this task, but they miss the explicit relation representation between joints, such as skeleton structure and pairwise distance, which is crucial for accurate interaction modeling. In this paper, we propose the Joint-Relation Transformer, which utilizes relation information to enhance interaction modeling and improve future motion prediction. Our relation information contains the relative distance and the intra-/inter-person physical constraints. To fuse relation and joint information, we design a novel joint-relation fusion layer with relation-aware attention to update both features. Additionally, we supervise the relation information by forecasting future distance. Experimental results on four multi-person motion prediction datasets demonstrate that our proposed method achieves state-of-the-art or comparable performance. 

The repo is still in process.

## Get Start
### Environment
- python==3.10
- matplotlib==3.5.1
- numpy==1.22.3
- scipy==1.7.3
- torch==1.12.1
- transformers==4.18.0

### File Preparation 
- you can download the preprocessed 3dpw data `poseDate.pkl` form [Google Drive](https://drive.google.com/file/d/1tatpBjQ1rUyJ6NT5vsjmGOROqa9dw4l8/view?usp=drive_link) and put it in `data` directory.
- If you want to pretrain on AMASS dataset, please download from it's [website](https://amass.is.tue.mpg.de/index.html).
- We provide the trained model on 3dpw, you can download it from [Google Drive](https://drive.google.com/file/d/1W354xCv-q9C2cIADm4Obt8P1RkaUKKmQ/view?usp=drive_link) and put it in `output` directory.

## Test
We provide the evaluation code in `test_3dpw.py`.

You can run
```
python test_3dpw.py 
```
to get the result.

## Train
For training on 3DPW-SoMoF/RC dataset, We recommend to first pretrain on AMASS dataset with  
```
python pretrain_amass.py 
```
and then finetune on the 3DPW-SoMoF/RC dataset with 
```
python train_3dpw.py --pretrain_path ./path/to/pretrained/model.pt
```

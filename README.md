# Deep Attentive Features for Prostate Segmentation in 3D Transrectal Ultrasound. 
This repository is the implementation for the DAF3D by Haoran Dou in Shenzhen University  

[Deep Attentive Features for Prostate Segmentation in 3D Transrectal Ultrasound.](https://arxiv.org/abs/1907.01743)   
*Yi Wang, Haoran Dou, Xiaowei Hu, Lei Zhu, Xin Yang, Ming Xu, Jing Qin, Pheng-Ann Heng, Tianfu Wang, and Dong Ni.*  
IEEE Transactions on Medical Imaging(**IEEE TMI**), 2019.  
  
![framwork](img/framework.png)  

> Automatic prostate segmentation in transrectal ultrasound (TRUS) images is of essential importance for image-guided prostate interventions and treatment planning. However, developing such automatic solutions remains very challenging due to the missing/ambiguous boundary and inhomogeneous intensity distribution of the prostate in TRUS, as well as the large variability in prostate shapes. This paper develops a novel 3D deep neural network equipped with attention modules for better prostate segmentation in TRUS by fully exploiting the complementary information encoded in different layers of the convolutional neural network (CNN). Our attention module utilizes the attention mechanism to selectively leverage the multilevel features integrated from different layers to refine the features at each individual layer, suppressing the non-prostate noise at shallow layers of the CNN and increasing more prostate details into features at deep layers. Experimental results on challenging 3D TRUS volumes show that our method attains satisfactory segmentation performance. The proposed attention mechanism is a general strategy to aggregate multi-level deep features and has the potential to be used for other medical image segmentation tasks.

## Usage  
### Dependencies  
This work depends on the following libraries:  
Pytorch == 0.4.0  
Python == 3.6  

### Train and Validate
Run  
```
python Train.py
```
You can rewrite the DataOprate.py to train your own data.

## Result
One example to illustrate the effectiveness of the proposed attention module for the feature refinement.  
![result](img/attentionresult.png)    
metric results  
  
| Metric    | 3D FCN | 3D U-Net | Ours   |
| ------    | ------ | ------   | ------ |
| Dice      | 0.8210 | 0.8453   | 0.9004 |
| Jaccard   | 0.6985 | 0.7340   | 0.8200 |
| CC        | 0.5579 | 0.6293   | 0.7762 |
| ADB       | 9.5801 | 8.2715   | 3.3198 |
| 95HD      | 25.113 | 20.390   | 8.3684 |
| Precision | 0.8105 | 0.8283   | 0.8995 |
| Recall    | 0.8486 | 0.8764   | 0.9055 |
  
## Citation  
If this work is helpful for you, please cite our paper as follow:   
```
@article{wang2019deep,
  title={Deep attentive features for prostate segmentation in 3d transrectal ultrasound},
  author={Wang, Yi and Dou, Haoran and Hu, Xiaowei and Zhu, Lei and Yang, Xin and Xu, Ming and Qin, Jing and Heng, Pheng-Ann and Wang, Tianfu and Ni, Dong},
  journal={IEEE transactions on medical imaging},
  volume={38},
  number={12},
  pages={2768--2778},
  year={2019},
  publisher={IEEE}
}
```

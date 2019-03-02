# gradCAM, guidedBackProp, smoothGrad on pytorch
Pytorch implementation of gradCAM, guidedBackProp, smoothGrad

Resnet50 is used in this implementation (also other models can be used)

* gradCAM
https://arxiv.org/pdf/1610.02391.pdf
* guidedBackProp
https://arxiv.org/pdf/1412.6806.pdf
* smoothGrad
https://arxiv.org/pdf/1706.03825.pdf

## dependencies
* Python 3.6
* torch 1.0.0
* torchvison 0.2.1
* cv2
* tqdm


## usage
install requirements
```
pip install -r requirements.txt
```
execute script
```
python main.py image_path --cuda --index hoge
```

* add cuda option to use gpu
* add index option to specify the target imagenet index of gradCAM

  Please see imagenet_class_index.json (e.g. ostrich: 9, tusker: 101)
  
  If not specified, target index will be estimated by the model

## examples
||ostrich.jpg|elephant.jpg|
|---|---|---|
|raw|<img src="https://i.imgur.com/7bXQsgJ.jpg" width="300">|<img src="https://i.imgur.com/K8uSY69.jpg" width="300">|
|gradCAM|<img src="https://i.imgur.com/DP8O67J.png" width=300>|<img src="https://i.imgur.com/s5scZJV.png" width=300>|
|guidedBackProp|<img src="https://i.imgur.com/zwAlC7D.png" width=300>|<img src="https://i.imgur.com/Am1Pp0q.png" width=300>|
|guidedGradCAM|<img src="https://i.imgur.com/4GQa0Lw.png" width=300>|<img src="https://i.imgur.com/xwRqYNm.png" width=300>|
|smoothGrad|<img src="https://i.imgur.com/PqnI1mL.png" width=300>|<img src="https://i.imgur.com/hzAtgpM.png" width=300>|

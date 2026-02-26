# CNN (TextCNN) — training and prediction


# training (default save to output/checkpoints/model.pt)
python CNN/train.py --quick

or use argument --save ../outputs/checkpoints/modelX.pt

# predict
python CNN/predict.py -c CNN/checkpoints/model.pt "some text here"






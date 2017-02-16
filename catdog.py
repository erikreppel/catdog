from classifier import VGGImgClassifier


DATA_HOME = '/media/erik/Spining/dev/datasets/cat_dog'

model = VGGImgClassifier()

model.learn_dataset(DATA_HOME, nb_epoch=50, save_model_name='./models/cat_dog3.h5')



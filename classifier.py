import os

from keras.optimizers import Adam
from keras.models import load_model
from keras.layers.core import Dense
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from vgg16 import create_vgg16


class VGGImgClassifier(object):
    '''
    Wrapper to aid with using VGG16 for other image classification problems.
    '''

    def __init__(self):
        self.model = create_vgg16()


    def adjust_model(self, nb_classes, trainable=0):
        '''
        Adjusts VGG16 for a differente number of classes.
        :params - nb_classes: how many target classes there are
                - trainable: how many of the hidden layers to keep training
                  (0 indicates to only train the weights of the new final layer)
        '''
        self.model.pop()
        layers = self.model.layers
        for layer in layers[:len(layers)-trainable]:
            layer.trainable = False
        self.model.add(Dense(nb_classes, activation='softmax'))
        self.compile()

    def compile(self, lr=0.0001):
        self.model.compile(optimizer=Adam(lr=lr),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def create_batches(self,
                       folderpath,
                       gen=image.ImageDataGenerator(),
                       shuffle=True,
                       batch_size=8,
                       target_size=(224,224),
                       class_mode='categorical'):

        return gen.flow_from_directory(folderpath,
                                       target_size=target_size,
                                       class_mode=class_mode,
                                       shuffle=shuffle,
                                       batch_size=batch_size)

    def fit(self, train_batches, validation_batches, nb_epoch=1, callbacks=[]):
        self.model.fit_generator(train_batches,
                                 samples_per_epoch=train_batches.nb_sample,
                                 nb_epoch=nb_epoch,
                                 validation_data=validation_batches,
                                 nb_val_samples=validation_batches.nb_sample,
                                 callbacks=callbacks)

    def learn_dataset(self, folderpath, nb_epoch=1, save_model_name=None, checkpoint="./models/ckpnt.h5", start_from_checkpoint=None):
        '''
        folderpath should point to a folder that has the follow subdirectory structure.

        Where the folders class 1 and class 2 contain images
        '''

        train_path = os.path.join(folderpath, 'train')
        validate_path = os.path.join(folderpath, 'validate')

        train_batches = self.create_batches(train_path)
        validate_batches = self.create_batches(validate_path)

        if start_from_checkpoint:
            self.model = load_model(start_from_checkpoint)
        else:
            self.adjust_model(train_batches.nb_class, 7)

        callbacks = [
            ModelCheckpoint(checkpoint, monitor='val_acc', save_best_only=True, save_weights_only=False, verbose=True),
            CSVLogger('log.csv', separator=',', append=False),
            EarlyStopping(monitor='val_acc', patience=2)
        ]

        self.fit(train_batches, validate_batches, nb_epoch=nb_epoch, callbacks=callbacks)

        if save_model_name:
            self.save_model(save_model_name)

    def save_model(self, filepath):
        self.model.save(filepath)
    
    def load_model(self, filepath):
        return load_model(filepath)

    def test(self, path, batch_size=32):
        test_batches = self.create_batches(path,
                                           shuffle=False,
                                           batch_size=batch_size,
                                           class_mode=None)
        return test_batches, self.model.predict_generator(test_batches,
                                                          test_batches.nb_sample)

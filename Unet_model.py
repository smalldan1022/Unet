import tensorflow as tf 
from tensorflow.keras.layers import Dense, MaxPool2D, Input, Conv2D, UpSampling2D, Concatenate


class Unet(tf.keras.Model):

    def __init__(self):

        super(Unet, self).__init__()
        
        # Make the first conv layers of Unet

        self.conv_input = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape = (512, 512, 3), data_format="channels_last", 
                                                 activation='relu', use_bias=True,
                                                 kernel_initializer='glorot_uniform')
        
        # Make the forward conv layers of Unet

        self.conv_64 = []
        self.conv_128 = []
        self.conv_256 = []
        self.conv_512 = []
        self.conv_1024 = []

        for ii in range(3):
            self.conv_64.append(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform'))

        for ii in range(4):
            self.conv_128.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform')) 
            self.conv_256.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform'))
            self.conv_512.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform'))
        
        for ii in range(2):
            self.conv_1024.append(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform'))

        # Make others layers that won't be an error when it is used repeatedly

        self.Maxpool = tf.keras.layers.MaxPool2D()
        self.upsampling2D = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.Concatenate = tf.keras.layers.Concatenate(axis=3)

        # Make up conv layers

        self.up_conv_512 = tf.keras.layers.Conv2D(filters=512, kernel_size=(2,2), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
        self.up_conv_256 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2,2), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
        self.up_conv_128 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2,2), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
        self.up_conv_64 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

        # Make the final conv layer

        self.final_conv_1 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1,1), padding='same', activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform')
    
    def call(self, inputs, training=False):

        one = self.conv_input(inputs)
        one = self.conv_64[0](one)     
        two = self.Maxpool(one)

        two = self.conv_128[0](two)
        two = self.conv_128[1](two)
        three = self.Maxpool(two)

        three = self.conv_256[0](three)
        three = self.conv_256[1](three)
        four = self.Maxpool(three)

        four = self.conv_512[0](four)
        four = self.conv_512[1](four)
        five = self.Maxpool(four)

        five = self.conv_1024[0](five)
        five = self.conv_1024[1](five)
        up_four = self.upsampling2D(five)
        up_four = self.up_conv_512(up_four)

        concat_four = self.Concatenate([up_four, four])
        concat_four = self.conv_512[2](concat_four)
        concat_four = self.conv_512[3](concat_four)
        up_three = self.upsampling2D(concat_four)
        up_three = self.up_conv_256(up_three)

        concat_three = self.Concatenate([up_three, three])
        concat_three = self.conv_256[2](concat_three)
        concat_three = self.conv_256[3](concat_three)
        up_two = self.upsampling2D(concat_three)
        up_two = self.up_conv_128(up_two)

        concat_two = self.Concatenate([up_two, two])
        concat_two = self.conv_128[2](concat_two)
        concat_two = self.conv_128[3](concat_two)
        up_one = self.upsampling2D(concat_two)
        up_one = self.up_conv_64(up_one)

        concat_one = self.Concatenate([up_one, one])
        concat_one = self.conv_64[1](concat_one)
        concat_one = self.conv_64[2](concat_one)
        output = self.final_conv_1(concat_one)
        
        return output
